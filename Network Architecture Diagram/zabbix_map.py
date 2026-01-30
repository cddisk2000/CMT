import os
import platform
import re
import subprocess
import sys
import json
import sqlite3
import socket
import time
from typing import Dict, List, Tuple

import streamlit as st
import streamlit.components.v1 as components

from Alert.wx_webhook import send_webhook
from Alert.e_mail import send_email

try:
    from ping3 import ping as ping_once
    HAS_PING3 = True
except Exception:
    HAS_PING3 = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "zabbix_map.db")
COMPONENT_DIR = os.path.join(BASE_DIR, "components", "zabbix_map_component")
MAP_COMPONENT = components.declare_component("zabbix_map_component", path=COMPONENT_DIR)

DAEMON_PID_PATH = os.path.join(DATA_DIR, "ping_daemon.pid")
DAEMON_STOP_PATH = os.path.join(DATA_DIR, "ping_daemon.stop")
DAEMON_HB_PATH = os.path.join(DATA_DIR, "ping_daemon.hb")

try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except Exception:
    HAS_AUTOREFRESH = False

NODE_TYPES = [
    "router",
    "firewall",
    "switch",
    "core_switch",
    "server",
    "vm",
    "storage",
    "wireless_ap",
    "cloud",
    "cloud_vps",
    "domain",
    "text",
]

FLOOR_OPTIONS = ["B3", "B2", "B1", "1F", "2F", "3F", "4F", "5F"]

TYPE_SHAPES = {
    "router": "triangle",
    "firewall": "diamond",
    "switch": "rectangle",
    "server": "round-rectangle",
    "vm": "round-rectangle",
    "storage": "hexagon",
    "wireless_ap": "ellipse",
    "domain": "ellipse",
    "text": "round-rectangle",
}

STATUS_COLORS = {
    "up": "#f97316",
    "degraded": "#fbbf24",
    "down": "#ef4444",
    "unknown": "#6b7280",
}


def ensure_db():
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS nodes (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            ip TEXT,
            node_type TEXT NOT NULL,
            site TEXT,
            floor TEXT,
            line TEXT,
            pos_x REAL,
            pos_y REAL,
            last_status TEXT,
            last_rtt_ms REAL,
            last_loss REAL,
            last_seen REAL
        )
        """
    )
    # Add size columns if missing (for node resizing).
    cur.execute("PRAGMA table_info(nodes)")
    cols = {row[1] for row in cur.fetchall()}
    if "size_w" not in cols:
        cur.execute("ALTER TABLE nodes ADD COLUMN size_w REAL")
    if "size_h" not in cols:
        cur.execute("ALTER TABLE nodes ADD COLUMN size_h REAL")
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS edges (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            target TEXT NOT NULL,
            status TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ping_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            node_id TEXT NOT NULL,
            ts REAL NOT NULL,
            rtt_ms REAL,
            loss REAL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS alert_state (
            node_id TEXT PRIMARY KEY,
            down_count INTEGER NOT NULL,
            last_alert_ts REAL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS alert_config (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            down_threshold INTEGER NOT NULL,
            cooldown_sec INTEGER NOT NULL,
            webhook_url TEXT,
            email_to TEXT,
            email_from TEXT,
            smtp_host TEXT,
            smtp_port INTEGER,
            smtp_user TEXT,
            smtp_pass TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """
    )
    cur.execute("SELECT COUNT(*) FROM alert_config")
    if cur.fetchone()[0] == 0:
        cur.execute(
            """
            INSERT INTO alert_config
            (id, down_threshold, cooldown_sec, webhook_url)
            VALUES (1, 3, 300, '')
            """
        )
    cur.execute("SELECT COUNT(*) FROM settings WHERE key = 'ping_retention_days'")
    if cur.fetchone()[0] == 0:
        cur.execute(
            """
            INSERT INTO settings (key, value)
            VALUES ('ping_retention_days', '30')
            """
        )
    conn.commit()
    conn.close()


def seed_if_empty():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM nodes")
    if cur.fetchone()[0] == 0:
        sample_nodes = [
            ("r1", "Core Router", "10.0.0.1", "router", "Plant-A", "1F", "Line-1", 100, 120),
            ("fw1", "Edge FW", "10.0.0.2", "firewall", "Plant-A", "1F", "Line-1", 280, 120),
            ("sw1", "Access SW", "10.0.0.3", "switch", "Plant-A", "1F", "Line-1", 440, 120),
            ("srv1", "APP Server", "10.0.1.10", "server", "Plant-A", "1F", "Line-1", 620, 120),
            ("vm1", "VM Host", "10.0.1.20", "vm", "Plant-A", "1F", "Line-1", 620, 240),
            ("st1", "Storage", "10.0.2.10", "storage", "Plant-A", "1F", "Line-1", 440, 240),
            ("ap1", "WiFi AP", "10.0.3.10", "wireless_ap", "Plant-A", "1F", "Line-1", 280, 240),
        ]
        cur.executemany(
            """
            INSERT INTO nodes
            (id, name, ip, node_type, site, floor, line, pos_x, pos_y, last_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'unknown')
            """,
            sample_nodes,
        )
        sample_edges = [
            ("e1", "r1", "fw1", "unknown"),
            ("e2", "fw1", "sw1", "unknown"),
            ("e3", "sw1", "srv1", "unknown"),
            ("e4", "sw1", "vm1", "unknown"),
            ("e5", "sw1", "st1", "unknown"),
            ("e6", "sw1", "ap1", "unknown"),
        ]
        cur.executemany(
            "INSERT INTO edges (id, source, target, status) VALUES (?, ?, ?, ?)",
            sample_edges,
        )
    conn.commit()
    conn.close()


def fetch_nodes(filters: Dict[str, str]) -> List[Dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    clauses = []
    values = []
    for key in ["site", "floor", "line"]:
        val = filters.get(key)
        if val and val != "All":
            clauses.append(f"{key} = ?")
            values.append(val)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    cur.execute(f"SELECT * FROM nodes {where}", values)
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows


def fetch_edges() -> List[Dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM edges")
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows


def get_group_options(field: str) -> List[str]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(f"SELECT DISTINCT {field} FROM nodes WHERE {field} IS NOT NULL")
    values = sorted({row[0] for row in cur.fetchall() if row[0]})
    conn.close()
    return ["All"] + values


def upsert_node(node: Dict):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO nodes
        (id, name, ip, node_type, site, floor, line, pos_x, pos_y, size_w, size_h, last_status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            name=excluded.name,
            ip=excluded.ip,
            node_type=excluded.node_type,
            site=excluded.site,
            floor=excluded.floor,
            line=excluded.line,
            pos_x=excluded.pos_x,
            pos_y=excluded.pos_y,
            size_w=excluded.size_w,
            size_h=excluded.size_h
        """,
        (
            node["id"],
            node["name"],
            node.get("ip", ""),
            node["node_type"],
            node.get("site"),
            node.get("floor"),
            node.get("line"),
            node.get("pos_x", 0),
            node.get("pos_y", 0),
            node.get("size_w"),
            node.get("size_h"),
            node.get("last_status", "unknown"),
        ),
    )
    conn.commit()
    conn.close()


def update_node_position(node_id: str, pos_x: float, pos_y: float):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE nodes
        SET pos_x=?, pos_y=?
        WHERE id=?
        """,
        (pos_x, pos_y, node_id),
    )
    conn.commit()
    conn.close()


def update_node_size(node_id: str, size_w: float, size_h: float):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE nodes
        SET size_w=?, size_h=?
        WHERE id=?
        """,
        (size_w, size_h, node_id),
    )
    conn.commit()
    conn.close()


def update_node_ip(node_id: str, ip: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE nodes
        SET ip=?
        WHERE id=?
        """,
        (ip, node_id),
    )
    conn.commit()
    conn.close()


def resolve_domain(domain: str) -> str:
    if not domain:
        return ""
    try:
        return socket.gethostbyname(domain)
    except Exception:
        return ""


def snap_value(value: float, grid_size: int) -> float:
    if grid_size <= 0:
        return value
    return round(value / grid_size) * grid_size


def align_all_to_grid(grid_size: int):
    if grid_size <= 0:
        return
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, pos_x, pos_y FROM nodes")
    rows = cur.fetchall()
    for node_id, x, y in rows:
        new_x = snap_value(x or 0, grid_size)
        new_y = snap_value(y or 0, grid_size)
        cur.execute(
            "UPDATE nodes SET pos_x=?, pos_y=? WHERE id=?",
            (new_x, new_y, node_id),
        )
    conn.commit()
    conn.close()


def upsert_edge(edge: Dict):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO edges (id, source, target, status)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            source=excluded.source,
            target=excluded.target,
            status=excluded.status
        """,
        (edge["id"], edge["source"], edge["target"], edge.get("status", "unknown")),
    )
    conn.commit()
    conn.close()


def record_ping(node_id: str, rtt_ms: float, loss: float, status: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    ts = time.time()
    cur.execute(
        """
        INSERT INTO ping_samples (node_id, ts, rtt_ms, loss)
        VALUES (?, ?, ?, ?)
        """,
        (node_id, ts, rtt_ms, loss),
    )
    cur.execute(
        """
        UPDATE nodes
        SET last_status=?, last_rtt_ms=?, last_loss=?, last_seen=?
        WHERE id=?
        """,
        (status, rtt_ms, loss, ts, node_id),
    )
    conn.commit()
    conn.close()


def update_alert_state(node_id: str, is_down: bool, threshold: int, cooldown: int) -> Tuple[bool, int, bool]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT down_count, last_alert_ts FROM alert_state WHERE node_id = ?", (node_id,))
    row = cur.fetchone()
    down_count = row[0] if row else 0
    last_alert_ts = row[1] if row else None
    was_down = down_count > 0

    if is_down:
        down_count += 1
    else:
        down_count = 0

    now = time.time()
    should_alert = False
    if is_down and down_count >= threshold:
        if last_alert_ts is None or (now - last_alert_ts) >= cooldown:
            should_alert = True
            last_alert_ts = now

    cur.execute(
        """
        INSERT INTO alert_state (node_id, down_count, last_alert_ts)
        VALUES (?, ?, ?)
        ON CONFLICT(node_id) DO UPDATE SET
            down_count=excluded.down_count,
            last_alert_ts=excluded.last_alert_ts
        """,
        (node_id, down_count, last_alert_ts),
    )
    conn.commit()
    conn.close()
    recovered = was_down and not is_down
    return should_alert, down_count, recovered


def load_alert_config() -> Dict:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM alert_config WHERE id = 1")
    row = dict(cur.fetchone())
    conn.close()
    return row


def save_alert_config(cfg: Dict):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE alert_config
        SET down_threshold=?, cooldown_sec=?, webhook_url=?, email_to=?, email_from=?,
            smtp_host=?, smtp_port=?, smtp_user=?, smtp_pass=?
        WHERE id = 1
        """,
        (
            cfg["down_threshold"],
            cfg["cooldown_sec"],
            cfg.get("webhook_url", ""),
            cfg.get("email_to", ""),
            cfg.get("email_from", ""),
            cfg.get("smtp_host", ""),
            cfg.get("smtp_port", 587),
            cfg.get("smtp_user", ""),
            cfg.get("smtp_pass", ""),
        ),
    )
    conn.commit()
    conn.close()


def load_setting(key: str, default: str = "") -> str:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT value FROM settings WHERE key = ?", (key,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row and row[0] is not None else default


def save_setting(key: str, value: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO settings (key, value)
        VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """,
        (key, value),
    )
    conn.commit()
    conn.close()


def cleanup_ping_samples(retention_days: int):
    if retention_days <= 0:
        return
    cutoff = time.time() - (retention_days * 86400)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM ping_samples WHERE ts < ?", (cutoff,))
    conn.commit()
    conn.close()


def clear_all_ping_samples():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM ping_samples")
    conn.commit()
    conn.close()




def build_elements(nodes: List[Dict], edges: List[Dict], group_mode: str) -> List[Dict]:
    elements = []
    group_nodes = {}
    if group_mode != "None":
        for node in nodes:
            site = node.get("site") or "Site"
            floor = node.get("floor") or "Floor"
            line = node.get("line") or "Line"
            site_id = f"group:{site}"
            floor_id = f"group:{site}:{floor}"
            line_id = f"group:{site}:{floor}:{line}"
            if group_mode in {"Site", "Site > Floor", "Site > Floor > Line"}:
                group_nodes[site_id] = {"id": site_id, "label": site}
                node["parent"] = site_id
            if group_mode in {"Site > Floor", "Site > Floor > Line"}:
                group_nodes[floor_id] = {"id": floor_id, "label": floor, "parent": site_id}
                node["parent"] = floor_id
            if group_mode == "Site > Floor > Line":
                group_nodes[line_id] = {"id": line_id, "label": line, "parent": floor_id}
                node["parent"] = line_id

        for gid, gdata in group_nodes.items():
            elements.append({"data": {"id": gid, "label": gdata["label"], "parent": gdata.get("parent")}})

    for node in nodes:
        node_type = node.get("node_type", "")
        status = node.get("last_status") or "unknown"
        rtt = node.get("last_rtt_ms")
        loss = node.get("last_loss")
        rtt_text = f"{rtt:.1f}ms" if rtt is not None else "--"
        loss_text = f"{loss:.0f}%" if loss is not None else "--"
        ip_text = node.get("ip") or "--"
        avg_rtt = fetch_avg_rtt(node["id"], limit=5)
        avg_text = f"{avg_rtt:.1f}ms" if avg_rtt is not None else "--"
        if node_type == "text":
            label = f"{node['name']}"
        elif node_type == "domain":
            label = f"{node['name']}\n{ip_text}\n{status} {rtt_text} {loss_text}\navg {avg_text}"
        else:
            label = f"{node['name']}\n{ip_text}\n{status} {rtt_text} {loss_text}\navg {avg_text}"
        elements.append(
            {
                "data": {
                    "id": node["id"],
                    "label": label,
                    "status": status,
                    "color": STATUS_COLORS.get(status, STATUS_COLORS["unknown"]),
                    "shape": TYPE_SHAPES.get(node_type, "ellipse"),
                    "node_type": node_type,
                    "parent": node.get("parent"),
                    "size_w": node.get("size_w"),
                    "size_h": node.get("size_h"),
                },
                "position": {"x": node.get("pos_x", 0), "y": node.get("pos_y", 0)},
            }
        )

    for edge in edges:
        status = edge.get("status") or "unknown"
        elements.append(
            {
                "data": {
                    "id": edge["id"],
                    "source": edge["source"],
                    "target": edge["target"],
                    "color": STATUS_COLORS.get(status, STATUS_COLORS["unknown"]),
                }
            }
        )
    return elements


def _ping_system(ip: str, timeout_sec: int = 1) -> Tuple[bool, float]:
    if platform.system().lower().startswith("win"):
        cmd = ["ping", "-n", "1", "-w", str(timeout_sec * 1000), ip]
        rtt_re = re.compile(r"(?:time|时间|時間)[=<]\s*([\d.]+)\s*ms", re.IGNORECASE)
    else:
        cmd = ["ping", "-c", "1", "-W", str(timeout_sec), ip]
        rtt_re = re.compile(r"time[=<]?\s*([\d.]+)\s*ms", re.IGNORECASE)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec + 1)
        ok = proc.returncode == 0
        rtt = None
        match = rtt_re.search(proc.stdout)
        if match:
            rtt = float(match.group(1))
        return ok, rtt if rtt is not None else None
    except Exception:
        return False, None


def _read_daemon_hb() -> Dict:
    if not os.path.exists(DAEMON_HB_PATH):
        return {}
    try:
        with open(DAEMON_HB_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _daemon_running() -> bool:
    hb = _read_daemon_hb()
    ts = hb.get("ts")
    interval = hb.get("interval", 5)
    if not ts:
        return False
    return (time.time() - float(ts)) <= max(10, int(interval) * 3)


def ping_sweep(nodes: List[Dict], alert_cfg: Dict):

    for node in nodes:
        node_type = node.get("node_type", "")
        ip = node.get("ip") or ""
        target = ip
        if node_type == "domain":
            resolved_ip = resolve_domain(node.get("name", ""))
            if resolved_ip and resolved_ip != ip:
                update_node_ip(node["id"], resolved_ip)
                ip = resolved_ip
            target = node.get("name") or ip
        if not target:
            continue
        ok, rtt_val = _ping_system(target, timeout_sec=1)
        if not ok:
            status = "down"
            loss = 100.0
        else:
            status = "up" if (rtt_val is not None and rtt_val < 150) else "degraded"
            loss = 0.0

        record_ping(node["id"], rtt_val, loss, status)
        should_alert, down_count, recovered = update_alert_state(
            node["id"],
            status == "down",
            alert_cfg["down_threshold"],
            alert_cfg["cooldown_sec"],
        )
        if should_alert:
            payload = {
                "node_id": node["id"],
                "name": node["name"],
                "ip": ip or target,
                "status": status,
                "down_count": down_count,
                "site": node.get("site", ""),
                "floor": node.get("floor", ""),
                "line": node.get("line", ""),
                "rtt_ms": rtt_val,
                "loss": loss,
            }
            send_webhook(alert_cfg.get("webhook_url", ""), payload)
            send_email(
                alert_cfg,
                subject=f"[Zabbix Map] {node['name']} down",
                body=f"Node {node['name']} ({ip}) is down for {down_count} checks.",
            )
        if recovered:
            payload = {
                "node_id": node["id"],
                "name": node["name"],
                "ip": ip or target,
                "status": "recovered",
                "site": node.get("site", ""),
                "floor": node.get("floor", ""),
                "line": node.get("line", ""),
                "rtt_ms": rtt_val,
                "loss": loss,
            }
            send_webhook(alert_cfg.get("webhook_url", ""), payload)
            send_email(
                alert_cfg,
                subject=f"[Zabbix Map] {node['name']} recovered",
                body=f"Node {node['name']} ({ip}) recovered.",
            )


def fetch_rtt_series(node_id: str, seconds: int = 60) -> Tuple[List[float], List[float]]:
    since = time.time() - seconds
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT ts, rtt_ms FROM ping_samples
        WHERE node_id = ? AND ts >= ?
        ORDER BY ts ASC
        """,
        (node_id, since),
    )
    rows = cur.fetchall()
    conn.close()
    x = [row[0] for row in rows]
    y = [row[1] if row[1] is not None else 0 for row in rows]
    return x, y


def fetch_avg_rtt(node_id: str, limit: int = 5) -> float:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT rtt_ms FROM ping_samples
        WHERE node_id = ? AND rtt_ms IS NOT NULL
        ORDER BY ts DESC
        LIMIT ?
        """,
        (node_id, limit),
    )
    rows = [row[0] for row in cur.fetchall() if row[0] is not None]
    conn.close()
    if not rows:
        return None
    return sum(rows) / len(rows)


def run():
    st.header("Network Architecture Diagram")
    if not os.path.isdir(COMPONENT_DIR):
        st.error("Missing component: Network Architecture Diagram/components/zabbix_map_component")
        st.info("Restore the component folder to enable the map canvas.")
        return

    ensure_db()
    # Keep map empty by default; only add nodes manually.
    if "component_key" not in st.session_state:
        st.session_state["component_key"] = 0

    with st.sidebar:
        st.subheader("Filters")
        floor = st.selectbox("Floor", ["All"] + FLOOR_OPTIONS)
        site = "All"
        line = "All"
        group_mode = "None"
        grid_size = 20
        snap_on_save = False
        st.divider()
        st.subheader("Ping sweep")
        interval_sec = st.number_input("Interval (sec)", min_value=10, max_value=3600, value=10, step=1)
        if st.button("Start auto ping"):
            st.session_state["auto_ping_enabled"] = True
            st.session_state["auto_ping_interval"] = max(10, int(interval_sec))
            st.session_state["last_ping_ts"] = 0.0
            st.session_state["node_editor_open"] = False
            st.session_state["run_ping_once"] = False
        if st.button("Stop auto ping"):
            st.session_state["auto_ping_enabled"] = False
        st.divider()
        st.subheader("Ping retention")
        retention_options = [7, 30, 180, 365]
        current_retention = load_setting("ping_retention_days", "30")
        try:
            current_retention_val = int(current_retention)
        except Exception:
            current_retention_val = 30
        if current_retention_val not in retention_options:
            current_retention_val = 30
        retention_days = st.selectbox(
            "Keep data (days)",
            retention_options,
            index=retention_options.index(current_retention_val),
        )
        col_apply, col_clear = st.columns(2)
        with col_apply:
            if st.button("Apply retention"):
                save_setting("ping_retention_days", str(retention_days))
                cleanup_ping_samples(retention_days)
                st.success(f"Retention set to {retention_days} days")
        with col_clear:
            if st.button("Clear ping data"):
                clear_all_ping_samples()
                st.success("Ping data cleared")

    filters = {"site": site, "floor": floor, "line": line}
    nodes = fetch_nodes(filters)
    edges = fetch_edges()

    if st.session_state.get("run_ping_once"):
        ping_sweep(nodes, load_alert_config())
        st.session_state["run_ping_once"] = False
        st.success("Ping sweep completed")

    if st.session_state.get("auto_ping_enabled"):
        interval = int(st.session_state.get("auto_ping_interval", 5))
        now = time.time()
        last_ts = float(st.session_state.get("last_ping_ts", 0))
        if now - last_ts >= interval:
            ping_sweep(nodes, load_alert_config())
            st.session_state["last_ping_ts"] = now
        if True:
            if HAS_AUTOREFRESH:
                st_autorefresh(interval=1000, key="auto-ping")
            else:
                time.sleep(1)
                if hasattr(st, "rerun"):
                    st.rerun()
                else:
                    st.experimental_rerun()

    elements = build_elements(nodes, edges, group_mode)
    stylesheet = [
        {
            "selector": "node",
            "style": {
                "label": "data(label)",
                "background-color": "data(color)",
                "shape": "data(shape)",
                "text-wrap": "wrap",
                "text-max-width": "140px",
                "color": "#e5e7eb",
                "font-size": "10px",
                "text-outline-color": "#0f172a",
                "text-outline-width": 2,
            },
        },
        {
            "selector": "edge",
            "style": {
                "line-color": "data(color)",
                "target-arrow-color": "data(color)",
                "target-arrow-shape": "triangle",
                "curve-style": "bezier",
                "width": 2,
            },
        },
        {
            "selector": ":parent",
            "style": {
                "background-opacity": 0.08,
                "border-color": "#94a3b8",
                "border-width": 1,
                "label": "data(label)",
                "font-size": "11px",
                "text-valign": "top",
            },
        },
    ]
    event = MAP_COMPONENT(
        elements=elements,
        stylesheet=stylesheet,
        height=680,
        palette=NODE_TYPES,
        type_shapes=TYPE_SHAPES,
        node_data=nodes,
        node_types=NODE_TYPES,
        floor_options=FLOOR_OPTIONS,
        key=f"zabbix-map-{st.session_state['component_key']}",
    )

    node_ids = [n["id"] for n in nodes]
    selected_pos = None
    last_ts_map = st.session_state.get("last_event_ts", {})
    if isinstance(event, dict) and event.get("event") == "select":
        pass
    elif isinstance(event, dict) and event.get("event") == "dragstop":
        selected_pos = event.get("position")
    elif isinstance(event, dict) and event.get("event") == "edit_save":
        ev_ts = event.get("ts")
        if ev_ts is not None and last_ts_map.get("edit_save") == ev_ts:
            return
        if ev_ts is not None:
            last_ts_map["edit_save"] = ev_ts
            st.session_state["last_event_ts"] = last_ts_map
        edit_id = event.get("node_id")
        if edit_id:
            target = next((n for n in nodes if n["id"] == edit_id), None)
            if target:
                upsert_node(
                    {
                        "id": edit_id,
                        "name": (event.get("name") or "").strip() or target["name"],
                        "ip": (event.get("ip") or "").strip(),
                        "node_type": event.get("node_type") or target.get("node_type"),
                        "site": target.get("site", ""),
                        "floor": event.get("floor") or target.get("floor"),
                        "line": target.get("line", ""),
                        "pos_x": target.get("pos_x", 0),
                        "pos_y": target.get("pos_y", 0),
                        "size_w": target.get("size_w"),
                        "size_h": target.get("size_h"),
                        "last_status": target.get("last_status", "unknown"),
                    }
                )
        st.session_state["page"] = "zabbix_map"
        st.session_state["component_key"] += 1
        st.rerun()
    elif isinstance(event, dict) and event.get("event") == "create_save":
        ev_ts = event.get("ts")
        if ev_ts is not None and last_ts_map.get("create_save") == ev_ts:
            return
        if ev_ts is not None:
            last_ts_map["create_save"] = ev_ts
            st.session_state["last_event_ts"] = last_ts_map
        node_type = event.get("node_type") or "router"
        name_value = (event.get("name") or "").strip()
        ip_value = (event.get("ip") or "").strip()
        floor_val = event.get("floor") or ""
        pos = event.get("position") or {}
        new_id = f"node-{int(time.time() * 1000)}"
        new_name = name_value or f"New {node_type.replace('_', ' ').title()}"
        upsert_node(
            {
                "id": new_id,
                "name": new_name,
                "ip": ip_value,
                "node_type": node_type,
                "site": "",
                "floor": floor_val,
                "line": "",
                "pos_x": pos.get("x", 100),
                "pos_y": pos.get("y", 100),
                "size_w": event.get("size_w"),
                "size_h": event.get("size_h"),
                "last_status": "unknown",
            }
        )
        st.session_state["page"] = "zabbix_map"
        st.session_state["component_key"] += 1
        st.rerun()
    elif isinstance(event, dict) and event.get("event") == "delete_confirm":
        ev_ts = event.get("ts")
        if ev_ts is not None and last_ts_map.get("delete_confirm") == ev_ts:
            return
        if ev_ts is not None:
            last_ts_map["delete_confirm"] = ev_ts
            st.session_state["last_event_ts"] = last_ts_map
        delete_id = event.get("node_id")
        if delete_id:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            target = next((n for n in nodes if n["id"] == delete_id), None)
            if target and target.get("node_type") == "domain" and target.get("name"):
                cur.execute(
                    "SELECT id FROM nodes WHERE node_type = 'domain' AND name = ?",
                    (target["name"],),
                )
                ids = [row[0] for row in cur.fetchall()]
                for node_id in ids:
                    cur.execute("DELETE FROM edges WHERE source = ? OR target = ?", (node_id, node_id))
                    cur.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
            else:
                cur.execute("DELETE FROM edges WHERE source = ? OR target = ?", (delete_id, delete_id))
                cur.execute("DELETE FROM nodes WHERE id = ?", (delete_id,))
            conn.commit()
            conn.close()
        st.session_state["page"] = "zabbix_map"
        st.session_state["component_key"] += 1
        st.rerun()
    elif isinstance(event, dict) and event.get("event") == "link":
        source_id = event.get("source_id")
        target_id = event.get("target_id")
        if source_id and target_id and source_id != target_id:
            edge_id = f"e-{int(time.time() * 1000)}"
            upsert_edge({"id": edge_id, "source": source_id, "target": target_id, "status": "unknown"})
    if selected_pos and selected_pos.get("x") is not None and selected_pos.get("y") is not None:
        pos_x = float(selected_pos["x"])
        pos_y = float(selected_pos["y"])
        if snap_on_save:
            pos_x = snap_value(pos_x, int(grid_size))
            pos_y = snap_value(pos_y, int(grid_size))
        if isinstance(event, dict) and event.get("event") == "dragstop":
            update_node_position(event.get("selected_node_id"), pos_x, pos_y)
    if isinstance(event, dict) and event.get("event") == "resize":
        node_id = event.get("node_id")
        size = event.get("size") or {}
        if node_id and size.get("w") and size.get("h"):
            update_node_size(node_id, float(size["w"]), float(size["h"]))

    st.divider()



    st.divider()
    with st.expander("Alert", expanded=False):
        alert_cfg = load_alert_config()
        down_threshold = st.number_input("Down threshold (N)", min_value=1, max_value=20, value=max(1, alert_cfg["down_threshold"]))
        cooldown_sec = st.number_input("Cooldown (sec)", min_value=60, max_value=3600, value=alert_cfg["cooldown_sec"])
        st.caption("Cooldown = same node alert minimum interval (after a trigger, wait this long before next alert).")
        webhook_url = st.text_input("Webhook URL", value=alert_cfg.get("webhook_url", ""))
        email_to = st.text_input("Email to", value=alert_cfg.get("email_to", ""))
        email_from = st.text_input("Email from", value=alert_cfg.get("email_from", ""))
        smtp_host = st.text_input("SMTP host", value=alert_cfg.get("smtp_host", ""))
        smtp_port_val = alert_cfg.get("smtp_port")
        if smtp_port_val is None:
            smtp_port_val = 587
        smtp_port = st.number_input("SMTP port", min_value=1, max_value=65535, value=int(smtp_port_val))
        smtp_user = st.text_input("SMTP user", value=alert_cfg.get("smtp_user", ""))
        smtp_pass = st.text_input("SMTP pass", value=alert_cfg.get("smtp_pass", ""), type="password")
        col_save, col_test = st.columns([1, 1])
        with col_save:
            save_clicked = st.button("Save alert config")
        with col_test:
            test_clicked = st.button("Test webhook")
        if save_clicked:
            save_alert_config(
                {
                    "down_threshold": max(1, int(down_threshold)),
                    "cooldown_sec": int(cooldown_sec),
                    "webhook_url": webhook_url,
                    "email_to": email_to,
                    "email_from": email_from,
                    "smtp_host": smtp_host,
                    "smtp_port": int(smtp_port),
                    "smtp_user": smtp_user,
                    "smtp_pass": smtp_pass,
                }
            )
            st.success("Alert config saved")
        if test_clicked:
            url = webhook_url.strip() if webhook_url else ""
            if url:
                st.caption(f"URL: {url}")
            result = send_webhook(
                url,
                {
                    "test": True,
                    "content": "Webhook test from Cisco Maintain Tools",
                    "mentioned_list": ["@all"],
                    "ts": time.time(),
                },
            )
            if result.get("ok"):
                st.success(f"Webhook sent: {result.get('status_code')}")
                resp_text = result.get("text")
                if resp_text:
                    st.code(resp_text, language="json")
            else:
                st.error(f"Webhook error: {result.get('error')}")

    st.divider()
    with st.expander("RTT chart (last 60s)", expanded=False):
        chart_id = st.selectbox("Select node for RTT", node_ids, key="chart-node")
        x, y = fetch_rtt_series(chart_id)
        if y:
            st.line_chart({"rtt_ms": y})
        else:
            st.info("No RTT data yet. Run ping sweep first.")
