# Cisco_Smart_health_check.py
# Cisco Smart Health Check (IOS-XE) - Submodule for cmt.py
# IMPORTANT:
# - Do NOT call st.set_page_config() here (handled by cmt.py)
# - Provide def run(): entry point
# - Read-only show commands only

import re
import io
import time
import streamlit as st
import pandas as pd
from netmiko import ConnectHandler
from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================================================
# Read-only command allowlist (hard block)
# =========================================================
def safe_send(conn, cmd: str) -> str:
    c = (cmd or "").strip()
    lc = c.lower()
    if not (lc.startswith("show ") or lc == "terminal length 0"):
        raise ValueError(f"Blocked non-readonly command: {cmd}")
    try:
        return conn.send_command(c)
    except Exception:
        return ""

# =========================================================
# CSV helper
# =========================================================
def read_csv_ips(uploaded_file) -> list[str]:
    df = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
    df.columns = [c.strip().lower() for c in df.columns]
    if "ip" not in df.columns:
        raise ValueError("CSV must contain column: ip")

    ips = []
    for x in df["ip"].astype(str).tolist():
        ip = x.strip()
        if ip and ip.lower() != "nan":
            ips.append(ip)

    # unique keep order
    seen = set()
    uniq = []
    for ip in ips:
        if ip not in seen:
            uniq.append(ip)
            seen.add(ip)
    return uniq

def _sig_from_ips(ips: list[str]) -> str:
    return "|".join([x.strip() for x in ips if x.strip()])

# =========================================================
# Connection helper
# =========================================================
def connect_fallback(base_device: dict):
    """Auto-try device_type: cisco_xe -> cisco_ios"""
    last_err = None
    for dtype in ("cisco_xe", "cisco_ios"):
        try:
            d = dict(base_device)
            d["device_type"] = dtype
            conn = ConnectHandler(**d)
            return conn, dtype
        except Exception as e:
            last_err = e
    raise last_err

# =========================================================
# Parsers (best-effort for IOS-XE)
# =========================================================
def parse_hostname_from_run(output: str) -> str:
    m = re.search(r"^\s*hostname\s+(\S+)\s*$", output or "", flags=re.M)
    return m.group(1) if m else "N/A"

def parse_version(output: str) -> tuple[str, str, str, str]:
    model = version = serial = uptime = "N/A"
    out = output or ""
    lines = out.splitlines()

    for line in lines:
        if "Cisco IOS XE Software" in line and "Version" in line:
            version = line.strip()
            break
    if version == "N/A":
        for line in lines:
            if "Cisco IOS Software" in line and "Version" in line:
                version = line.strip()
                break

    for line in lines:
        if "Processor board ID" in line:
            serial = line.strip().split()[-1]
            break

    for line in lines:
        if " uptime is " in line.lower():
            uptime = line.split("uptime is", 1)[-1].strip()
            break

    for line in lines:
        if "Model Number" in line:
            model = line.split(":", 1)[-1].strip() if ":" in line else line.strip()
            break
    if model == "N/A":
        for line in lines:
            m = re.search(r"^\s*cisco\s+(\S+)\s+\(", line, flags=re.I)
            if m:
                model = m.group(1)
                break

    return model, version, serial, uptime

def parse_cpu_line(output: str):
    out = output or ""
    m = re.search(r"five seconds:\s*(\d+)%.*?one minute:\s*(\d+)%.*?five minutes:\s*(\d+)%", out, flags=re.I)
    if not m:
        return None, None, None
    return float(m.group(1)), float(m.group(2)), float(m.group(3))

def parse_memory_used_percent(output: str):
    out = output or ""
    m = re.search(r"Processor Pool Total:\s*(\d+)\s+Used:\s*(\d+)\s+Free:\s*(\d+)", out, flags=re.I)
    if not m:
        return None
    total = float(m.group(1))
    used = float(m.group(2))
    if total <= 0:
        return None
    return round(used / total * 100.0, 2)

def parse_ntp_sync(output: str):
    out = (output or "").lower()
    if "unsynchronized" in out:
        return False
    if "synchronized" in out:
        return True
    return None

def parse_env_alerts(output: str) -> int:
    """
    Best-effort count of environment alarms from 'show environment all'
    """
    out = output or ""
    count = 0
    for line in out.splitlines():
        if re.search(r"\b(fail|fault|critical|shutdown)\b", line, flags=re.I):
            count += 1
    return count

def parse_int_error_summary(output: str) -> dict:
    out = output or ""
    lines = [l.rstrip() for l in out.splitlines() if l.strip()]
    if len(lines) < 2:
        return {"ports": 0, "crc_sum": 0, "in_err_sum": 0, "out_err_sum": 0}

    header_idx = None
    for i, l in enumerate(lines[:10]):
        if re.search(r"\bPort\b|\bInterface\b", l, flags=re.I) and re.search(r"\bCRC\b", l, flags=re.I):
            header_idx = i
            break

    if header_idx is None:
        ports = sum(1 for l in lines if re.match(r"^(Gi|Te|Fa|Hu|Fo|Tw|Eth)\S+", l))
        return {"ports": ports, "crc_sum": 0, "in_err_sum": 0, "out_err_sum": 0}

    header = re.split(r"\s{2,}", lines[header_idx].strip())
    col_map = {name.lower(): idx for idx, name in enumerate(header)}

    def col_idx(names):
        for n in names:
            for k, idx in col_map.items():
                if n in k:
                    return idx
        return None

    idx_port = col_idx(["port", "interface"])
    idx_crc = col_idx(["crc"])
    idx_in = col_idx(["inerr", "input", "in err"])
    idx_out = col_idx(["outerr", "output", "out err"])

    crc_sum = in_sum = out_sum = 0
    ports = 0

    for l in lines[header_idx + 1:]:
        if not re.match(r"^(Gi|Te|Fa|Hu|Fo|Tw|Eth)\S+", l.strip()):
            continue
        parts = re.split(r"\s{2,}", l.strip())
        if idx_port is None or len(parts) <= idx_port:
            continue
        ports += 1

        def get_int(ix):
            try:
                if ix is None or len(parts) <= ix:
                    return 0
                return int(re.sub(r"[^\d]", "", parts[ix]) or "0")
            except Exception:
                return 0

        crc_sum += get_int(idx_crc)
        in_sum += get_int(idx_in)
        out_sum += get_int(idx_out)

    return {"ports": ports, "crc_sum": crc_sum, "in_err_sum": in_sum, "out_err_sum": out_sum}

# ---------------- Optical (A) ----------------
def parse_transceiver_table(output: str) -> pd.DataFrame:
    """
    Best-effort parse for 'show interfaces transceiver'
    Goal: Interface / Temp(C) / Tx(dBm) / Rx(dBm)
    """
    rows = []
    out = output or ""
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue

        # common patterns: Interface Temp Tx Rx ... (varies by platform)
        # Strategy:
        # 1) Must start with interface
        # 2) Extract ALL floats in line, then map likely Temp/Tx/Rx by position
        if not re.match(r"^(Gi|Te|Fa|Hu|Fo|Tw|Eth)\S+", line):
            continue

        intf = line.split()[0]
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", line)
        nums_f = [float(x) for x in nums]

        # Heuristic:
        # - Temp usually in 0~100
        # - Tx/Rx usually negative/positive small dBm (-40~10)
        temp = None
        tx = None
        rx = None

        # find temp first (first number within plausible temp range)
        for v in nums_f[1:]:
            if 0 <= v <= 100:
                temp = v
                break

        # find dBm candidates (within -40~10)
        dbm = [v for v in nums_f if -40 <= v <= 10]
        # often tx & rx are last 2 dbm-like numbers
        if len(dbm) >= 2:
            tx, rx = dbm[-2], dbm[-1]
        elif len(dbm) == 1:
            rx = dbm[-1]

        rows.append({
            "Interface": intf,
            "Temp(C)": None if temp is None else round(temp, 2),
            "Tx(dBm)": None if tx is None else round(tx, 2),
            "Rx(dBm)": None if rx is None else round(rx, 2),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by="Interface", kind="stable")
    return df

def classify_optical(rx, temp):
    # You can adjust thresholds to match your IE policy
    RX_WARN = -18.0
    RX_CRIT = -22.0
    TEMP_WARN = 60.0
    TEMP_CRIT = 70.0

    state = "OK"
    if rx is not None:
        if rx <= RX_CRIT:
            state = "CRITICAL"
        elif rx <= RX_WARN:
            state = "WARNING"
    if temp is not None:
        if temp >= TEMP_CRIT:
            state = "CRITICAL"
        elif temp >= TEMP_WARN and state != "CRITICAL":
            state = "WARNING"
    return state

def style_optical_row(row: pd.Series):
    # Return per-cell style
    st_map = {}
    state = row.get("State", "")
    if state == "OK":
        bg = "#1f7a1f"
    elif state == "WARNING":
        bg = "#a07900"
    elif state == "CRITICAL":
        bg = "#a00000"
    else:
        bg = ""
    if bg:
        st_map["State"] = f"background-color:{bg};color:white;"
    return st_map

# ---------------- VLAN/SVI (B) ----------------
def parse_vlan_brief(output: str) -> pd.DataFrame:
    rows = []
    out = output or ""
    for line in out.splitlines():
        line = line.rstrip()
        if not line or line.strip().startswith("VLAN") or line.strip().startswith("----"):
            continue
        m = re.match(r"^\s*(\d+)\s+(\S+)\s+(\S+)\s*(.*)$", line)
        if not m:
            continue
        rows.append({
            "VLAN": int(m.group(1)),
            "Name": m.group(2),
            "Status": m.group(3),
            "Ports": m.group(4).strip(),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by="VLAN", kind="stable")
    return df

def parse_ip_int_brief_svi(output: str) -> pd.DataFrame:
    rows = []
    out = output or ""
    for line in out.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("interface"):
            continue
        # Vlan1  10.1.1.1  YES manual  up  up
        if not line.startswith("Vlan"):
            continue
        parts = re.split(r"\s+", line)
        if len(parts) < 6:
            continue
        rows.append({
            "SVI": parts[0],
            "IP": parts[1],
            "Status": parts[-2],
            "Protocol": parts[-1],
        })
    df = pd.DataFrame(rows)
    return df

def parse_trunk_ports(output: str) -> pd.DataFrame:
    """
    show interfaces trunk
    Best-effort: capture port and status/encap/native if present.
    """
    rows = []
    out = output or ""
    lines = [l.rstrip() for l in out.splitlines() if l.strip()]
    if not lines:
        return pd.DataFrame(rows)

    # Find table header that contains "Port" and "Status"
    header_i = None
    for i, l in enumerate(lines[:30]):
        if re.search(r"\bPort\b", l) and re.search(r"\bStatus\b", l):
            header_i = i
            break
    if header_i is None:
        return pd.DataFrame(rows)

    # subsequent lines until blank or next section
    for l in lines[header_i + 1:]:
        if l.lower().startswith("port") or l.lower().startswith("----"):
            continue
        # stop at next section
        if "Vlans allowed" in l or "Vlans in spanning tree" in l:
            break
        parts = re.split(r"\s+", l.strip())
        if len(parts) < 2:
            continue
        rows.append({
            "Port": parts[0],
            "Status": parts[1],
            "NativeVLAN": parts[3] if len(parts) >= 4 else "",
        })

    return pd.DataFrame(rows)

# =========================================================
# Health rules (add VLAN/SVI checks)
# =========================================================
def decide_status(cpu5, mem_pct, err_crc, trans_crit, vlan1_exists, vlan1_svi_has_ip, svi_down_cnt, env_alerts, ntp_sync) -> tuple[str, int, int]:
    critical = 0
    warning = 0

    # CPU thresholds
    if cpu5 is not None:
        if cpu5 >= 80:
            critical += 1
        elif cpu5 >= 60:
            warning += 1

    # Memory thresholds
    if mem_pct is not None:
        if mem_pct >= 85:
            critical += 1
        elif mem_pct >= 75:
            warning += 1

    # CRC/Errors thresholds (coarse)
    if err_crc >= 200:
        critical += 1
    elif err_crc >= 20:
        warning += 1

    # Optical critical ports
    if trans_crit >= 1:
        critical += 1

    # VLAN/SVI (B)
    if not vlan1_exists:
        critical += 1
    if vlan1_exists and (not vlan1_svi_has_ip):
        critical += 1
    if svi_down_cnt >= 1:
        warning += 1

    # Environment
    if env_alerts >= 1:
        critical += 1

    # NTP
    if ntp_sync is False:
        warning += 1

    if critical > 0:
        return "CRITICAL", critical, warning
    if warning > 0:
        return "WARNING", critical, warning
    return "OK", critical, warning

def style_status(v: str):
    v = (v or "").upper()
    if v == "OK":
        return "background-color:#1f7a1f;color:white;"
    if v == "WARNING":
        return "background-color:#a07900;color:white;"
    if v == "CRITICAL":
        return "background-color:#a00000;color:white;"
    return ""

# =========================================================
# Per-device worker
# =========================================================
def get_device(base_device: dict) -> tuple[dict, dict]:
    ip = base_device.get("host", "N/A")

    overview = {
        "IP": ip,
        "Hostname": "N/A",
        "Model": "N/A",
        "IOS-XE": "N/A",
        "Serial": "N/A",
        "Uptime": "N/A",
        "CPU5%": None,
        "MemUsed%": None,
        "CRC Sum": 0,
        "InErr Sum": 0,
        "OutErr Sum": 0,

        # A Optical summary
        "XCVR CritPorts": 0,
        "XCVR WarnPorts": 0,
        "XCVR MinRx(dBm)": None,
        "XCVR MaxTemp(C)": None,

        # Environment/NTP
        "Env Alerts": 0,
        "NTP Sync": "Unknown",

        # B VLAN/SVI summary
        "VLAN1 Exists": False,
        "VLAN1 SVI IP": "N/A",
        "SVI Down Count": 0,
        "Trunk Count": 0,

        "Health Score": 0,
        "Issues": "",
        "Status": "N/A",
        "Critical": 0,
        "Warning": 0,
        "DeviceType": "N/A",
    }
    details = {"raw": {}, "tables": {}}

    t0 = time.time()
    try:
        conn, dtype = connect_fallback(base_device)
        overview["DeviceType"] = dtype
        safe_send(conn, "terminal length 0")

        # --- Collect base ---
        run_out = safe_send(conn, "show run | include ^hostname")
        ver_out = safe_send(conn, "show version")
        cpu_out = safe_send(conn, "show processes cpu | include CPU utilization")
        mem_out = safe_send(conn, "show memory statistics")
        err_out = safe_send(conn, "show interfaces counters errors")

        # --- A Optical ---
        xcvr_out = safe_send(conn, "show interfaces transceiver")

        # --- Env/NTP ---
        env_out = safe_send(conn, "show environment all")
        ntp_out = safe_send(conn, "show ntp status")

        # --- B VLAN/SVI ---
        vlan_out = safe_send(conn, "show vlan brief")
        ipif_out = safe_send(conn, "show ip interface brief")
        trunk_out = safe_send(conn, "show interfaces trunk")

        # --- Parse base ---
        overview["Hostname"] = parse_hostname_from_run(run_out)
        model, version, serial, uptime = parse_version(ver_out)
        overview["Model"] = model
        overview["IOS-XE"] = version
        overview["Serial"] = serial
        overview["Uptime"] = uptime

        cpu5, cpu1, cpu5m = parse_cpu_line(cpu_out)
        overview["CPU5%"] = None if cpu5 is None else round(cpu5, 2)

        mem_pct = parse_memory_used_percent(mem_out)
        overview["MemUsed%"] = mem_pct

        err_sum = parse_int_error_summary(err_out)
        overview["CRC Sum"] = int(err_sum.get("crc_sum", 0))
        overview["InErr Sum"] = int(err_sum.get("in_err_sum", 0))
        overview["OutErr Sum"] = int(err_sum.get("out_err_sum", 0))

        # --- A Optical tables + summary ---
        df_xcvr = parse_transceiver_table(xcvr_out)
        if not df_xcvr.empty:
            df_xcvr["State"] = df_xcvr.apply(
                lambda r: classify_optical(r.get("Rx(dBm)"), r.get("Temp(C)")),
                axis=1
            )
            overview["XCVR CritPorts"] = int((df_xcvr["State"] == "CRITICAL").sum())
            overview["XCVR WarnPorts"] = int((df_xcvr["State"] == "WARNING").sum())
            if df_xcvr["Rx(dBm)"].notna().any():
                overview["XCVR MinRx(dBm)"] = float(df_xcvr["Rx(dBm)"].min())
            if df_xcvr["Temp(C)"].notna().any():
                overview["XCVR MaxTemp(C)"] = float(df_xcvr["Temp(C)"].max())
        details["tables"]["optical"] = df_xcvr

        # --- Env/NTP summary ---
        env_alerts = parse_env_alerts(env_out)
        overview["Env Alerts"] = int(env_alerts)

        ntp_sync = parse_ntp_sync(ntp_out)
        if ntp_sync is True:
            overview["NTP Sync"] = "Yes"
        elif ntp_sync is False:
            overview["NTP Sync"] = "No"
        else:
            overview["NTP Sync"] = "Unknown"

        # --- B VLAN / SVI / Trunk tables + summary ---
        df_vlan = parse_vlan_brief(vlan_out)
        details["tables"]["vlan"] = df_vlan
        if not df_vlan.empty:
            overview["VLAN1 Exists"] = bool((df_vlan["VLAN"] == 1).any())

        df_svi = parse_ip_int_brief_svi(ipif_out)
        details["tables"]["svi"] = df_svi

        vlan1_ip = "N/A"
        vlan1_has_ip = False
        svi_down_cnt = 0
        if not df_svi.empty:
            # VLAN1 SVI IP
            row1 = df_svi[df_svi["SVI"].str.lower() == "vlan1"]
            if not row1.empty:
                vlan1_ip = str(row1.iloc[0]["IP"])
                vlan1_has_ip = vlan1_ip.lower() not in ("unassigned", "n/a", "")
            # SVI down count
            svi_down_cnt = int(((df_svi["Status"].str.lower() != "up") | (df_svi["Protocol"].str.lower() != "up")).sum())

        overview["VLAN1 SVI IP"] = vlan1_ip
        overview["SVI Down Count"] = svi_down_cnt

        df_trunk = parse_trunk_ports(trunk_out)
        details["tables"]["trunk"] = df_trunk
        overview["Trunk Count"] = int(len(df_trunk)) if isinstance(df_trunk, pd.DataFrame) else 0

        # --- Final status ---
        status, c, w = decide_status(
            cpu5=overview["CPU5%"],
            mem_pct=overview["MemUsed%"],
            err_crc=overview["CRC Sum"],
            trans_crit=overview["XCVR CritPorts"],
            vlan1_exists=overview["VLAN1 Exists"],
            vlan1_svi_has_ip=vlan1_has_ip,
            svi_down_cnt=overview["SVI Down Count"],
            env_alerts=overview["Env Alerts"],
            ntp_sync=ntp_sync,
        )
        overview["Status"] = status
        overview["Critical"] = c
        overview["Warning"] = w
        overview["Health Score"] = max(0, 100 - (c * 30) - (w * 10))

        issues = []
        if overview["CPU5%"] is not None and overview["CPU5%"] >= 80:
            issues.append("High CPU")
        if overview["MemUsed%"] is not None and overview["MemUsed%"] >= 85:
            issues.append("High Memory")
        if overview["CRC Sum"] >= 200:
            issues.append("CRC Errors")
        if overview["XCVR CritPorts"] >= 1:
            issues.append("XCVR Critical")
        if overview["Env Alerts"] >= 1:
            issues.append("Env Alert")
        if overview["NTP Sync"] == "No":
            issues.append("NTP Unsync")
        if not overview["VLAN1 Exists"]:
            issues.append("VLAN1 Missing")
        if overview["VLAN1 Exists"] and (not vlan1_has_ip):
            issues.append("VLAN1 IP Missing")
        if overview["SVI Down Count"] >= 1:
            issues.append("SVI Down")
        overview["Issues"] = "; ".join(issues)

        # raw (for troubleshooting)
        details["raw"]["cpu"] = cpu_out
        details["raw"]["memory"] = mem_out
        details["raw"]["errors"] = err_out
        details["raw"]["transceiver"] = xcvr_out
        details["raw"]["environment"] = env_out
        details["raw"]["ntp"] = ntp_out
        details["raw"]["vlan"] = vlan_out
        details["raw"]["ip_int_brief"] = ipif_out
        details["raw"]["trunk"] = trunk_out
        details["elapsed_s"] = round(time.time() - t0, 2)

        try:
            conn.disconnect()
        except Exception:
            pass

    except Exception as e:
        overview["Status"] = "CRITICAL"
        overview["Critical"] = 1
        details["error"] = str(e)
        details["elapsed_s"] = round(time.time() - t0, 2)

    # normalize numeric rounding
    if overview.get("XCVR MinRx(dBm)") is not None:
        overview["XCVR MinRx(dBm)"] = round(float(overview["XCVR MinRx(dBm)"]), 2)
    if overview.get("XCVR MaxTemp(C)") is not None:
        overview["XCVR MaxTemp(C)"] = round(float(overview["XCVR MaxTemp(C)"]), 2)

    return overview, details

# =========================================================
# Stable selectbox
# =========================================================
def stable_selectbox_ip(label: str, options: list[str], key: str) -> str:
    if not options:
        return ""
    cur = str(st.session_state.get(key, "")).strip()
    if cur not in options:
        st.session_state[key] = options[0]
        cur = options[0]
    return st.selectbox(label, options, index=options.index(cur), key=key)

# =========================================================
# UI ENTRY (REQUIRED by cmt.py)
# =========================================================
def run():
    st.title("Cisco Smart Health Check (Read-Only)")
    st.caption("🔒 IOS-XE first | show-only commands | batch by CSV (ip) | A=Optical + B=VLAN/SVI")

    # -------------------------
    # Sidebar (match your other modules)
    # -------------------------
    with st.sidebar:
        st.markdown("## SSH Credentials")
        username = st.text_input("SSH Username", value="", key="shc_user")
        password = st.text_input("SSH Password", value="", type="password", key="shc_pass")

        st.markdown("### Concurrency")
        workers = st.number_input("Workers", min_value=1, max_value=50, value=5, step=1, key="shc_workers")

        st.markdown("### Per-device timeout (sec)")
        timeout = st.number_input("Timeout", min_value=3, max_value=120, value=8, step=1, key="shc_timeout")

        st.markdown("### Upload CSV (ip)")
        uploaded = st.file_uploader("CSV file", type=["csv"], accept_multiple_files=False, key="shc_csv")

        ready = bool(username.strip()) and bool(password.strip()) and (uploaded is not None)
        c1, c2 = st.columns(2)
        run_btn = c1.button("Run Checks", use_container_width=True, disabled=not ready, key="shc_run")
        rerun_btn = c2.button("Re-run Checks", use_container_width=True, disabled=not ready, key="shc_rerun")

    # -------------------------
    # Run / Re-run => collect and store to session_state
    # -------------------------
    if run_btn or rerun_btn:
        try:
            ips = read_csv_ips(uploaded)
            if not ips:
                st.error("CSV loaded but no valid IPs found in 'ip' column.")
                st.stop()

            ips_sig = _sig_from_ips(ips)

            base_devices = [{
                "host": ip,
                "username": username,
                "password": password,
                "fast_cli": True,
                "timeout": int(timeout),
                "auth_timeout": int(timeout),
                "banner_timeout": int(timeout),
            } for ip in ips]

            rows = []
            details_map = {}

            with st.spinner("Running Smart Health Check... (CPU/Memory/Errors + Optical + VLAN/SVI)"):
                with ThreadPoolExecutor(max_workers=int(workers)) as executor:
                    futures = [executor.submit(get_device, b) for b in base_devices]
                    for f in as_completed(futures):
                        o, d = f.result()
                        o["IP"] = str(o.get("IP", "")).strip()
                        rows.append(o)
                        if d and o["IP"]:
                            details_map[o["IP"]] = d

            df = pd.DataFrame(rows)
            if not df.empty and "IP" in df.columns:
                df["IP"] = df["IP"].astype(str).str.strip()
                df = df.sort_values(by=["Status", "IP"], ascending=[True, True], kind="stable")

            st.session_state["shc_ips_sig"] = ips_sig
            st.session_state["shc_df_over"] = df
            st.session_state["shc_details_map"] = details_map

            ip_options = df["IP"].tolist() if (isinstance(df, pd.DataFrame) and not df.empty and "IP" in df.columns) else []
            if "shc_selected_ip" not in st.session_state or str(st.session_state["shc_selected_ip"]).strip() not in ip_options:
                if ip_options:
                    st.session_state["shc_selected_ip"] = ip_options[0]

        except Exception as e:
            st.error(f"Failed: {e}")
            return

    # -------------------------
    # Display from session_state
    # -------------------------
    df_over = st.session_state.get("shc_df_over")
    details_map = st.session_state.get("shc_details_map", {})

    if df_over is None or not isinstance(df_over, pd.DataFrame) or df_over.empty:
        st.info("Upload CSV and click **Run Checks** to start.")
        return

    # Summary cards
    c1, c2, c3 = st.columns(3)
    ok_cnt = int((df_over["Status"] == "OK").sum()) if "Status" in df_over.columns else 0
    warn_cnt = int((df_over["Status"] == "WARNING").sum()) if "Status" in df_over.columns else 0
    crit_cnt = int((df_over["Status"] == "CRITICAL").sum()) if "Status" in df_over.columns else 0
    c1.metric("OK", ok_cnt)
    c2.metric("WARNING", warn_cnt)
    c3.metric("CRITICAL", crit_cnt)

    if "Health Score" in df_over.columns:
        c1, c2 = st.columns(2)
        avg_score = int(df_over["Health Score"].mean()) if not df_over.empty else 0
        c1.metric("Avg Health Score", avg_score)
        c2.metric("Devices", int(len(df_over)))

    st.subheader("Anomaly Top N")
    if "Status" in df_over.columns:
        top_n = df_over[df_over["Status"].isin(["CRITICAL", "WARNING"])].copy()
        if top_n.empty:
            st.info("No anomalies detected.")
        else:
            top_n = top_n.sort_values(
                by=["Critical", "Warning", "Health Score", "IP"],
                ascending=[False, False, True, True],
                kind="stable",
            )
            cols = [c for c in ["IP", "Hostname", "Status", "Health Score", "Issues", "Critical", "Warning"] if c in top_n.columns]
            st.dataframe(top_n.head(5)[cols], use_container_width=True)

    st.subheader("Results (Overview)")
    if "Status" in df_over.columns:
        st.dataframe(df_over.style.applymap(style_status, subset=["Status"]), use_container_width=True)
    else:
        st.dataframe(df_over, use_container_width=True)

    ip_options = df_over["IP"].astype(str).str.strip().tolist()
    selected_ip = stable_selectbox_ip("Select device", ip_options, key="shc_selected_ip")
    if not selected_ip:
        return

    d = details_map.get(selected_ip, {})
    tables = d.get("tables", {}) if isinstance(d, dict) else {}

    tabs = st.tabs([
        "Summary",
        "A) Optical",
        "B) VLAN/SVI",
        "CPU",
        "Memory",
        "Interface Errors",
        "Trunk",
        "Environment/NTP",
        "Raw / Error"
    ])

    with tabs[0]:
        st.markdown("### Device Summary")
        row = df_over[df_over["IP"] == selected_ip].iloc[0].to_dict()
        st.json(row)

    with tabs[1]:
        st.markdown("### A) Optical (Transceiver)")
        df_x = tables.get("optical")
        if isinstance(df_x, pd.DataFrame) and not df_x.empty:
            # style State + keep numeric 2 decimals
            styled = df_x.style
            if "State" in df_x.columns:
                # style per State cell
                def _state_style(v):
                    if v == "OK":
                        return "background-color:#1f7a1f;color:white;"
                    if v == "WARNING":
                        return "background-color:#a07900;color:white;"
                    if v == "CRITICAL":
                        return "background-color:#a00000;color:white;"
                    return ""
                styled = styled.applymap(_state_style, subset=["State"])
            st.dataframe(styled, use_container_width=True)
        else:
            st.info("No transceiver rows parsed. Showing raw output below:")
            st.code((d.get("raw", {}) or {}).get("transceiver", ""), language="text")

    with tabs[2]:
        st.markdown("### B) VLAN / SVI")
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### VLAN Brief")
            df_vlan = tables.get("vlan")
            if isinstance(df_vlan, pd.DataFrame) and not df_vlan.empty:
                st.dataframe(df_vlan, use_container_width=True)
            else:
                st.info("No VLAN rows parsed.")
                st.code((d.get("raw", {}) or {}).get("vlan", ""), language="text")

        with c2:
            st.markdown("#### SVI (show ip interface brief -> VlanX)")
            df_svi = tables.get("svi")
            if isinstance(df_svi, pd.DataFrame) and not df_svi.empty:
                st.dataframe(df_svi, use_container_width=True)
            else:
                st.info("No SVI rows parsed.")
                st.code((d.get("raw", {}) or {}).get("ip_int_brief", ""), language="text")

    with tabs[3]:
        st.markdown("### CPU")
        st.code((d.get("raw", {}) or {}).get("cpu", ""), language="text")

    with tabs[4]:
        st.markdown("### Memory")
        st.code((d.get("raw", {}) or {}).get("memory", ""), language="text")

    with tabs[5]:
        st.markdown("### Interface Errors (counters errors)")
        st.code((d.get("raw", {}) or {}).get("errors", ""), language="text")

    with tabs[6]:
        st.markdown("### Trunk Ports (show interfaces trunk)")
        df_t = tables.get("trunk")
        if isinstance(df_t, pd.DataFrame) and not df_t.empty:
            st.dataframe(df_t, use_container_width=True)
        else:
            st.info("No trunk table parsed. Showing raw output:")
            st.code((d.get("raw", {}) or {}).get("trunk", ""), language="text")

    with tabs[7]:
        st.markdown("### Environment / NTP")
        st.code((d.get("raw", {}) or {}).get("environment", ""), language="text")
        st.code((d.get("raw", {}) or {}).get("ntp", ""), language="text")

    with tabs[8]:
        st.markdown("### Raw / Error")
        if d.get("error"):
            st.error(d["error"])
        st.write("Elapsed(s):", d.get("elapsed_s"))
