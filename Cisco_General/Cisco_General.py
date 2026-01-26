# Cisco_General.py
# Cisco General Information (Read-Only) - Submodule version for cmt.py
# IMPORTANT:
# - Do NOT call st.set_page_config() at import time
# - Provide def run(): entry for cmt.py

import re
import io
import pandas as pd
import streamlit as st
from netmiko import ConnectHandler
from concurrent.futures import ThreadPoolExecutor, as_completed


# =========================================================
# Generic helpers
# =========================================================
def safe_strip(x: str) -> str:
    return (x or "").strip() or "N/A"


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


def safe_send(conn, cmd: str) -> str:
    """Never let one command kill the whole device. Returns '' on error."""
    try:
        return conn.send_command(cmd)
    except Exception:
        return ""


# =========================================================
# Parsers (Core)
# =========================================================
def parse_hostname(output: str) -> str:
    if not output:
        return "N/A"
    m = re.search(r"^\s*hostname\s+(\S+)\s*$", output, flags=re.MULTILINE)
    return m.group(1).strip() if m else "N/A"


def parse_version(output: str):
    model = version = serial = uptime = "N/A"
    if not output:
        return model, version, serial, uptime

    lines = output.splitlines()

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
            m = re.search(r"^\s*cisco\s+(\S+)\s+\(", line, flags=re.IGNORECASE)
            if m:
                model = m.group(1).strip()
                break

    return model, version, serial, uptime


def parse_vtp_mode(output: str) -> str:
    if not output:
        return "N/A"
    for line in output.splitlines():
        if "VTP Operating Mode" in line:
            return line.split(":", 1)[-1].strip() or "N/A"
    return "N/A"


def parse_vlan_brief(output: str) -> pd.DataFrame:
    rows = []
    for line in (output or "").splitlines():
        m = re.match(r"^\s*(\d+)\s+(\S+)\s+(active|suspend|act/unsup)\b", line, flags=re.IGNORECASE)
        if m:
            rows.append({"VLAN": m.group(1), "Name": m.group(2), "Status": m.group(3)})
    return pd.DataFrame(rows)


def parse_svi_ip_int_brief(output: str) -> pd.DataFrame:
    rows = []
    for line in (output or "").splitlines():
        line = line.strip()
        if line.lower().startswith("vlan"):
            parts = re.split(r"\s+", line)
            if len(parts) >= 6:
                rows.append({
                    "Interface": parts[0],
                    "IP": parts[1],
                    "Status": parts[4],
                    "Protocol": parts[5],
                })
    return pd.DataFrame(rows)


def parse_dhcp_pool_count(output: str):
    out = output or ""
    if "% Invalid" in out or "Invalid input" in out:
        return "N/A"
    pools = []
    for line in out.splitlines():
        m = re.match(r"^\s*Pool\s+(.+?)\s*:\s*$", line)
        if m:
            pools.append(m.group(1).strip())
    return len(pools) if pools else 0


def parse_dhcp_binding_count(output: str):
    out = output or ""
    if "% Invalid" in out or "Invalid input" in out:
        return "N/A"
    cnt = 0
    for line in out.splitlines():
        if re.match(r"^\s*\d{1,3}(?:\.\d{1,3}){3}\s+", line):
            cnt += 1
    return cnt


def parse_relay_helpers(output: str) -> pd.DataFrame:
    rows = []
    current_vlan = None
    for line in (output or "").splitlines():
        if line.startswith("interface Vlan"):
            current_vlan = line.split()[-1]
        elif "ip helper-address" in line and current_vlan:
            rows.append({"VLAN": current_vlan, "Helper Address": line.split()[-1]})
    return pd.DataFrame(rows)


# =========================================================
# LLDP / CDP
# =========================================================
def parse_lldp_neighbors_detail(output: str) -> pd.DataFrame:
    rows = []
    out = output or ""
    if not out or "% Invalid" in out or "Invalid input" in out:
        return pd.DataFrame(rows)

    local_intf = None
    sys_name = None
    chassis = None
    port_id = None

    def flush():
        nonlocal local_intf, sys_name, chassis, port_id
        if local_intf and (sys_name or chassis):
            rows.append({
                "Device ID": (sys_name or chassis or "N/A"),
                "Local Intf": local_intf,
                "Remote Port": (port_id or "N/A"),
                "Protocol": "LLDP",
            })
        local_intf = None
        sys_name = None
        chassis = None
        port_id = None

    for line in out.splitlines():
        s = line.strip()
        if not s:
            flush()
            continue

        m = re.search(r"^Local (?:Intf|Interface)\s*:\s*(\S+)", s, re.I)
        if m:
            flush()
            local_intf = m.group(1).strip()
            continue

        m = re.search(r"^System Name\s*:\s*(.+)$", s, re.I)
        if m:
            sys_name = m.group(1).strip()
            continue

        m = re.search(r"^Chassis id\s*:\s*(.+)$", s, re.I)
        if m:
            chassis = m.group(1).strip()
            continue

        m = re.search(r"^Port id\s*:\s*(.+)$", s, re.I)
        if m:
            port_id = m.group(1).strip()
            continue

    flush()
    return pd.DataFrame(rows)


def parse_lldp_neighbors_brief(output: str) -> pd.DataFrame:
    rows = []
    out = output or ""
    if not out or "% Invalid" in out or "Invalid input" in out:
        return pd.DataFrame(rows)

    started = False
    for line in out.splitlines():
        if line.strip().startswith("Device ID"):
            started = True
            continue
        if not started:
            continue
        if not line.strip():
            continue

        parts = re.split(r"\s+", line.strip())
        if len(parts) >= 2:
            rows.append({
                "Device ID": parts[0],
                "Local Intf": parts[1],
                "Remote Port": "N/A",
                "Protocol": "LLDP",
            })
    return pd.DataFrame(rows)


def parse_cdp_neighbors_detail(output: str) -> pd.DataFrame:
    rows = []
    out = output or ""
    if not out or "% Invalid" in out or "Invalid input" in out:
        return pd.DataFrame(rows)

    dev = None
    local_intf = None
    remote_port = None

    def flush():
        nonlocal dev, local_intf, remote_port
        if dev and local_intf:
            rows.append({
                "Device ID": dev,
                "Local Intf": local_intf,
                "Remote Port": (remote_port or "N/A"),
                "Protocol": "CDP",
            })
        dev = None
        local_intf = None
        remote_port = None

    for line in out.splitlines():
        s = line.strip()
        if not s:
            flush()
            continue

        m = re.search(r"^Device ID:\s*(.+)$", s, re.I)
        if m:
            flush()
            dev = m.group(1).strip()
            continue

        m = re.search(r"^Interface:\s*([^,]+),", s, re.I)
        if m:
            local_intf = m.group(1).strip()
            continue

        m = re.search(r"^Port ID\s*\(outgoing port\)\s*:\s*(.+)$", s, re.I)
        if m:
            remote_port = m.group(1).strip()
            continue

    flush()
    return pd.DataFrame(rows)


# =========================================================
# L2NAT
# =========================================================
_IF_PREFIXES = (
    "gi", "te", "fa", "tw", "hu", "et", "po", "lo", "vl", "mg", "fo", "xe", "ge",
    "gigabitethernet", "tengigabitethernet", "twentyfivegigabitethernet", "hundredgigabitethernet",
    "port-channel", "vlan", "loopback", "management"
)

def _looks_like_interface(token: str) -> bool:
    t = (token or "").strip()
    if not t:
        return False
    tl = t.lower()
    if any(tl.startswith(p) for p in _IF_PREFIXES):
        return True
    if "/" in t:
        return True
    return False


def parse_l2nat_instance_table(output: str) -> pd.DataFrame:
    rows = []
    out = output or ""
    if "% Invalid" in out or "Invalid input" in out:
        return pd.DataFrame(rows)

    for line in out.splitlines():
        s = line.strip()
        if not s:
            continue

        m = re.match(r"^(?:Instance\s*:?\s*)(\S+)\b(.*)$", s, flags=re.IGNORECASE)
        if m:
            rows.append({"Instance": m.group(1), "Info": m.group(2).strip()})
            continue

        m2 = re.match(r"^(\d+)\s+(.*)$", s)
        if m2:
            rows.append({"Instance": m2.group(1), "Info": m2.group(2).strip()})
            continue

    return pd.DataFrame(rows)


def parse_l2nat_interface_table(output: str) -> pd.DataFrame:
    rows = []
    out = output or ""
    if "% Invalid" in out or "Invalid input" in out:
        return pd.DataFrame(rows)

    for line in out.splitlines():
        s = line.strip()
        if not s:
            continue
        parts = re.split(r"\s+", s)
        if parts and _looks_like_interface(parts[0]):
            rows.append({"Interface": parts[0], "Info": " ".join(parts[1:])})
    return pd.DataFrame(rows)


def parse_l2nat_statistics_table(output: str) -> pd.DataFrame:
    rows = []
    out = output or ""
    if "% Invalid" in out or "Invalid input" in out:
        return pd.DataFrame(rows)

    for line in out.splitlines():
        s = line.strip()
        if not s:
            continue
        if re.search(r"\d", s) and re.search(r"(pkt|pack|byte|drop|error|rx|tx)", s, flags=re.IGNORECASE):
            rows.append({"Statistic": s})
    return pd.DataFrame(rows)


def l2nat_summary(instance_df: pd.DataFrame, interface_df: pd.DataFrame, stats_raw: str):
    inst_cnt = int(len(instance_df)) if isinstance(instance_df, pd.DataFrame) and not instance_df.empty else 0
    if_cnt = int(len(interface_df)) if isinstance(interface_df, pd.DataFrame) and not interface_df.empty else 0

    out = stats_raw or ""
    if "% Invalid" in out or "Invalid input" in out or not out.strip():
        drops_val = "N/A"
    else:
        drops = 0
        found_any = False
        for line in out.splitlines():
            if re.search(r"drop", line, flags=re.IGNORECASE):
                nums = re.findall(r"\b\d+\b", line)
                if nums:
                    found_any = True
                    drops += sum(int(x) for x in nums)
        drops_val = drops if found_any else 0

    return inst_cnt, if_cnt, drops_val


# =========================================================
# Topology builder (Graphviz DOT)
# =========================================================
def build_topology_dot(overview_df: pd.DataFrame, details_map: dict) -> str:
    ip_to_name = {}
    for _, r in overview_df.iterrows():
        ip = str(r.get("IP", "")).strip()
        hn = str(r.get("Hostname", "")).strip()
        ip_to_name[ip] = hn if hn and hn != "N/A" else ip

    nodes = set(ip_to_name.values())
    edges = set()

    def norm(s: str) -> str:
        return (s or "").strip() or "N/A"

    for ip, d in details_map.items():
        src = ip_to_name.get(ip, ip)

        neigh_df = d.get("lldp_df")
        if neigh_df is None or neigh_df.empty:
            neigh_df = d.get("cdp_df")

        if neigh_df is None or neigh_df.empty:
            continue

        for _, row in neigh_df.iterrows():
            nbr = norm(str(row.get("Device ID", "N/A")))
            local_intf = norm(str(row.get("Local Intf", "N/A")))
            remote_port = norm(str(row.get("Remote Port", "N/A")))

            dst = nbr
            nodes.add(src)
            nodes.add(dst)

            label = local_intf
            if remote_port and remote_port != "N/A":
                label = f"{local_intf} → {remote_port}"

            a, b = sorted([src, dst])
            edges.add((a, b, label))

    dot = [
        "graph Topology {",
        "  rankdir=LR;",
        "  overlap=false;",
        "  splines=true;",
    ]
    for n in sorted(nodes):
        safe = n.replace('"', '\\"')
        dot.append(f'  "{safe}";')

    for a, b, label in sorted(edges):
        sa = a.replace('"', '\\"')
        sb = b.replace('"', '\\"')
        sl = label.replace('"', '\\"')
        dot.append(f'  "{sa}" -- "{sb}" [label="{sl}"];')

    dot.append("}")
    return "\n".join(dot)


# =========================================================
# Worker
# =========================================================
def get_device(base_device: dict):
    overview = {
        "IP": base_device.get("host", "N/A"),
        "Hostname": "N/A",
        "Model": "N/A",
        "Version": "N/A",
        "Serial": "N/A",
        "Uptime": "N/A",
        "Clock": "N/A",
        "VTP_Mode": "N/A",
        "L2NAT_Inst": "N/A",
        "L2NAT_If": "N/A",
        "L2NAT_Drops": "N/A",
        "Status": "OK",
    }

    try:
        conn, dtype = connect_fallback(base_device)
        safe_send(conn, "terminal length 0")

        host_out = safe_send(conn, "show running-config | include ^hostname")
        ver_out = safe_send(conn, "show version")
        clk_out = safe_send(conn, "show clock")
        vtp_out = safe_send(conn, "show vtp status")

        vlan_out = safe_send(conn, "show vlan brief")
        svi_out = safe_send(conn, "show ip interface brief")

        dhcp_pool_out = safe_send(conn, "show ip dhcp pool")
        dhcp_bind_out = safe_send(conn, "show ip dhcp binding")
        relay_out = safe_send(conn, "show running-config | section interface Vlan")

        lldp_detail_out = safe_send(conn, "show lldp neighbors detail")
        lldp_out = safe_send(conn, "show lldp neighbors")
        cdp_detail_out = safe_send(conn, "show cdp neighbors detail")

        l2nat_inst_out = safe_send(conn, "show l2nat instance")
        l2nat_intf_out = safe_send(conn, "show l2nat interface")
        l2nat_stat_out = safe_send(conn, "show l2nat statistics")

        hostname = parse_hostname(host_out)
        if hostname == "N/A" and ver_out:
            m = re.search(r"^\s*(\S+)\s+uptime\s+is\s+", ver_out, flags=re.MULTILINE | re.IGNORECASE)
            if m:
                hostname = m.group(1).strip()

        model, version, serial, uptime = parse_version(ver_out)

        overview["IP"] = str(overview.get("IP", "")).strip()
        overview["Hostname"] = safe_strip(hostname)
        overview["Model"] = safe_strip(model)
        overview["Version"] = safe_strip(version)
        overview["Serial"] = safe_strip(serial)
        overview["Uptime"] = safe_strip(uptime)
        overview["Clock"] = safe_strip(clk_out)
        overview["VTP_Mode"] = safe_strip(parse_vtp_mode(vtp_out))

        vlan_df = parse_vlan_brief(vlan_out)
        svi_df = parse_svi_ip_int_brief(svi_out)
        relay_df = parse_relay_helpers(relay_out)

        dhcp_pools = parse_dhcp_pool_count(dhcp_pool_out)
        dhcp_bindings = parse_dhcp_binding_count(dhcp_bind_out)

        lldp_df = parse_lldp_neighbors_detail(lldp_detail_out)
        if lldp_df.empty:
            lldp_df = parse_lldp_neighbors_brief(lldp_out)

        cdp_df = parse_cdp_neighbors_detail(cdp_detail_out)

        l2nat_inst_df = parse_l2nat_instance_table(l2nat_inst_out)
        l2nat_intf_df = parse_l2nat_interface_table(l2nat_intf_out)
        l2nat_stat_df = parse_l2nat_statistics_table(l2nat_stat_out)
        inst_cnt, if_cnt, drops_val = l2nat_summary(l2nat_inst_df, l2nat_intf_df, l2nat_stat_out)

        overview["L2NAT_Inst"] = inst_cnt
        overview["L2NAT_If"] = if_cnt
        overview["L2NAT_Drops"] = drops_val

        details = {
            "device_type": dtype,

            "vlan_df": vlan_df,
            "svi_df": svi_df,

            "dhcp_pools": dhcp_pools,
            "dhcp_bindings": dhcp_bindings,
            "relay_df": relay_df,

            "lldp_df": lldp_df,
            "cdp_df": cdp_df,

            "l2nat_inst_df": l2nat_inst_df,
            "l2nat_intf_df": l2nat_intf_df,
            "l2nat_stat_df": l2nat_stat_df,

            "raw_l2nat_instance": l2nat_inst_out or "",
            "raw_l2nat_interface": l2nat_intf_out or "",
            "raw_l2nat_statistics": l2nat_stat_out or "",
        }

        try:
            conn.disconnect()
        except Exception:
            pass

        return overview, details

    except Exception as e:
        overview["IP"] = str(overview.get("IP", "")).strip()
        overview["Status"] = f"Error: {e}"
        return overview, None


def style_status(val: str):
    if isinstance(val, str) and val.startswith("Error"):
        return "background-color: #ffcccc; color: #7a0000; font-weight: 700;"
    return "background-color: #ccffcc; color: #006400; font-weight: 700;"


# =========================================================
# ✅ Stable selectbox (不会空白)
# =========================================================
def stable_selectbox_ip(label: str, options: list[str], key: str) -> str:
    clean = []
    for x in options or []:
        s = str(x).strip()
        if s and s.lower() != "nan":
            clean.append(s)

    if not clean:
        st.warning("No selectable devices.")
        return ""

    if key not in st.session_state or str(st.session_state.get(key, "")).strip() not in clean:
        st.session_state[key] = clean[0]

    current = str(st.session_state[key]).strip()
    idx = clean.index(current) if current in clean else 0
    return st.selectbox(label, options=clean, index=idx, key=key)


# =========================================================
# ✅ NEW: Persist results in session_state
# =========================================================
def _sig_from_ips(ips: list[str]) -> str:
    return "|".join([str(x).strip() for x in (ips or []) if str(x).strip()])


def run():
    st.title("Cisco General Information (Read-Only)")
    st.caption("🔒 Read-only module — no configuration commands executed")

    with st.sidebar:
        st.markdown("## SSH Credentials")
        username = st.text_input("SSH Username", value="", key="cg_user")
        password = st.text_input("SSH Password", value="", type="password", key="cg_pass")

        st.markdown("### Concurrency")
        workers = st.number_input("Workers", min_value=1, max_value=50, value=5, step=1, key="cg_workers")

        st.markdown("### Per-device timeout (sec)")
        timeout = st.number_input("Timeout", min_value=3, max_value=120, value=8, step=1, key="cg_timeout")

        st.markdown("### Upload CSV (ip)")
        uploaded = st.file_uploader("CSV file", type=["csv"], accept_multiple_files=False, key="cg_csv")

        # 这两个按钮：一个第一次跑，一个强制重跑
        ready = bool(username.strip()) and bool(password.strip()) and (uploaded is not None)
        c1, c2 = st.columns(2)
        run_btn = c1.button("Run Checks", use_container_width=True, disabled=not ready, key="cg_run")
        rerun_btn = c2.button("Re-run Checks", use_container_width=True, disabled=not ready, key="cg_rerun")

    # -----------------------------------------------------
    # 当用户按下 Run / Re-run：执行采集并存到 session_state
    # -----------------------------------------------------
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

            overview_rows = []
            details_map = {}

            with st.spinner("Running checks... (VLAN/SVI/DHCP/VTP/LLDP/CDP/L2NAT + Topology)"):
                with ThreadPoolExecutor(max_workers=int(workers)) as executor:
                    futures = [executor.submit(get_device, b) for b in base_devices]
                    for f in as_completed(futures):
                        o, d = f.result()
                        o["IP"] = str(o.get("IP", "")).strip()
                        overview_rows.append(o)
                        if d and o["IP"]:
                            details_map[o["IP"]] = d

            df_over = pd.DataFrame(overview_rows)
            if "IP" in df_over.columns:
                df_over["IP"] = df_over["IP"].astype(str).str.strip()
                df_over = df_over.sort_values(by="IP", kind="stable")

            topo_dot = build_topology_dot(df_over, details_map)

            # ✅ 存入 session_state（重点）
            st.session_state["cg_ips_sig"] = ips_sig
            st.session_state["cg_df_over"] = df_over
            st.session_state["cg_details_map"] = details_map
            st.session_state["cg_topo_dot"] = topo_dot

            # 如果之前选的 IP 不在新结果里，重置
            ip_options = df_over["IP"].tolist() if not df_over.empty else []
            if "cg_selected_ip" not in st.session_state or str(st.session_state["cg_selected_ip"]).strip() not in ip_options:
                if ip_options:
                    st.session_state["cg_selected_ip"] = ip_options[0]

        except Exception as e:
            st.error(f"Failed: {e}")
            return

    # -----------------------------------------------------
    # 没按按钮也要显示：从 session_state 取结果（重点）
    # -----------------------------------------------------
    df_over = st.session_state.get("cg_df_over")
    details_map = st.session_state.get("cg_details_map", {})
    topo_dot = st.session_state.get("cg_topo_dot", "")

    if df_over is None or not isinstance(df_over, pd.DataFrame) or df_over.empty:
        st.info("Upload CSV and click **Run Checks** to start.")
        return

    st.subheader("Results (Overview)")
    if "Status" in df_over.columns:
        st.dataframe(df_over.style.applymap(style_status, subset=["Status"]), use_container_width=True)
    else:
        st.dataframe(df_over, use_container_width=True)

    ip_options = df_over["IP"].astype(str).str.strip().tolist()

    # ✅ 现在切换下拉选单，不会再消失/空白（因为不会 return）
    selected_ip = stable_selectbox_ip("Select device", ip_options, key="cg_selected_ip")
    if not selected_ip:
        return

    d = details_map.get(selected_ip)

    tabs = st.tabs(["Topology", "VLAN", "SVI", "DHCP / Relay", "LLDP", "CDP", "L2NAT"])

    with tabs[0]:
        st.markdown("### Topology (LLDP / CDP)")
        st.graphviz_chart(topo_dot, use_container_width=True)
        with st.expander("Raw DOT"):
            st.code(topo_dot, language="dot")

    if not d:
        st.warning("No details available for the selected device (connection or parsing failed).")
    else:
        with tabs[1]:
            st.markdown("### VLAN")
            vlan_df = d.get("vlan_df", pd.DataFrame())
            st.dataframe(vlan_df, use_container_width=True, height=360) if not vlan_df.empty else st.info("No VLAN data parsed.")

        with tabs[2]:
            st.markdown("### SVI (VLAN Interface)")
            svi_df = d.get("svi_df", pd.DataFrame())
            st.dataframe(svi_df, use_container_width=True, height=360) if not svi_df.empty else st.info("No SVI data parsed.")

        with tabs[3]:
            st.markdown("### DHCP Summary")
            st.write(f"DHCP Pools: **{d.get('dhcp_pools', 'N/A')}**")
            st.write(f"DHCP Bindings: **{d.get('dhcp_bindings', 'N/A')}**")
            st.markdown("### DHCP Relay (ip helper-address)")
            relay_df = d.get("relay_df", pd.DataFrame())
            st.dataframe(relay_df, use_container_width=True, height=300) if not relay_df.empty else st.info("No relay helpers found.")

        with tabs[4]:
            st.markdown("### LLDP Neighbors")
            lldp_df = d.get("lldp_df", pd.DataFrame())
            st.dataframe(lldp_df, use_container_width=True, height=360) if not lldp_df.empty else st.info("No LLDP neighbors parsed.")

        with tabs[5]:
            st.markdown("### CDP Neighbors (Detail)")
            cdp_df = d.get("cdp_df", pd.DataFrame())
            st.dataframe(cdp_df, use_container_width=True, height=360) if not cdp_df.empty else st.info("No CDP neighbors parsed.")

        with tabs[6]:
            st.markdown("### L2NAT")
            c1, c2, c3 = st.columns(3)

            inst_df = d.get("l2nat_inst_df", pd.DataFrame())
            intf_df = d.get("l2nat_intf_df", pd.DataFrame())

            drops_val = "N/A"
            try:
                drops_val = df_over.loc[df_over["IP"] == selected_ip, "L2NAT_Drops"].values[0]
            except Exception:
                pass

            c1.metric("Instances", int(inst_df.shape[0]) if isinstance(inst_df, pd.DataFrame) else 0)
            c2.metric("Interfaces", int(intf_df.shape[0]) if isinstance(intf_df, pd.DataFrame) else 0)
            c3.metric("Drops (best-effort)", drops_val)

            st.markdown("#### show l2nat instance")
            st.dataframe(inst_df, use_container_width=True, height=260) if not inst_df.empty else st.info("No instance table parsed.")

            st.markdown("#### show l2nat interface")
            st.dataframe(intf_df, use_container_width=True, height=260) if not intf_df.empty else st.info("No interface table parsed.")

            st.markdown("#### show l2nat statistics")
            stat_df = d.get("l2nat_stat_df", pd.DataFrame())
            st.dataframe(stat_df, use_container_width=True, height=260) if not stat_df.empty else st.info("No statistics parsed.")

            with st.expander("Raw Outputs"):
                st.code(d.get("raw_l2nat_instance", ""), language="text")
                st.code(d.get("raw_l2nat_interface", ""), language="text")
                st.code(d.get("raw_l2nat_statistics", ""), language="text")

    st.download_button(
        "Download Overview (CSV)",
        data=df_over.to_csv(index=False).encode("utf-8-sig"),
        file_name="cisco_general_overview.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    # Only for standalone debug run
    st.set_page_config(page_title="Cisco General Info", layout="wide")
    run()
