# IE_OpticalFiber_health.py
# Read-only Cisco IE Batch Optical Fiber Health Check (Submodule for cmt.py)
# - Read-only command allowlist: "terminal length 0", "show ..."
# - Upload CSV (ip,interface)
# - Show Hostname
# - Rx/Tx/Temp/Loss/Elapsed: force 2 decimals in UI
# - Rx/Tx/Temp/Loss cell colors: Green=Normal, Yellow=Attention, Red=Critical
#
# IMPORTANT:
# - Do NOT call st.set_page_config() here (handled by cmt.py)
# - Provide def run(): entry point

import re
import time
import streamlit as st
from netmiko import ConnectHandler
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed


# ------------------------------
# Read-only command allowlist (hard block)
# ------------------------------
def safe_send(conn, cmd: str) -> str:
    c = (cmd or "").strip()
    lc = c.lower()
    if not (lc.startswith("show ") or lc == "terminal length 0"):
        raise ValueError(f"Blocked non-readonly command: {cmd}")
    return conn.send_command(c)


# ------------------------------
# Thresholds (adjust if needed)
# ------------------------------
RX_WARN = -18.0
RX_CRIT = -22.0
TEMP_WARN = 60.0
TEMP_CRIT = 70.0
LOSS_WARN = 8.0
LOSS_CRIT = 12.0


# ------------------------------
# UI display format: force 2 decimals
# ------------------------------
TWO_DEC_FMT = {
    "Rx(dBm)": "{:.2f}",
    "Tx(dBm)": "{:.2f}",
    "Temp(°C)": "{:.2f}",
    "Loss(dB)": "{:.2f}",
    "Elapsed(s)": "{:.2f}",
}


def r2(v):
    """Data layer: keep 2 decimals; None stays None."""
    return None if v is None else round(float(v), 2)


# ------------------------------
# Parse hostname (read-only)
# ------------------------------
def parse_hostname(output: str) -> str:
    m = re.search(r"^hostname\s+(\S+)", output or "", re.M)
    return m.group(1) if m else "N/A"


# ------------------------------
# Parse: show interfaces transceiver (table)
# Typical row: Gi1/0/28  33.0  3.29  3.2  -5.72  -11.71
# We want: Temperature / Tx Power / Rx Power
# ------------------------------
def parse_transceiver_table(output: str, interface: str):
    iface = (interface or "").strip()
    if not iface:
        return {}

    # Convert long name to short if needed
    short = "Gi" + iface[len("GigabitEthernet"):] if iface.lower().startswith("gigabitethernet") else iface

    patterns = [
        rf"^{re.escape(short)}\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)",
        rf"^{re.escape(iface)}\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)",
    ]

    for line in (output or "").splitlines():
        line = line.strip()
        for pat in patterns:
            m = re.match(pat, line)
            if m:
                # columns (commonly): Temp, Voltage, Current, Tx, Rx
                return {
                    "temperature_c": float(m.group(1)),
                    "tx_dbm": float(m.group(4)),
                    "rx_dbm": float(m.group(5)),
                }
    return {}


# ------------------------------
# Parse: show interfaces <port>
# ------------------------------
def parse_show_interfaces(output: str):
    out = output or ""

    def gi(p):
        m = re.search(p, out, re.I)
        return int(m.group(1)) if m else None

    data = {
        "input_errors": gi(r"(\d+)\s+input errors"),
        "crc": gi(r"(\d+)\s+CRC") or gi(r"CRC\s*[:=]\s*(\d+)"),
        "output_errors": gi(r"(\d+)\s+output errors"),
        "interface_resets": gi(r"(\d+)\s+interface resets"),
    }

    m = re.search(r"\bis\s+(up|down)\b,\s+line protocol is\s+(up|down)", out, re.I)
    data["link_state"] = f"{m.group(1).lower()}/{m.group(2).lower()}" if m else "unknown"
    return data


# ------------------------------
# Severity levels (green/yellow/red/na)
# ------------------------------
def _is_na(v):
    return v is None or (isinstance(v, float) and pd.isna(v))


def level_rx(power_dbm):
    if _is_na(power_dbm):
        return "na"
    v = float(power_dbm)
    if v < RX_CRIT:
        return "red"
    if v < RX_WARN:
        return "yellow"
    return "green"


def level_temp(temp_c):
    if _is_na(temp_c):
        return "na"
    v = float(temp_c)
    if v > TEMP_CRIT:
        return "red"
    if v > TEMP_WARN:
        return "yellow"
    return "green"


def level_loss(loss_db):
    if _is_na(loss_db):
        return "na"
    v = float(loss_db)
    if v > LOSS_CRIT:
        return "red"
    if v >= LOSS_WARN:
        return "yellow"
    return "green"


def color_cell(lv: str) -> str:
    if lv == "green":
        return "background-color: rgba(0, 255, 0, 0.18)"
    if lv == "yellow":
        return "background-color: rgba(255, 255, 0, 0.18)"
    if lv == "red":
        return "background-color: rgba(255, 0, 0, 0.22)"
    return ""


# ------------------------------
# Overall verdict
# ------------------------------
def verdict_and_reason(rx, tx, temp, loss, crc, in_err, out_err, resets, dom_ok):
    reasons = []
    overall = "OK"

    if not dom_ok:
        reasons.append("DOM=N/A")

    if "red" in (level_rx(rx), level_rx(tx), level_temp(temp), level_loss(loss)):
        overall = "CRIT"
    elif "yellow" in (level_rx(rx), level_rx(tx), level_temp(temp), level_loss(loss)):
        overall = "WARN"

    for k, v in (("CRC", crc), ("in_err", in_err), ("out_err", out_err), ("resets", resets)):
        if v is not None and v > 0:
            if overall == "OK":
                overall = "WARN"
            reasons.append(f"{k}={v}")

    return overall, "; ".join(reasons) if reasons else "No issues"


# ------------------------------
# Single task: connect and check one interface (read-only)
# ------------------------------
def check_one(ip: str, interface: str, username: str, password: str, timeout_s: int = 8):
    start = time.time()

    result = {
        "Device": ip,
        "Hostname": "N/A",
        "Port": interface,
        "Link": "unknown",
        "Rx(dBm)": None,
        "Tx(dBm)": None,
        "Temp(°C)": None,
        "Loss(dB)": None,
        "CRC": None,
        "input errors": None,
        "output errors": None,
        "interface resets": None,
        "Verdict": "ERROR",
        "Reason": "",
        "Elapsed(s)": None,
    }

    try:
        conn = ConnectHandler(
            device_type="cisco_ios",
            host=ip,
            username=username,
            password=password,
            fast_cli=False,
            timeout=int(timeout_s),
        )

        safe_send(conn, "terminal length 0")

        # Hostname (read-only)
        h_out = safe_send(conn, "show running-config | include ^hostname")
        result["Hostname"] = parse_hostname(h_out)

        # DOM (one-shot table)
        trans_all = safe_send(conn, "show interfaces transceiver")
        optic = parse_transceiver_table(trans_all, interface)
        dom_ok = bool(optic)

        rx = optic.get("rx_dbm")
        tx = optic.get("tx_dbm")
        temp = optic.get("temperature_c")

        loss = None
        if tx is not None and rx is not None:
            loss = tx - rx

        # Interface counters
        intf_out = safe_send(conn, f"show interfaces {interface}")
        try:
            conn.disconnect()
        except Exception:
            pass

        iface = parse_show_interfaces(intf_out)

        result["Link"] = iface.get("link_state", "unknown")
        result["CRC"] = iface.get("crc")
        result["input errors"] = iface.get("input_errors")
        result["output errors"] = iface.get("output_errors")
        result["interface resets"] = iface.get("interface_resets")

        # DOM values (data layer 2dp)
        result["Rx(dBm)"] = r2(rx)
        result["Tx(dBm)"] = r2(tx)
        result["Temp(°C)"] = r2(temp)
        result["Loss(dB)"] = r2(loss)

        overall, reason = verdict_and_reason(
            rx=rx, tx=tx, temp=temp, loss=loss,
            crc=result["CRC"], in_err=result["input errors"],
            out_err=result["output errors"], resets=result["interface resets"],
            dom_ok=dom_ok
        )
        result["Verdict"] = overall
        result["Reason"] = reason

    except Exception as e:
        result["Verdict"] = "ERROR"
        result["Reason"] = str(e)

    result["Elapsed(s)"] = r2(time.time() - start)
    return result


# ------------------------------
# DataFrame styling (Rx/Tx/Temp/Loss)
# ------------------------------
def style_table(df: pd.DataFrame) -> pd.DataFrame:
    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    for i in df.index:
        rx = df.at[i, "Rx(dBm)"]
        tx = df.at[i, "Tx(dBm)"]
        temp = df.at[i, "Temp(°C)"]
        loss = df.at[i, "Loss(dB)"]

        styles.at[i, "Rx(dBm)"] = color_cell(level_rx(rx))
        styles.at[i, "Tx(dBm)"] = color_cell(level_rx(tx))
        styles.at[i, "Temp(°C)"] = color_cell(level_temp(temp))
        styles.at[i, "Loss(dB)"] = color_cell(level_loss(loss))
    return styles


# ------------------------------
# UI entry (callable submodule)
# ------------------------------
def run():
    st.title("IE Optical Fiber Health (Read-only)")
    st.caption("🔒 Read-only module — show commands only")

    with st.sidebar:
        st.header("SSH Credentials")
        username = st.text_input("SSH Username", "admin", key="ieh_user")
        password = st.text_input("SSH Password", type="password", key="ieh_pass")
        max_workers = st.number_input("Concurrency", min_value=1, max_value=20, value=5, step=1, key="ieh_workers")
        timeout_s = st.number_input("Per-device timeout (sec)", min_value=3, max_value=30, value=8, step=1, key="ieh_timeout")
        uploaded = st.file_uploader("Upload CSV (ip,interface)", type=["csv"], key="ieh_csv")
        run_btn = st.button("Run Checks", type="primary", disabled=(uploaded is None), key="ieh_run")

    df_in = None
    if uploaded is not None:
        df_in = pd.read_csv(uploaded)
        if not {"ip", "interface"}.issubset(set(df_in.columns)):
            st.error("CSV must contain two columns: ip, interface")
            st.stop()
        st.subheader("Target List")
        st.dataframe(df_in, use_container_width=True, hide_index=True)

    if run_btn and uploaded is not None:
        rows = df_in.to_dict("records")
        total = len(rows)
        results = []

        st.info(f"Ready: {total} checks ({df_in['ip'].nunique()} devices)")
        prog = st.progress(0)
        done = 0

        with ThreadPoolExecutor(max_workers=int(max_workers)) as ex:
            futures = [
                ex.submit(check_one, r["ip"], r["interface"], username, password, int(timeout_s))
                for r in rows
            ]
            for f in as_completed(futures):
                results.append(f.result())
                done += 1
                prog.progress(done / total)

        df_out = pd.DataFrame(results)

        # Sort by severity: CRIT/WARN/OK/ERROR
        order = {"CRIT": 0, "WARN": 1, "OK": 2, "ERROR": 3}
        df_out["__rank"] = df_out["Verdict"].map(order).fillna(99)
        df_out = df_out.sort_values(["__rank", "Device", "Port"]).drop(columns=["__rank"])

        st.subheader("Results (forced 2 decimals)")

        styled = (
            df_out
            .style
            .format(TWO_DEC_FMT)  # force 2 decimals in UI
            .apply(lambda _: style_table(df_out), axis=None)  # color cells
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CRIT", int((df_out["Verdict"] == "CRIT").sum()))
        c2.metric("WARN", int((df_out["Verdict"] == "WARN").sum()))
        c3.metric("OK", int((df_out["Verdict"] == "OK").sum()))
        c4.metric("ERROR", int((df_out["Verdict"] == "ERROR").sum()))

        st.download_button(
            "Download CSV",
            df_out.to_csv(index=False).encode("utf-8-sig"),
            "health_results.csv",
            "text/csv"
        )


if __name__ == "__main__":
    # Standalone debug run ONLY
    st.set_page_config(page_title="IE Optical Fiber Health", layout="wide")
    run()
