# cmt.py
import os
import importlib.util
import sys
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_module(module_name: str, file_path: str):
    """Load a python module by absolute file path (avoid name/path conflicts)."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module spec: {module_name} from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    # Ensure the module is registered so introspection (e.g., Streamlit components) can resolve it.
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

# ===== load submodules by exact file paths =====
IE_MOD = load_module(
    "IE_OpticalFiber_health",
    os.path.join(BASE_DIR, "Cisco_IE", "IE_OpticalFiber_health.py")
)

CG_MOD = load_module(
    "Cisco_General",
    os.path.join(BASE_DIR, "Cisco_General", "Cisco_General.py")
)

# ✅ NEW: Cisco Smart Health Check (inside Cisco_General folder)
SMART_MOD = load_module(
    "Cisco_Smart_health_check",
    os.path.join(BASE_DIR, "Cisco_General", "Cisco_Smart_health_check.py")
)

ZABBIX_MOD = load_module(
    "Zabbix_Map",
    os.path.join(BASE_DIR, "Network Architecture Diagram", "zabbix_map.py")
)

# ===== Streamlit page config (ONLY here) =====
st.set_page_config(
    page_title="Cisco Maintain Tools",
    layout="wide",
)

def apply_theme():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Source+Sans+3:wght@400;600;700&display=swap');

        :root {
            --bg-1: #0b0f14;
            --bg-2: #0f1722;
            --bg-3: #132033;
            --ink-1: #e7eef7;
            --ink-2: #b9c7d8;
            --accent: #35c2ff;
            --card: rgba(20, 31, 45, 0.75);
            --stroke: rgba(255, 255, 255, 0.08);
            --shadow: 0 12px 32px rgba(0, 0, 0, 0.35);
        }

        .stApp {
            background:
                radial-gradient(900px 420px at 10% -5%, rgba(53, 194, 255, 0.18), transparent 60%),
                radial-gradient(700px 360px at 90% 0%, rgba(116, 179, 255, 0.15), transparent 60%),
                linear-gradient(160deg, var(--bg-1), var(--bg-2) 45%, var(--bg-3));
            color: var(--ink-1);
            font-family: "Source Sans 3", system-ui, -apple-system, "Segoe UI", sans-serif;
        }

        h1, h2, h3, h4, h5 {
            font-family: "Space Grotesk", "Source Sans 3", system-ui, sans-serif;
            letter-spacing: -0.02em;
        }

        .block-container {
            padding-top: 2.5rem;
            padding-bottom: 3rem;
        }

        .hero {
            padding: 2.25rem 2.5rem;
            border-radius: 20px;
            background: rgba(14, 21, 32, 0.7);
            border: 1px solid var(--stroke);
            box-shadow: var(--shadow);
            margin-bottom: 1.5rem;
        }

        .hero small {
            display: inline-block;
            font-weight: 600;
            color: var(--accent);
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.5rem;
        }

        .hero h1 {
            font-size: 2.6rem;
            margin-bottom: 0.4rem;
        }

        .hero p {
            font-size: 1.05rem;
            color: var(--ink-2);
            max-width: 58ch;
        }

        .card {
            padding: 1.6rem 1.7rem;
          
          No tasks in progress
          
          
          
          Add Move To Panel 插件
          1m
          
          Codex task
          17m
          
          Codex task
          21h
          View all (11)
            border-radius: 18px;
            background: var(--card);
            border: 1px solid var(--stroke);
            box-shadow: var(--shadow);
            height: 100%;
        }

        .card h3 {
            margin-bottom: 0.4rem;
        }

        .card p {
            color: var(--ink-2);
            font-size: 0.95rem;
            min-height: 3.6rem;
        }

        .card .meta {
            font-size: 0.85rem;
            color: rgba(185, 199, 216, 0.8);
            margin-bottom: 1rem;
        }

        .stButton button {
            background: linear-gradient(135deg, #2ea8ff, #46d1ff);
            color: #06121f;
            font-weight: 700;
            border: none;
            border-radius: 12px;
            padding: 0.6rem 1rem;
            transition: transform 0.12s ease, box-shadow 0.12s ease;
            box-shadow: 0 10px 22px rgba(46, 168, 255, 0.35);
        }

        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 14px 28px rgba(46, 168, 255, 0.4);
        }

        .stButton button:focus {
            outline: 2px solid rgba(53, 194, 255, 0.6);
            outline-offset: 2px;
        }

        [data-testid="stSidebar"] {
            background: rgba(8, 12, 18, 0.7);
            border-right: 1px solid rgba(255, 255, 255, 0.06);
        }

        .sidebar-title {
            font-family: "Space Grotesk", "Source Sans 3", system-ui, sans-serif;
            font-size: 1.2rem;
            margin-bottom: 0.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def page_home():
    apply_theme()
    st.markdown(
        """
        <section class="hero">
            <small>Maintenance Console</small>
            <h1>Cisco Maintain Tools</h1>
            <p>
                Read-only health checks and diagnostics, presented as focused workflows
                for fast triage and consistent reporting.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="card">
                <h3>IE Optical Fiber Health</h3>
                <div class="meta">Signal quality and optical compliance</div>
                <p>Validate fiber integrity, transceiver readings, and alert thresholds.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open IE Optical Fiber Health", use_container_width=True):
            st.session_state["page"] = "ie_optical_fiber"
            st.rerun()

    with c2:
        st.markdown(
            """
            <div class="card">
                <h3>Cisco General</h3>
                <div class="meta">Inventory, CPU, memory, and system checks</div>
                <p>Run standard diagnostics across platform health and baseline status.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open Cisco General", use_container_width=True):
            st.session_state["page"] = "cisco_general"
            st.rerun()

    with c3:
        st.markdown(
            """
            <div class="card">
                <h3>Cisco Smart Health Check</h3>
                <div class="meta">Best-practice validations</div>
                <p>Automated checks for configuration, compliance, and risk signals.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open Cisco Smart Health Check", use_container_width=True):
            st.session_state["page"] = "cisco_smart_health"
            st.rerun()

    c4, c5, c6 = st.columns(3)
    with c4:
        st.markdown(
            """
            <div class="card">
                <h3>Network Architecture Diagram</h3>
                <div class="meta">Zabbix MAP style topology</div>
                <p>Interactive map with ping status, alerts, and grouping.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open Network Architecture Diagram", use_container_width=True):
            st.session_state["page"] = "zabbix_map"
            st.rerun()

def router():
    apply_theme()
    st.sidebar.markdown('<div class="sidebar-title">Cisco Maintain Tools</div>', unsafe_allow_html=True)

    page_key = st.session_state.get("page", "home")
    index_map = {
        "home": 0,
        "ie_optical_fiber": 1,
        "cisco_general": 2,
        "cisco_smart_health": 3,
        "zabbix_map": 4,
    }

    menu = st.sidebar.radio(
        "Menu",
        ["Home", "IE Optical Fiber Health", "Cisco General", "Cisco Smart Health Check", "Network Architecture Diagram"],
        index=index_map.get(page_key, 0),
    )

    if menu == "Home":
        st.session_state["page"] = "home"
        page_home()
        return

    if menu == "IE Optical Fiber Health":
        st.session_state["page"] = "ie_optical_fiber"
        if not hasattr(IE_MOD, "run"):
            st.error("Cisco_IE/IE_OpticalFiber_health.py does not provide: def run():")
            return
        IE_MOD.run()
        return

    if menu == "Cisco General":
        st.session_state["page"] = "cisco_general"
        if not hasattr(CG_MOD, "run"):
            st.error("Cisco_General/Cisco_General.py does not provide: def run():")
            return
        CG_MOD.run()
        return

    if menu == "Cisco Smart Health Check":
        st.session_state["page"] = "cisco_smart_health"
        if not hasattr(SMART_MOD, "run"):
            st.error("Cisco_General/Cisco_Smart_health_check.py does not provide: def run():")
            return
        SMART_MOD.run()
        return

    if menu == "Network Architecture Diagram":
        st.session_state["page"] = "zabbix_map"
        if not hasattr(ZABBIX_MOD, "run"):
            st.error("Network Architecture Diagram/zabbix_map.py does not provide: def run():")
            return
        ZABBIX_MOD.run()
        return

if "page" not in st.session_state:
    st.session_state["page"] = "home"

router()
