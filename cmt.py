# cmt.py
import os
import importlib.util
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_module(module_name: str, file_path: str):
    """Load a python module by absolute file path (avoid name/path conflicts)."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module spec: {module_name} from {file_path}")
    mod = importlib.util.module_from_spec(spec)
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

# ===== Streamlit page config (ONLY here) =====
st.set_page_config(
    page_title="Cisco Maintain Tools",
    layout="wide",
)

def page_home():
    st.title("Cisco Maintain Tools")
    st.caption("Read-only maintenance & health check utilities")

    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("IE Optical Fiber Health (Read-only)", use_container_width=True):
            st.session_state["page"] = "ie_optical_fiber"
            st.rerun()

    with c2:
        if st.button("Cisco General (Read-only)", use_container_width=True):
            st.session_state["page"] = "cisco_general"
            st.rerun()

    with c3:
        if st.button("Cisco Smart Health Check (Read-only)", use_container_width=True):
            st.session_state["page"] = "cisco_smart_health"
            st.rerun()

def router():
    st.sidebar.title("Cisco Maintain Tools")

    page_key = st.session_state.get("page", "home")
    index_map = {
        "home": 0,
        "ie_optical_fiber": 1,
        "cisco_general": 2,
        "cisco_smart_health": 3,
    }

    menu = st.sidebar.radio(
        "Menu",
        ["Home", "IE Optical Fiber Health", "Cisco General", "Cisco Smart Health Check"],
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

if "page" not in st.session_state:
    st.session_state["page"] = "home"

router()