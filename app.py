"""
PET/MRI Analysis Suite — Streamlit Application
===============================================
Run with:
    streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="PET/MRI Analysis Suite",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global dark medical theme ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* Root palette */
:root {
    --bg:        #080c10;
    --panel:     #0e1520;
    --border:    #1e2d42;
    --accent:    #00c8ff;
    --accent2:   #ff6b35;
    --text:      #c8d8e8;
    --text-dim:  #5a7080;
    --success:   #00e5a0;
    --warning:   #ffb547;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--panel) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Headers */
h1, h2, h3 { font-family: 'Space Mono', monospace !important; }
h1 { color: var(--accent) !important; letter-spacing: -1px; }
h2 { color: var(--text) !important; border-bottom: 1px solid var(--border); padding-bottom: 8px; }
h3 { color: var(--accent2) !important; font-size: 0.95rem !important; }

/* Metric cards */
[data-testid="metric-container"] {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 12px 16px !important;
}
[data-testid="metric-container"] label { color: var(--text-dim) !important; font-size: 0.75rem !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: var(--accent) !important; font-family: 'Space Mono', monospace !important; }

/* Buttons */
.stButton > button {
    background: transparent !important;
    border: 1px solid var(--accent) !important;
    color: var(--accent) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 1px !important;
    padding: 8px 20px !important;
    border-radius: 4px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: var(--accent) !important;
    color: var(--bg) !important;
}

/* Primary button */
.stButton > button[kind="primary"] {
    border-color: var(--accent2) !important;
    color: var(--accent2) !important;
}
.stButton > button[kind="primary"]:hover {
    background: var(--accent2) !important;
    color: var(--bg) !important;
}

/* Sliders */
.stSlider [data-baseweb="slider"] { background: var(--border) !important; }
.stSlider [data-baseweb="thumb"] { background: var(--accent) !important; border: none !important; }

/* Select boxes & inputs */
.stSelectbox > div, .stNumberInput > div {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    color: var(--text) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { border-bottom: 1px solid var(--border) !important; background: transparent !important; }
.stTabs [data-baseweb="tab"] { color: var(--text-dim) !important; font-family: 'Space Mono', monospace !important; font-size: 0.75rem !important; }
.stTabs [aria-selected="true"] { color: var(--accent) !important; border-bottom: 2px solid var(--accent) !important; }

/* Expanders */
.streamlit-expanderHeader { color: var(--accent) !important; font-family: 'Space Mono', monospace !important; font-size: 0.8rem !important; }
details { border: 1px solid var(--border) !important; border-radius: 6px !important; }

/* Info boxes */
.stAlert { background: var(--panel) !important; border: 1px solid var(--border) !important; color: var(--text) !important; }

/* Dividers */
hr { border-color: var(--border) !important; }

/* File uploader */
[data-testid="stFileUploader"] {
    border: 1px dashed var(--border) !important;
    border-radius: 8px !important;
    background: var(--panel) !important;
}

/* Hide default streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 20px 0 10px 0;'>
        <div style='font-family: Space Mono, monospace; font-size: 1.3rem; color: #00c8ff; letter-spacing: -1px;'>
            🧠 PET/MRI Suite
        </div>
        <div style='font-size: 0.72rem; color: #5a7080; margin-top: 4px; letter-spacing: 1px;'>
            DEEP LEARNING PROJECT
        </div>
    </div>
    <hr style='border-color:#1e2d42; margin: 8px 0 20px 0;'/>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🏠  Home",
         "📂  Load DICOM",
         "🖼️  PET Viewer",
         "🔗  Coregistration",
         "🎯  Segmentation",
         "📊  Analysis Dashboard"],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:#1e2d42; margin: 20px 0;'/>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size: 0.68rem; color: #5a7080; line-height: 1.8;'>
        <b style='color:#1e2d42;'>──────────────────</b><br>
        Workflow:<br>
        1 → Load DICOM<br>
        2 → Explore PET Viewer<br>
        3 → Run Coregistration<br>
        4 → Segment Tumour<br>
        5 → Dashboard
    </div>
    """, unsafe_allow_html=True)

# ── Page routing ──────────────────────────────────────────────────────────────
if   "Home"            in page: import pages.home            as _p
elif "Load DICOM"      in page: import pages.load_dicom      as _p
elif "PET Viewer"      in page: import pages.pet_viewer      as _p
elif "Coregistration"  in page: import pages.coregistration  as _p
elif "Segmentation"    in page: import pages.segmentation    as _p
elif "Analysis"        in page: import pages.analysis        as _p

_p.render()
