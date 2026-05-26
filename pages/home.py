"""pages/home.py — Landing page"""
import streamlit as st
from utils import sidebar_status, try_restore_from_cache, has

def render():
    try_restore_from_cache()
    sidebar_status()

    st.markdown("""
    <div style='padding: 40px 0 20px 0;'>
        <div style='font-family: Space Mono, monospace; font-size: 2.4rem;
                    color: #00c8ff; letter-spacing: -2px; line-height: 1.1;'>
            Dynamic PET / MRI<br>Analysis Suite
        </div>
        <div style='font-size: 0.85rem; color: #5a7080; margin-top: 12px; letter-spacing: 2px;'>
            DEEP LEARNING IN MEDICAL IMAGING — PROJECT DASHBOARD
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Modalities", "PET + MRI", "dynamic + structural")
    with col2:
        st.metric("Pipeline Steps", "3", "load → coreg → segment")
    with col3:
        ready = sum([has("pet_4d"), has("mri_vol"), has("pet_coreg"), has("tumor_mask")])
        st.metric("Steps Ready", f"{ready}/4", "")
    with col4:
        st.metric("AI Model", "MONAI")

    st.markdown("---")

    st.markdown("## 📋 Workflow Guide")

    steps = [
        ("📂", "Load DICOM",       "load_dicom",     has("pet_4d") and has("mri_vol"),
         "Load your PET and MRI DICOM files from the `data/` directory. The app will automatically parse headers, rearrange the 4D pixel array, and cache the results."),
        ("🖼️", "PET Viewer",       "pet_viewer",     has("pet_4d"),
         "Interactively explore the dynamic PET study. Scrub through time frames and slices, switch colormaps, compare last frame vs. temporal average, and export GIF animations."),
        ("🔗", "Coregistration",   "coregistration", has("pet_coreg"),
         "Rigidly coregister the PET average to the MRI reference using Mattes Mutual Information. Visualise the alignment quality with overlay and fusion views."),
        ("🎯", "Segmentation",     "segmentation",   has("tumor_mask"),
         "Define the tumour centre / bounding box and run SAM2 (or the built-in region-growing fallback) to get a 3D mask. Assess segmentation quality numerically."),
        ("📊", "Analysis Dashboard","analysis",       has("tumor_mask"),
         "Integrated dashboard: time-activity curves, intensity histograms, 3D surface renders, and all key quantitative metrics in one place — perfect for your presentation."),
    ]

    for icon, title, page_key, done, desc in steps:
        status_color = "#00e5a0" if done else "#1e2d42"
        status_text  = "READY" if done else "PENDING"
        st.markdown(f"""
        <div style='background:#0e1520; border:1px solid {status_color};
                    border-radius:8px; padding:18px 22px; margin-bottom:12px;
                    display:flex; align-items:flex-start; gap:16px;'>
            <div style='font-size:1.8rem; line-height:1;'>{icon}</div>
            <div style='flex:1;'>
                <div style='display:flex; align-items:center; gap:12px; margin-bottom:6px;'>
                    <span style='font-family:Space Mono,monospace; font-size:0.95rem; color:#c8d8e8;'>{title}</span>
                    <span style='font-family:Space Mono,monospace; font-size:0.62rem;
                                 color:{status_color}; border:1px solid {status_color};
                                 padding:1px 7px; border-radius:3px;'>{status_text}</span>
                </div>
                <div style='font-size:0.82rem; color:#5a7080; line-height:1.6;'>{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 📁 Data Directory Structure")
    st.code("""
pet_project/
├── app.py                  ← Streamlit entry point  (streamlit run app.py)
├── utils.py                ← Shared utilities
├── pages/                  ← One file per page
│   ├── home.py
│   ├── load_dicom.py
│   ├── pet_viewer.py
│   ├── coregistration.py
│   ├── segmentation.py
│   └── analysis.py
├── data/
│   ├── pet/                ← ★ PET .dcm FILES HERE ★
│   │   └── *.dcm
│   └── mri/                ← ★ MRI .dcm FILES HERE ★
│       └── *.dcm
├── outputs/
│   └── .cache/             ← Auto-generated cache )
└── requirements.txt
    """, language="")
