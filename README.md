# 🧠 Dynamic PET / MRI Analysis Suite

> **Deep Learning in Medical Imaging 

A full interactive pipeline for dynamic PET/MRI analysis: DICOM loading, 3D rigid coregistration, and AI-powered tumour segmentation — all accessible through a Streamlit web application.

---

## 📋 Project Overview

This project implements three main objectives from the course proposal:

| Objective | Description |
|-----------|-------------|
| **1 — DICOM Loading & Visualisation** | Load dynamic PET + MRI, rearrange pixel arrays from headers, animate 3 median planes |
| **2 — 3D Rigid Coregistration** | Register PET average → MRI using Mattes MI, generate rotating MIP animation |
| **3 — AI Tumour Segmentation** | Semi-automatic 3D segmentation via SAM2/MedSAM2 with bbox/centroid prompt |
| **Submission** | Report + GitHub | 
**Dataset used:**
- PET: `1 BRAIN DINAMIC COLINA` (dynamic brain PET study)
- MRI: `AX 3D T1` (structural MRI reference)

---

## 🗂️ Project Structure

```
pet_project/
│
├── 🚀 app.py                        ← Streamlit entry point  (run this)
├── utils.py                         ← Shared state, loaders, plot helpers
├── requirements.txt                 ← Python dependencies
│
├── .streamlit/
│   └── config.toml                  ← Upload size limit + theme config
│
├── pages/                           ← One file per Streamlit page
│   ├── home.py                      ← Landing page + workflow guide
│   ├── load_dicom.py                ← Upload PET & MRI DICOM files
│   ├── pet_viewer.py                ← Interactive 4D PET explorer
│   ├── coregistration.py            ← Rigid registration + rotating MIP
│   ├── segmentation.py              ← AI tumour segmentation
│   └── analysis.py                  ← Dashboard: TAC, histograms, 3D surface
│
├── 01_dicom_loading.py              ← Standalone script 
├── 02_coregistration.py             ← Standalone script 
├── 03_segmentation.py               ← Standalone script 
│
├── data/                            ← Local data directory (optional)
│   ├── pet/                         ← Place PET .dcm files here (local run)
│   └── mri/                         ← Place MRI .dcm files here (local run)
│
└── outputs/
    └── .cache/                      ← Auto-generated session cache
```

---

## Quickstart — Streamlit App

The Streamlit app is the **main deliverable**. It runs locally, also available on cloud (https://mari-pet-analysis.streamlit.app).

### 1. Install dependencies


# Install Python packages
pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

### 3. Load your data

In the app, go to **📂 Load DICOM** and upload your files:


### 4. Follow the workflow

```
📂 Load DICOM  →  🖼️ PET Viewer  →  🔗 Coregistration  →  🎯 Segmentation  →  📊 Dashboard
```

---

## 🌐 Cloud Deployment (Streamlit Community Cloud)

A similar version of the app is also deployed in streamlit cloud (Link here: https://mari-pet-analysis.streamlit.app ), but the limitations is that it can only take less than 200 MB dicom images.


---

## 🖥️ App Pages

### 📂 Load DICOM
- Upload PET and MRI as `.zip` or individual `.dcm` files
- Automatic DICOM header parsing (enhanced single-file + classic multi-file)
- Reads all required headers: `NumberOfFrames`, `PixelSpacing`, `SpacingBetweenSlices`, `FramePositionsVector`, `FrameStartTimes`, `FrameDurations`
- Instant preview of loaded volumes
- Session reset button

### 🖼️ PET Viewer
- **Frame Explorer:** scrub through time frames and slices with sliders
- **Orthogonal Navigator:** axial / coronal / sagittal with crosshair overlay
- **Avg vs Last:** side-by-side comparison + difference map
- **Export GIF:** downloadable animation of any plane across all time frames

### 🔗 Coregistration
- One-click 3D rigid registration (PET average → MRI)
- Mattes Mutual Information metric, gradient descent optimiser, 3-level pyramid
- Live progress bar during registration
- Alignment viewer with adjustable alpha fusion
- **Rotating MIP GIF:** 36-angle coronal-sagittal sweep — MRI / PET / fusion panels

### 🎯 Segmentation
- Browse PET last frame to locate the tumour hot-spot
- Enter centroid `(z, y, x)` and bounding box coordinates
- Run **SAM2 / MedSAM2** (AI) or region-growing fallback
- Orthogonal overlay with contour + alpha mask
- Sweep GIF export around the tumour region

### 📊 Analysis Dashboard
- **Time-Activity Curves** with custom spherical VOIs
- **Intensity histograms** with tumour voxel overlay
- **Multi-modal overview** (MRI / PET / fusion / mask)
- **3D surface render** of the tumour (rotatable elevation + azimuth)
- **Summary report** — downloadable `.md` file

---

## 💻 Optional: Standalone CLI Scripts

The three standalone scripts can be run independently of the app:

```bash
# Step 1 — Load and visualise PET
python 01_dicom_loading.py --pet_dir data/pet --out_dir outputs

# Step 2 — Coregister PET → MRI
python 02_coregistration.py --mri_dir data/mri --out_dir outputs

# Step 3 — Segment tumour (with SAM2)
python 03_segmentation.py \
    --centroid 45,128,140 \
    --bbox 35,110,120,55,145,160 \
    --checkpoint models/sam2.1_hiera_large.pt

# Step 3 — Segment tumour (without SAM2, region-growing fallback)
python 03_segmentation.py \
    --centroid 45,128,140 \
    --bbox 35,110,120,55,145,160
```

Outputs are saved to `outputs/` including PNG figures, GIF animations, and a metrics JSON.

---


## Notes

- Results are **cached** in `outputs/.cache/` — delete this folder to force reprocessing
- The app restores previous results automatically on reload within the same session
- PET and MRI can have different voxel dimensions — the app handles this automatically
- For cloud deployment, uploaded files are processed in-memory and not stored permanently

---
