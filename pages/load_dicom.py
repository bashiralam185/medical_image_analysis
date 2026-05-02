"""pages/load_dicom.py — Load PET & MRI DICOM files"""
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pydicom
import streamlit as st
from tqdm import tqdm

import utils

warnings.filterwarnings("ignore")


# ── DICOM parsing ( 01_dicom_loading.py) ────────────────────────

def _safe_tag(ds, group, element):
    try:
        return list(ds[group, element].value)
    except Exception:
        return None


def _parse_enhanced_pet(ds):
    n_frames_total = int(ds[0x0028, 0x0008].value)
    rows           = int(ds[0x0028, 0x0010].value)
    cols           = int(ds[0x0028, 0x0011].value)

    frame_positions   = _safe_tag(ds, 0x0055, 0x1002)
    frame_start_times = _safe_tag(ds, 0x0055, 0x1001)
    frame_durations   = _safe_tag(ds, 0x0055, 0x1004)

    pixel_spacing = [float(v) for v in ds[0x0028, 0x0030].value]
    try:
        slice_thickness = float(ds[0x0018, 0x0088].value)
    except Exception:
        slice_thickness = 1.0

    if frame_positions is not None:
        fp = np.array(frame_positions, dtype=float)
        if fp.ndim == 1:
            fp = fp.reshape(-1, 3)
        z_positions = fp[:, 2]
        unique_z    = np.unique(np.round(z_positions, decimals=2))
        n_slices    = len(unique_z)
        n_time      = n_frames_total // n_slices
    elif frame_start_times is not None:
        n_time   = len(np.unique(frame_start_times))
        n_slices = n_frames_total // n_time
    else:
        n_slices = int(np.sqrt(n_frames_total))
        n_time   = n_frames_total // n_slices

    raw       = ds.pixel_array.astype(np.float32)
    slope     = float(getattr(ds, "RescaleSlope",     1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    raw       = raw * slope + intercept
    pet_4d    = raw.reshape(n_time, n_slices, rows, cols)

    return {
        "pet_4d": pet_4d, "n_time": n_time, "n_slices": n_slices,
        "rows": rows, "cols": cols,
        "pixel_spacing": pixel_spacing, "slice_thickness": slice_thickness,
        "frame_start_times": frame_start_times,
        "frame_durations": frame_durations,
    }


def _parse_classic_pet(dcm_files):
    slices = [pydicom.dcmread(str(f)) for f in dcm_files]

    def sort_key(ds):
        t = int(getattr(ds, "TemporalPositionIdentifier", 1))
        z = float(ds.ImagePositionPatient[2]) if hasattr(ds, "ImagePositionPatient") else 0.0
        return (t, z)

    slices.sort(key=sort_key)
    rows = int(slices[0][0x0028, 0x0010].value)
    cols = int(slices[0][0x0028, 0x0011].value)

    t_ids = sorted(set(int(getattr(s, "TemporalPositionIdentifier", 1)) for s in slices))
    z_pos = sorted(set(round(float(s.ImagePositionPatient[2]), 2)
                        for s in slices if hasattr(s, "ImagePositionPatient")))
    n_time   = len(t_ids)
    n_slices = len(z_pos)

    pet_4d = np.zeros((n_time, n_slices, rows, cols), dtype=np.float32)
    for ds in slices:
        t  = t_ids.index(int(getattr(ds, "TemporalPositionIdentifier", 1)))
        zv = round(float(ds.ImagePositionPatient[2]), 2)
        z  = z_pos.index(zv)
        sl = ds.pixel_array.astype(np.float32)
        pet_4d[t, z] = sl * float(getattr(ds, "RescaleSlope", 1)) + float(getattr(ds, "RescaleIntercept", 0))

    pixel_spacing   = [float(v) for v in slices[0][0x0028, 0x0030].value]
    try:
        slice_thickness = float(slices[0][0x0018, 0x0088].value)
    except Exception:
        slice_thickness = 1.0

    return {
        "pet_4d": pet_4d, "n_time": n_time, "n_slices": n_slices,
        "rows": rows, "cols": cols,
        "pixel_spacing": pixel_spacing, "slice_thickness": slice_thickness,
        "frame_start_times": None, "frame_durations": None,
    }


def load_pet_from_dir(pet_dir: Path) -> dict:
    dcm_files = sorted(pet_dir.glob("**/*.dcm"))
    if not dcm_files:
        raise FileNotFoundError(f"No .dcm files in {pet_dir}")
    if len(dcm_files) == 1:
        ds = pydicom.dcmread(str(dcm_files[0]))
        return _parse_enhanced_pet(ds)
    return _parse_classic_pet(dcm_files)


def load_mri_from_dir(mri_dir: Path) -> np.ndarray:
    import SimpleITK as sitk
    reader = sitk.ImageSeriesReader()
    dcm_names = reader.GetGDCMSeriesFileNames(str(mri_dir))
    if not dcm_names:
        dcm_names = sorted([str(f) for f in mri_dir.glob("**/*.dcm")])
    reader.SetFileNames(dcm_names)
    img = reader.Execute()
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    spacing = img.GetSpacing()   # (X, Y, Z)
    return arr, spacing


# ── Page ─────────────────────────────────────────────────────────────────────

def render():
    utils.try_restore_from_cache()
    utils.sidebar_status()

    st.title("📂 Load DICOM Data")
    st.markdown(
        "<p style='color:#5a7080;'>Parse and cache PET and MRI DICOM studies. "
        "Files must be placed in the <code>data/</code> subdirectories.</p>",
        unsafe_allow_html=True,
    )

    # ── Data location info ────────────────────────────────────
    st.markdown("### 📁 Where to put your files")
    c1, c2 = st.columns(2)
    with c1:
        pet_count = len(list(utils.PET_DIR.glob("**/*.dcm")))
        color = "#00e5a0" if pet_count > 0 else "#ff6b35"
        st.markdown(f"""
        <div style='background:#0e1520; border:1px solid {color}; border-radius:8px; padding:16px;'>
            <div style='font-family:Space Mono,monospace; font-size:0.8rem; color:{color};'>
                PET DICOM DIRECTORY
            </div>
            <div style='font-family:Space Mono,monospace; font-size:1rem; color:#c8d8e8; margin:8px 0;'>
                data/pet/
            </div>
            <div style='font-size:0.8rem; color:#5a7080;'>
                Study: <b>1 BRAIN DINAMIC COLINA</b><br>
                Files found: <b style='color:{color};'>{pet_count} .dcm</b>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        mri_count = len(list(utils.MRI_DIR.glob("**/*.dcm")))
        color = "#00e5a0" if mri_count > 0 else "#ff6b35"
        st.markdown(f"""
        <div style='background:#0e1520; border:1px solid {color}; border-radius:8px; padding:16px;'>
            <div style='font-family:Space Mono,monospace; font-size:0.8rem; color:{color};'>
                MRI DICOM DIRECTORY
            </div>
            <div style='font-family:Space Mono,monospace; font-size:1rem; color:#c8d8e8; margin:8px 0;'>
                data/mri/
            </div>
            <div style='font-size:0.8rem; color:#5a7080;'>
                Study: <b>AX 3D T1</b><br>
                Files found: <b style='color:{color};'>{mri_count} .dcm</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Load PET ─────────────────────────────────────────────
    st.markdown("### 1 — Load Dynamic PET")
    col_a, col_b = st.columns([2, 1])
    with col_a:
        if utils.has("pet_4d"):
            meta = utils.get("pet_meta")
            st.success(f"✓ PET loaded — shape {utils.get('pet_4d').shape}  "
                       f"({meta['n_time']} frames × {meta['n_slices']} slices)")
        else:
            st.warning("PET not yet loaded.")
    with col_b:
        load_pet_btn = st.button("Load PET", use_container_width=True)

    if load_pet_btn:
        if not list(utils.PET_DIR.glob("**/*.dcm")):
            st.error(f"No .dcm files found in `{utils.PET_DIR}`. "
                     "Please copy your PET DICOM files there first.")
        else:
            with st.spinner("Parsing PET DICOM…"):
                try:
                    data = load_pet_from_dir(utils.PET_DIR)
                    pet_4d  = utils.ensure_4d(data["pet_4d"])
                    pet_avg  = utils.ensure_3d(pet_4d.mean(axis=0))
                    pet_last = utils.ensure_3d(pet_4d[-1])
                    meta = {k: data[k] for k in
                            ["pixel_spacing", "slice_thickness", "n_time", "n_slices",
                             "rows", "cols", "frame_start_times", "frame_durations"]}

                    utils.put("pet_4d",  pet_4d)
                    utils.put("pet_avg", pet_avg)
                    utils.put("pet_last", pet_last)
                    utils.put("pet_meta", meta)

                    utils.save_cache("pet_4d",  pet_4d)
                    utils.save_cache("pet_avg",  pet_avg)
                    utils.save_cache("pet_last", pet_last)
                    np.save(str(utils.CACHE_DIR / "pet_meta.npy"), meta, allow_pickle=True)

                    st.success(f"✓ PET loaded!  Shape: {pet_4d.shape}  "
                               f"(frames × slices × rows × cols)")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading PET: {e}")

    # ── Show PET metadata ─────────────────────────────────────
    if utils.has("pet_meta"):
        with st.expander("PET metadata", expanded=False):
            meta = utils.get("pet_meta")
            cols = st.columns(4)
            cols[0].metric("Time frames",    meta["n_time"])
            cols[1].metric("Slices",         meta["n_slices"])
            cols[2].metric("Pixel spacing",  f"{meta['pixel_spacing'][0]:.2f} mm")
            cols[3].metric("Slice spacing",  f"{meta['slice_thickness']:.2f} mm")

            if meta.get("frame_durations"):
                durs = np.array(meta["frame_durations"])
                st.markdown(f"**Frame durations**: min={durs.min():.0f} ms, "
                            f"max={durs.max():.0f} ms, mean={durs.mean():.0f} ms")
            if meta.get("frame_start_times"):
                times = np.array(meta["frame_start_times"])
                st.markdown(f"**Total scan duration**: {(times[-1]/1000):.1f} s  "
                            f"({(times[-1]/60000):.1f} min)")

    st.markdown("---")

    # ── Load MRI ─────────────────────────────────────────────
    st.markdown("### 2 — Load MRI")
    col_a, col_b = st.columns([2, 1])
    with col_a:
        if utils.has("mri_vol"):
            st.success(f"✓ MRI loaded — shape {utils.get('mri_vol').shape}")
        else:
            st.warning("MRI not yet loaded.")
    with col_b:
        load_mri_btn = st.button("Load MRI", use_container_width=True)

    if load_mri_btn:
        if not list(utils.MRI_DIR.glob("**/*.dcm")):
            st.error(f"No .dcm files found in `{utils.MRI_DIR}`.")
        else:
            with st.spinner("Parsing MRI DICOM…"):
                try:
                    mri_vol, spacing = load_mri_from_dir(utils.MRI_DIR)
                    mri_vol = utils.ensure_3d(mri_vol)
                    utils.put("mri_vol", mri_vol)
                    utils.save_cache("mri_vol", mri_vol)
                    st.success(f"✓ MRI loaded!  Shape: {mri_vol.shape}  "
                               f"Spacing: {tuple(round(s,2) for s in spacing)} mm")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading MRI: {e}")

    st.markdown("---")
    st.markdown("### 3 — Quick Preview")
    if utils.has("pet_4d") or utils.has("mri_vol"):
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        fig.patch.set_facecolor(utils.DARK_BG)
        titles, imgs, cmaps = [], [], []

        if utils.has("pet_4d"):
            pet_4d = utils.get("pet_4d")
            T, Z = pet_4d.shape[:2]
            titles += [f"PET frame 0  (z={Z//2})", f"PET last  (z={Z//2})"]
            imgs   += [pet_4d[0, Z//2], pet_4d[-1, Z//2]]
            cmaps  += ["hot", "hot"]
        if utils.has("mri_vol"):
            mri = utils.get("mri_vol")
            Z   = mri.shape[0]
            titles += [f"MRI axial  (z={Z//2})", f"MRI coronal (y={mri.shape[1]//2})"]
            imgs   += [mri[Z//2], mri[:, mri.shape[1]//2]]
            cmaps  += ["gray", "gray"]

        for i, (ax, img, cmap, ttl) in enumerate(zip(axes, imgs, cmaps, titles)):
            ax.set_facecolor(utils.PANEL)
            ax.imshow(utils.norm01(img), cmap=cmap, origin="lower", aspect="auto")
            ax.set_title(ttl, color="#c8d8e8", fontsize=8)
            ax.axis("off")
        for ax in axes[len(imgs):]:
            ax.set_visible(False)

        plt.tight_layout(pad=0.3)
        st.pyplot(fig, use_container_width=True)
        plt.close()
    else:
        st.info("Load PET and/or MRI data above to see a preview.")