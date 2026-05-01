"""
pages/load_dicom.py — Upload & parse PET + MRI DICOM files
Supports both file upload (cloud deployment) and local data/ directory (local run).
"""
import io
import os
import shutil
import tempfile
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pydicom
import streamlit as st

import utils

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# DICOM PARSING HELPERS  (shared by upload + local paths)
# ─────────────────────────────────────────────────────────────

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
        unique_z = np.unique(np.round(fp[:, 2], decimals=2))
        n_slices = len(unique_z)
        n_time   = n_frames_total // n_slices
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
        "frame_durations":   frame_durations,
    }


def _parse_classic_pet(datasets):
    """datasets: list of pydicom Dataset objects (already read)."""
    def sort_key(ds):
        t = int(getattr(ds, "TemporalPositionIdentifier", 1))
        z = float(ds.ImagePositionPatient[2]) if hasattr(ds, "ImagePositionPatient") else 0.0
        return (t, z)

    datasets.sort(key=sort_key)
    rows = int(datasets[0][0x0028, 0x0010].value)
    cols = int(datasets[0][0x0028, 0x0011].value)

    t_ids = sorted(set(int(getattr(s, "TemporalPositionIdentifier", 1)) for s in datasets))
    z_pos = sorted(set(round(float(s.ImagePositionPatient[2]), 2)
                        for s in datasets if hasattr(s, "ImagePositionPatient")))
    n_time   = len(t_ids)
    n_slices = len(z_pos) if z_pos else max(1, len(datasets) // max(n_time, 1))

    pet_4d = np.zeros((n_time, n_slices, rows, cols), dtype=np.float32)
    for ds in datasets:
        t  = t_ids.index(int(getattr(ds, "TemporalPositionIdentifier", 1)))
        if z_pos:
            zv = round(float(ds.ImagePositionPatient[2]), 2)
            z  = z_pos.index(zv)
        else:
            z  = 0
        sl = ds.pixel_array.astype(np.float32)
        pet_4d[t, z] = sl * float(getattr(ds, "RescaleSlope", 1)) \
                          + float(getattr(ds, "RescaleIntercept", 0))

    pixel_spacing = [float(v) for v in datasets[0][0x0028, 0x0030].value]
    try:
        slice_thickness = float(datasets[0][0x0018, 0x0088].value)
    except Exception:
        slice_thickness = 1.0

    return {
        "pet_4d": pet_4d, "n_time": n_time, "n_slices": n_slices,
        "rows": rows, "cols": cols,
        "pixel_spacing": pixel_spacing, "slice_thickness": slice_thickness,
        "frame_start_times": None, "frame_durations": None,
    }


def parse_pet_datasets(datasets):
    """Entry point: given a list of pydicom Datasets, parse into 4D array."""
    if len(datasets) == 1 and hasattr(datasets[0], "NumberOfFrames"):
        return _parse_enhanced_pet(datasets[0])
    return _parse_classic_pet(datasets)


def parse_mri_datasets(datasets, tmp_dir):
    """
    Write datasets to a temp dir and load with SimpleITK (preserves spacing/orientation).
    Falls back to pure-pydicom stack if SimpleITK fails.
    """
    import SimpleITK as sitk

    # Write to temp dir so SimpleITK series reader can sort them
    for i, ds in enumerate(datasets):
        ds.save_as(str(tmp_dir / f"slice_{i:05d}.dcm"))

    try:
        reader    = sitk.ImageSeriesReader()
        dcm_names = reader.GetGDCMSeriesFileNames(str(tmp_dir))
        if not dcm_names:
            dcm_names = sorted([str(f) for f in tmp_dir.glob("*.dcm")])
        reader.SetFileNames(dcm_names)
        img     = reader.Execute()
        arr     = img.GetArrayFromImage(img).astype(np.float32) \
                  if False else sitk.GetArrayFromImage(img).astype(np.float32)
        spacing = img.GetSpacing()
        return arr, spacing
    except Exception:
        # Pure pydicom fallback — sort by ImagePositionPatient Z
        def z_key(ds):
            return float(ds.ImagePositionPatient[2]) \
                   if hasattr(ds, "ImagePositionPatient") else 0.0
        datasets.sort(key=z_key)
        slices = [ds.pixel_array.astype(np.float32) for ds in datasets]
        arr = np.stack(slices, axis=0)
        try:
            ps = [float(v) for v in datasets[0][0x0028, 0x0030].value]
            st_ = float(datasets[0][0x0018, 0x0088].value)
            spacing = (ps[1], ps[0], st_)
        except Exception:
            spacing = (1.0, 1.0, 1.0)
        return arr, spacing


# ─────────────────────────────────────────────────────────────
# READ UPLOADED FILES → pydicom datasets
# ─────────────────────────────────────────────────────────────

def uploaded_files_to_datasets(uploaded_files):
    """
    Convert a list of Streamlit UploadedFile objects to pydicom Datasets.
    Handles:
      - Plain .dcm files
      - .zip archives containing .dcm files
    """
    datasets = []
    for uf in uploaded_files:
        name = uf.name.lower()
        data = uf.read()

        if name.endswith(".zip"):
            import zipfile
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                for zname in zf.namelist():
                    if zname.lower().endswith(".dcm") \
                       and not zname.startswith("__MACOSX"):
                        with zf.open(zname) as zfile:
                            try:
                                ds = pydicom.dcmread(io.BytesIO(zfile.read()),
                                                     force=True)
                                datasets.append(ds)
                            except Exception:
                                pass
        elif name.endswith(".dcm") or name.endswith(""):
            try:
                ds = pydicom.dcmread(io.BytesIO(data), force=True)
                datasets.append(ds)
            except Exception:
                pass

    return datasets


# ─────────────────────────────────────────────────────────────
# PAGE
# ─────────────────────────────────────────────────────────────

def render():
    utils.try_restore_from_cache()
    utils.sidebar_status()

    st.title("📂 Load DICOM Data")
    st.markdown(
        "<p style='color:#5a7080;'>"
        "Upload your DICOM files directly — no local setup required. "
        "Works locally and when deployed to Streamlit Cloud / any server."
        "</p>",
        unsafe_allow_html=True,
    )

    # ── How-to banner ─────────────────────────────────────────
    with st.expander("ℹ️  How to upload — click to expand", expanded=not utils.has("pet_4d")):
        st.markdown("""
        **Accepted formats:**
        - 📦 **ZIP file** containing all `.dcm` files — *recommended, easiest*
        - 📄 **Individual `.dcm` files** — select multiple files at once

        **Steps:**
        1. Download your studies from the course link
        2. Zip the PET folder → upload below as *PET upload*
        3. Zip the MRI folder → upload below as *MRI upload*
        4. Click **Process PET** / **Process MRI**

        > 💡 Files are processed in-memory — nothing is stored permanently on the server.
        > Results are cached in your browser session and reset when you close the tab.
        """)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════
    # PET UPLOAD
    # ══════════════════════════════════════════════════════════
    st.markdown("### 1 — Dynamic PET  `(1 BRAIN DINAMIC COLINA)`")

    if utils.has("pet_4d"):
        meta = utils.get("pet_meta") or {}
        st.success(
            f"✓ PET loaded — shape {utils.get('pet_4d').shape}  "
            f"({meta.get('n_time','?')} frames × {meta.get('n_slices','?')} slices)"
        )
        if st.button("🗑️  Clear PET & re-upload", key="clear_pet"):
            utils.put("pet_4d",  None)
            utils.put("pet_avg", None)
            utils.put("pet_last", None)
            utils.put("pet_meta", None)
            st.rerun()
    else:
        pet_files = st.file_uploader(
            "Upload PET DICOM files",
            type=["dcm", "zip"],
            accept_multiple_files=True,
            key="pet_uploader",
            help="Select a ZIP of your PET .dcm files, or pick all .dcm files individually.",
        )

        n_pet = len(pet_files) if pet_files else 0
        if n_pet > 0:
            st.markdown(
                f"<span style='color:#00e5a0; font-size:0.8rem;'>"
                f"✓ {n_pet} file(s) selected</span>",
                unsafe_allow_html=True,
            )

        if st.button("⚙️  Process PET", disabled=n_pet == 0, key="proc_pet", type="primary"):
            with st.spinner(f"Reading {n_pet} file(s) and parsing DICOM…"):
                try:
                    datasets = uploaded_files_to_datasets(pet_files)
                    if not datasets:
                        st.error("No valid DICOM files found in the upload.")
                        return

                    st.info(f"Found {len(datasets)} DICOM slice(s) — parsing 4D array…")
                    data     = parse_pet_datasets(datasets)
                    pet_4d   = utils.ensure_4d(data["pet_4d"])
                    pet_avg  = utils.ensure_3d(pet_4d.mean(axis=0))
                    pet_last = utils.ensure_3d(pet_4d[-1])
                    meta = {k: data[k] for k in
                            ["pixel_spacing", "slice_thickness", "n_time", "n_slices",
                             "rows", "cols", "frame_start_times", "frame_durations"]}

                    utils.put("pet_4d",   pet_4d)
                    utils.put("pet_avg",  pet_avg)
                    utils.put("pet_last", pet_last)
                    utils.put("pet_meta", meta)
                    utils.save_cache("pet_4d",   pet_4d)
                    utils.save_cache("pet_avg",  pet_avg)
                    utils.save_cache("pet_last", pet_last)
                    np.save(str(utils.CACHE_DIR / "pet_meta.npy"), meta, allow_pickle=True)

                    st.success(f"✓ PET processed! Shape: {pet_4d.shape}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing PET: {e}")
                    import traceback; st.code(traceback.format_exc())

    # PET metadata expander
    if utils.has("pet_meta"):
        with st.expander("PET metadata", expanded=False):
            meta = utils.get("pet_meta")
            cols = st.columns(4)
            cols[0].metric("Time frames",   meta.get("n_time", "?"))
            cols[1].metric("Slices",        meta.get("n_slices", "?"))
            cols[2].metric("Pixel spacing", f"{meta['pixel_spacing'][0]:.2f} mm"
                           if meta.get("pixel_spacing") else "?")
            cols[3].metric("Slice spacing", f"{meta.get('slice_thickness', 1.0):.2f} mm")
            if meta.get("frame_durations"):
                durs = np.array(meta["frame_durations"])
                st.markdown(f"**Frame durations**: min={durs.min():.0f} ms  "
                            f"max={durs.max():.0f} ms  mean={durs.mean():.0f} ms")
            if meta.get("frame_start_times"):
                times = np.array(meta["frame_start_times"])
                st.markdown(f"**Total scan duration**: {times[-1]/1000:.1f} s "
                            f"({times[-1]/60000:.1f} min)")

    st.markdown("---")

    # ══════════════════════════════════════════════════════════
    # MRI UPLOAD
    # ══════════════════════════════════════════════════════════
    st.markdown("### 2 — MRI  `(AX 3D T1)`")

    if utils.has("mri_vol"):
        st.success(f"✓ MRI loaded — shape {utils.get('mri_vol').shape}")
        if st.button("🗑️  Clear MRI & re-upload", key="clear_mri"):
            utils.put("mri_vol", None)
            st.rerun()
    else:
        mri_files = st.file_uploader(
            "Upload MRI DICOM files",
            type=["dcm", "zip"],
            accept_multiple_files=True,
            key="mri_uploader",
            help="Select a ZIP of your MRI .dcm files, or pick all .dcm files individually.",
        )

        n_mri = len(mri_files) if mri_files else 0
        if n_mri > 0:
            st.markdown(
                f"<span style='color:#00e5a0; font-size:0.8rem;'>"
                f"✓ {n_mri} file(s) selected</span>",
                unsafe_allow_html=True,
            )

        if st.button("⚙️  Process MRI", disabled=n_mri == 0, key="proc_mri", type="primary"):
            with st.spinner(f"Reading {n_mri} file(s) and parsing DICOM…"):
                try:
                    datasets = uploaded_files_to_datasets(mri_files)
                    if not datasets:
                        st.error("No valid DICOM files found in the upload.")
                        return

                    st.info(f"Found {len(datasets)} DICOM slice(s) — building 3D volume…")
                    with tempfile.TemporaryDirectory() as tmp:
                        tmp_dir = Path(tmp)
                        mri_vol, spacing = parse_mri_datasets(datasets, tmp_dir)

                    mri_vol = utils.ensure_3d(mri_vol)
                    utils.put("mri_vol", mri_vol)
                    utils.save_cache("mri_vol", mri_vol)
                    st.success(
                        f"✓ MRI processed! Shape: {mri_vol.shape}  "
                        f"Spacing: {tuple(round(s,2) for s in spacing)} mm"
                    )
                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing MRI: {e}")
                    import traceback; st.code(traceback.format_exc())

    st.markdown("---")

    # ══════════════════════════════════════════════════════════
    # QUICK PREVIEW
    # ══════════════════════════════════════════════════════════
    st.markdown("### 3 — Quick Preview")

    if utils.has("pet_4d") or utils.has("mri_vol"):
        panels, titles, cmaps = [], [], []

        if utils.has("pet_4d"):
            pet_4d = utils.get("pet_4d")
            T, Z   = pet_4d.shape[:2]
            panels += [pet_4d[0,  Z//2], pet_4d[-1, Z//2]]
            titles += [f"PET frame 1  z={Z//2}", f"PET last frame  z={Z//2}"]
            cmaps  += ["hot", "hot"]

        if utils.has("mri_vol"):
            mri  = utils.get("mri_vol")
            Z    = mri.shape[0]
            panels += [mri[Z//2], mri[:, mri.shape[1]//2, :]]
            titles += [f"MRI axial  z={Z//2}", f"MRI coronal  y={mri.shape[1]//2}"]
            cmaps  += ["gray", "gray"]

        n = len(panels)
        fig, axes = plt.subplots(1, n, figsize=(5*n, 4))
        fig.patch.set_facecolor(utils.DARK_BG)
        if n == 1:
            axes = [axes]
        for ax, img, cmap, ttl in zip(axes, panels, cmaps, titles):
            ax.set_facecolor(utils.PANEL)
            ax.imshow(utils.norm01(img), cmap=cmap, origin="lower", aspect="auto")
            ax.set_title(ttl, color="#c8d8e8", fontsize=8)
            ax.axis("off")

        plt.tight_layout(pad=0.3)
        st.pyplot(fig, use_container_width=True)
        plt.close()
    else:
        st.info("Upload and process PET and/or MRI data above to see a preview here.")

    # ── Session reset ─────────────────────────────────────────
    st.markdown("---")
    with st.expander("⚠️  Reset entire session"):
        st.warning("This clears all loaded data, coregistration results, and segmentation masks.")
        if st.button("🔴  Reset everything", key="reset_all"):
            for k in list(utils.KEYS.values()):
                if k in st.session_state:
                    del st.session_state[k]
            # Also wipe disk cache
            import shutil
            if utils.CACHE_DIR.exists():
                shutil.rmtree(str(utils.CACHE_DIR))
                utils.CACHE_DIR.mkdir(parents=True, exist_ok=True)
            st.success("Session cleared.")
            st.rerun()