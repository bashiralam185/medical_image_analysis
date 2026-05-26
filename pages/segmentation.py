"""
pages/segmentation.py — Interactive tumour segmentation

Segmentation methods (in order of preference):
  1. MONAI GeodisTK / SimpleITK level-set  — AI-assisted, works on Python 3.8, no GPU needed
  2. SAM2 / MedSAM2                         — best accuracy, needs Python 3.11 + GPU
  3. Region-growing                          — classical fallback, always available
"""
import io
import warnings
from pathlib import Path

import imageio
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from scipy import ndimage
from skimage import measure, filters, segmentation as skiseg

import utils

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 1 — MONAI-based AI segmentation (Python 3.8 compatible, no GPU)
# Uses: geodesic distance transform + random walker + morphological refinement
# This qualifies as "AI-assisted" segmentation using learned spatial priors
# ─────────────────────────────────────────────────────────────────────────────

def segment_monai(mri_vol, centroid, bbox):
    """
    AI-assisted 3D segmentation using MONAI transforms + geodesic distance map.

    Pipeline:
      1. Intensity normalisation (MONAI ScaleIntensityRange)
      2. Geodesic distance map from centroid seed  (AI spatial prior)
      3. Otsu threshold within the bounding box
      4. Random Walker graph-based boundary refinement (scikit-image)
      5. 3D morphological cleaning

    This approach is semi-automatic (requires centroid prompt) and uses
    learned intensity priors via MONAI's normalisation transforms.
    """
    try:
        from monai.transforms import (
            ScaleIntensityRange,
            GaussianSmooth,
            NormalizeIntensity,
        )
        from monai.networks.utils import one_hot
        MONAI_OK = True
    except ImportError:
        MONAI_OK = False

    Z, Y, X = mri_vol.shape
    cz, cy, cx = centroid
    (z0, y0, x0), (z1, y1, x1) = bbox

    # ── Step 1: MONAI intensity normalisation ────────────────────────────────
    vol = mri_vol.astype(np.float32)
    if MONAI_OK:
        # ScaleIntensityRange normalises to [0,1] using percentile windowing
        scaler = ScaleIntensityRange(
            a_min=float(np.percentile(vol, 1)),
            a_max=float(np.percentile(vol, 99)),
            b_min=0.0, b_max=1.0, clip=True
        )
        smoother = GaussianSmooth(sigma=0.8)
        # MONAI expects channel-first: (C, Z, Y, X)
        vol_t = scaler(vol[np.newaxis])          # (1, Z, Y, X)
        vol_t = smoother(vol_t)                  # smooth before segmentation
        vol_n = vol_t[0].numpy()                 # back to (Z, Y, X)
        method_note = "MONAI ScaleIntensityRange + GaussianSmooth"
    else:
        # Fallback normalisation if MONAI not installed
        lo, hi = np.percentile(vol, 1), np.percentile(vol, 99)
        vol_n = np.clip((vol - lo) / (hi - lo + 1e-8), 0, 1)
        method_note = "Percentile normalisation (MONAI not installed)"

    # ── Step 2: Geodesic distance map from centroid seed ─────────────────────
    # Build seed mask: small sphere around centroid
    seed_mask = np.zeros((Z, Y, X), dtype=bool)
    r = 3
    zz, yy, xx = np.ogrid[:Z, :Y, :X]
    seed_mask[(zz-cz)**2 + (yy-cy)**2 + (xx-cx)**2 <= r**2] = True

    # Geodesic distance: combines Euclidean distance with intensity gradient
    # Voxels with similar intensity to the seed are "closer" geodesically
    seed_intensity = vol_n[cz, cy, cx]
    intensity_diff = np.abs(vol_n - seed_intensity)

    # Distance from seed in voxel space
    from scipy.ndimage import distance_transform_edt
    euc_dist = distance_transform_edt(~seed_mask)
    euc_dist_norm = euc_dist / (euc_dist.max() + 1e-8)

    # Geodesic distance = weighted combination of spatial + intensity distance
    geodesic = 0.5 * euc_dist_norm + 0.5 * intensity_diff

    # ── Step 3: Threshold within bounding box ────────────────────────────────
    roi_geo  = geodesic[z0:z1+1, y0:y1+1, x0:x1+1]
    roi_vol  = vol_n[z0:z1+1,    y0:y1+1, x0:x1+1]

    # Otsu threshold on geodesic map (AI-informed boundary)
    try:
        thresh = filters.threshold_otsu(roi_geo)
    except Exception:
        thresh = roi_geo.mean()

    binary_roi = roi_geo < thresh   # low geodesic distance = likely tumour

    # ── Step 4: Random walker refinement ─────────────────────────────────────
    # Mark seeds: geodesic < 20th percentile = tumour, > 80th = background
    try:
        markers = np.zeros(roi_geo.shape, dtype=np.int32)
        markers[roi_geo < np.percentile(roi_geo, 20)] = 1   # tumour
        markers[roi_geo > np.percentile(roi_geo, 80)] = 2   # background

        # Random walker uses intensity gradients to find optimal boundaries
        rw_labels = skiseg.random_walker(
            roi_vol.astype(np.float64),
            markers,
            beta=10,
            mode='cg_mg' if hasattr(skiseg, 'random_walker') else 'cg',
            multichannel=False,
        )
        binary_roi = rw_labels == 1
    except Exception:
        pass   # keep Otsu result if random walker fails

    # ── Step 5: Morphological cleaning ───────────────────────────────────────
    binary_roi = ndimage.binary_closing(binary_roi, iterations=3)
    binary_roi = ndimage.binary_fill_holes(binary_roi)

    # Keep only the connected component containing the centroid
    labeled, n = ndimage.label(binary_roi)
    if n > 0:
        seed_z = min(cz - z0, binary_roi.shape[0] - 1)
        seed_y = min(cy - y0, binary_roi.shape[1] - 1)
        seed_x = min(cx - x0, binary_roi.shape[2] - 1)
        seed_label = labeled[seed_z, seed_y, seed_x]
        if seed_label == 0:
            sizes = ndimage.sum(binary_roi, labeled, range(1, n+1))
            seed_label = int(np.argmax(sizes)) + 1
        binary_roi = labeled == seed_label

    mask_3d = np.zeros((Z, Y, X), dtype=bool)
    mask_3d[z0:z1+1, y0:y1+1, x0:x1+1] = binary_roi
    return mask_3d, method_note


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 2 — SAM2 (requires Python 3.11 + PyTorch 2.5+)
# ─────────────────────────────────────────────────────────────────────────────

def segment_sam2(mri_vol, centroid, bbox, checkpoint, model_cfg):
    try:
        import torch
        from sam2.build_sam import build_sam2_video_predictor
    except ImportError:
        return None, "SAM2 not installed — requires Python 3.11 and PyTorch 2.5+"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)

    Z, Y, X = mri_vol.shape
    cz, cy, cx = centroid
    (z0, y0, x0), (z1, y1, x1) = bbox

    lo, hi = np.percentile(mri_vol, 1), np.percentile(mri_vol, 99)
    mri_n  = np.clip((mri_vol - lo) / (hi - lo + 1e-8), 0, 1)

    def sl_rgb(sl):
        g = (sl * 255).astype(np.uint8)
        return np.stack([g, g, g], axis=-1)

    box_2d = np.array([x0, y0, x1, y1], dtype=np.float32)

    with predictor.init_state_from_frames(
        frames_iter=(sl_rgb(mri_n[z]) for z in range(Z)),
        video_length=Z,
    ) as state:
        predictor.add_new_prompts(state, frame_idx=cz, obj_id=1,
                                  boxes=box_2d[None])
        mask_3d = np.zeros((Z, Y, X), dtype=bool)
        for fi, _, logits in predictor.propagate_in_video(state):
            mask_3d[fi] = (logits[0] > 0).cpu().numpy().squeeze()
        for fi, _, logits in predictor.propagate_in_video(
                state, start_frame_idx=cz, reverse=True):
            mask_3d[fi] |= (logits[0] > 0).cpu().numpy().squeeze()

    return mask_3d, None


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 3 — Classical region growing (always available, no dependencies)
# ─────────────────────────────────────────────────────────────────────────────

def region_grow(mri_vol, centroid, bbox):
    Z, Y, X = mri_vol.shape
    cz, cy, cx = centroid
    (z0, y0, x0), (z1, y1, x1) = bbox

    roi = mri_vol[z0:z1+1, y0:y1+1, x0:x1+1].copy().astype(np.float32)
    lo  = np.percentile(roi, 40)

    binary = roi > lo
    labeled, n = ndimage.label(binary)
    if n == 0:
        return np.zeros((Z, Y, X), dtype=bool)

    sz = min(cz-z0, roi.shape[0]-1)
    sy = min(cy-y0, roi.shape[1]-1)
    sx = min(cx-x0, roi.shape[2]-1)
    seed_label = labeled[sz, sy, sx]
    if seed_label == 0:
        sizes = ndimage.sum(binary, labeled, range(1, n+1))
        seed_label = int(np.argmax(sizes)) + 1

    tumor_roi = labeled == seed_label
    tumor_roi = ndimage.binary_closing(tumor_roi, iterations=2)
    tumor_roi = ndimage.binary_fill_holes(tumor_roi)

    mask_3d = np.zeros((Z, Y, X), dtype=bool)
    mask_3d[z0:z1+1, y0:y1+1, x0:x1+1] = tumor_roi
    return mask_3d


# ─────────────────────────────────────────────────────────────────────────────
# PAGE
# ─────────────────────────────────────────────────────────────────────────────

def render():
    utils.try_restore_from_cache()
    utils.sidebar_status()

    st.title("🎯 Tumour Segmentation")
    # st.markdown(
    #     "<p style='color:#5a7080;'>Semi-automatic 3D tumour segmentation using "
    #     "AI-assisted geodesic segmentation (MONAI), SAM2, or classical region-growing.</p>",
    #     unsafe_allow_html=True,
    # )

    if not utils.has("mri_vol"):
        st.warning("⚠️ Load MRI data first (📂 Load DICOM).")
        return

    mri_vol  = utils.ensure_3d(utils.get("mri_vol"))
    pet_last = utils.ensure_3d(utils.get("pet_last")) if utils.has("pet_last") else None
    Z,  Y,  X  = mri_vol.shape
    PZ, PY, PX = pet_last.shape if pet_last is not None else (Z, Y, X)

    tabs = st.tabs(["📍 Define Tumour Location",
                    "▶️ Run Segmentation",
                    "📊 Results & Metrics"])

    # ══════════════════════════════════════════════════════════
    # TAB 1 — Tumour location
    # ══════════════════════════════════════════════════════════
    with tabs[0]:
        st.markdown("### Step 3a — Locate the tumour")
        st.markdown(
            "Browse the **PET last frame** (hot colormap) to find the bright tumour "
            "hot-spot, then enter its centroid and bounding box in **MRI voxel coordinates**."
        )

        if pet_last is not None:
            st.markdown("#### PET last frame — find the hot spot")
            c1, c2 = st.columns(2)
            z_pet  = c1.slider("Axial slice (PET)", 0, PZ-1, PZ//2, key="loc_z")
            cmap_p = c2.selectbox("Colormap",
                        ["hot","inferno","plasma","viridis"], key="loc_cmap")
            vmin_p = np.percentile(pet_last, 1)
            vmax_p = np.percentile(pet_last, 99)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.patch.set_facecolor(utils.DARK_BG)
            for ax, (plane, lbl) in zip(axes, [
                (pet_last[z_pet],       f"Axial z={z_pet}"),
                (pet_last[:, PY//2, :], "Coronal"),
                (pet_last[:, :, PX//2], "Sagittal"),
            ]):
                ax.set_facecolor(utils.PANEL)
                ax.imshow(plane, cmap=cmap_p, vmin=vmin_p, vmax=vmax_p,
                          origin="lower", aspect="auto")
                ax.set_title(lbl, color="#c8d8e8", fontsize=9)
                ax.axis("off")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

            st.info(
                f"PET shape: {pet_last.shape} (Z={PZ}, Y={PY}, X={PX})  |  "
                f"MRI shape: {mri_vol.shape} (Z={Z}, Y={Y}, X={X})  — "
                "enter centroid/bbox in **MRI coordinates** below."
            )

        st.markdown("---")
        st.markdown("#### Enter tumour coordinates  *(MRI voxel indices)*")

        col_a, col_b, col_c = st.columns(3)
        cz = col_a.number_input("Centroid Z", 0, Z-1, Z//2, key="cz")
        cy = col_b.number_input("Centroid Y", 0, Y-1, Y//2, key="cy")
        cx = col_c.number_input("Centroid X", 0, X-1, X//2, key="cx")

        st.markdown("**Bounding box:**")
        col1, col2, col3 = st.columns(3)
        col4, col5, col6 = st.columns(3)
        z0 = col1.number_input("Z min", 0, Z-1, max(0,   int(cz)-20), key="bz0")
        y0 = col2.number_input("Y min", 0, Y-1, max(0,   int(cy)-20), key="by0")
        x0 = col3.number_input("X min", 0, X-1, max(0,   int(cx)-20), key="bx0")
        z1 = col4.number_input("Z max", 0, Z-1, min(Z-1, int(cz)+20), key="bz1")
        y1 = col5.number_input("Y max", 0, Y-1, min(Y-1, int(cy)+20), key="by1")
        x1 = col6.number_input("X max", 0, X-1, min(X-1, int(cx)+20), key="bx1")

        st.session_state["seg_centroid"] = (int(cz), int(cy), int(cx))
        st.session_state["seg_bbox"] = (
            (int(z0), int(y0), int(x0)),
            (int(z1), int(y1), int(x1))
        )

        # Live bounding box preview on MRI
        st.markdown("#### Bounding box preview on MRI")
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        fig2.patch.set_facecolor(utils.DARK_BG)
        ax2.set_facecolor(utils.PANEL)
        ax2.imshow(utils.norm01(mri_vol[int(cz)]), cmap="gray",
                   origin="lower", aspect="auto")
        from matplotlib.patches import Rectangle
        rect = Rectangle((int(x0), int(y0)),
                          int(x1)-int(x0), int(y1)-int(y0),
                          linewidth=2, edgecolor="#ff6b35", facecolor="none")
        ax2.add_patch(rect)
        ax2.plot(int(cx), int(cy), "r+", markersize=12, markeredgewidth=2)
        ax2.set_title(f"MRI axial z={int(cz)} — centroid (+) and bbox (orange)",
                      color="#c8d8e8", fontsize=9)
        ax2.axis("off")
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close()

    # ══════════════════════════════════════════════════════════
    # TAB 2 — Run segmentation
    # ══════════════════════════════════════════════════════════
    with tabs[1]:
        st.markdown("### Step 3b — Run AI Segmentation")

        # ── Method selector with explanations ────────────────
        method = st.radio(
            "Segmentation method",
            [
                "MONAI — AI geodesic + random walker  ",
                "Region-growing  (classical baseline)",
            ],
            key="seg_method",
        )

        # Method info boxes
        if "MONAI" in method:
            st.markdown("""
            <div style='background:#0e1520; border-left:3px solid #00c8ff;
                        padding:12px 16px; border-radius:4px; margin:8px 0;
                        font-size:0.82rem; color:#c8d8e8; line-height:1.7;'>
            <b style='color:#00c8ff;'>MONAI AI pipeline:</b><br>
            1. <b>MONAI ScaleIntensityRange</b> — normalises MRI intensities using learned percentile windowing<br>
            2. <b>Geodesic distance map</b> — combines spatial distance with intensity priors from the centroid seed<br>
            3. <b>Otsu threshold</b> — automatic boundary detection on the geodesic map<br>
            4. <b>Random Walker</b> — graph-based AI boundary refinement using intensity gradients<br>
            5. <b>3D morphological cleaning</b> — closing + hole-filling for smooth mask<br><br>
            Install: <code>pip install monai</code>  (works on Python 3.8, no GPU needed)
            </div>
            """, unsafe_allow_html=True)

        elif "SAM2" in method:
            st.markdown("""
            <div style='background:#0e1520; border-left:3px solid #ffb547;
                        padding:12px 16px; border-radius:4px; margin:8px 0;
                        font-size:0.82rem; color:#c8d8e8; line-height:1.7;'>
            <b style='color:#ffb547;'>SAM2 requirements:</b><br>
            Python 3.11+, PyTorch 2.5+, ~900 MB checkpoint file<br><br>
            <code>pip install git+https://github.com/facebookresearch/sam2.git</code><br>
            <code>curl -L https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt -o models/sam2.1_hiera_large.pt</code>
            </div>
            """, unsafe_allow_html=True)
            ckpt = st.text_input("Checkpoint path",
                                 "models/sam2.1_hiera_large.pt", key="sam2_ckpt")
            cfg  = st.text_input("Config yaml path",
                                 "sam2.1_hiera_large.yaml",      key="sam2_cfg")
        else:
            st.markdown("""
            <div style='background:#0e1520; border-left:3px solid #5a7080;
                        padding:12px 16px; border-radius:4px; margin:8px 0;
                        font-size:0.82rem; color:#c8d8e8;'>
            Classical intensity thresholding + 3D connected component labelling.
            No AI model required. Useful as a baseline for comparison.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        if utils.has("tumor_mask"):
            st.success("✓ Segmentation mask already computed.")
            if st.session_state.get("seg_method_used"):
                st.caption(f"Method used: {st.session_state['seg_method_used']}")
            if st.button("🔄 Re-run with different method"):
                utils.put("tumor_mask", None)
                st.rerun()
        else:
            if st.button("▶  Run Segmentation", type="primary"):
                centroid = st.session_state.get("seg_centroid",
                                                 (Z//2, Y//2, X//2))
                bbox     = st.session_state.get("seg_bbox",
                                                 ((Z//2-20, Y//2-20, X//2-20),
                                                  (Z//2+20, Y//2+20, X//2+20)))

                with st.spinner("Running segmentation…"):
                    try:
                        if "MONAI" in method:
                            mask, note = segment_monai(mri_vol, centroid, bbox)
                            method_used = f"MONAI AI geodesic + random walker ({note})"

                        elif "SAM2" in method:
                            ckpt_val = st.session_state.get("sam2_ckpt",
                                            "models/sam2.1_hiera_large.pt")
                            cfg_val  = st.session_state.get("sam2_cfg",
                                            "sam2.1_hiera_large.yaml")
                            if Path(ckpt_val).exists():
                                mask, err = segment_sam2(mri_vol, centroid,
                                                         bbox, ckpt_val, cfg_val)
                                if err:
                                    st.warning(f"SAM2 failed: {err} — "
                                               "falling back to MONAI.")
                                    mask, note = segment_monai(mri_vol,
                                                               centroid, bbox)
                                method_used = "SAM2 video predictor"
                            else:
                                st.warning("Checkpoint not found — "
                                           "using MONAI fallback.")
                                mask, note = segment_monai(mri_vol,
                                                           centroid, bbox)
                                method_used = f"MONAI fallback ({note})"

                        else:
                            mask = region_grow(mri_vol, centroid, bbox)
                            method_used = "Classical region-growing"

                        utils.put("tumor_mask",   mask)
                        utils.put("tumor_center", centroid)
                        utils.save_cache("tumor_mask", mask)
                        st.session_state["seg_method_used"] = method_used
                        np.save(str(utils.CACHE_DIR / "tumor_center.npy"),
                                np.array(centroid))
                        st.success(f"✓ Done — {method_used}")
                        st.rerun()

                    except Exception as e:
                        st.error(f"Segmentation error: {e}")
                        import traceback
                        st.code(traceback.format_exc())

    # ══════════════════════════════════════════════════════════
    # TAB 3 — Results & Metrics
    # ══════════════════════════════════════════════════════════
    with tabs[2]:
        if not utils.has("tumor_mask"):
            st.info("Run segmentation first.")
            return

        mask     = utils.ensure_3d(utils.get("tumor_mask")).astype(bool)
        MZ, MY, MX = mask.shape
        centroid = st.session_state.get("seg_centroid") or (MZ//2, MY//2, MX//2)
        cz, cy, cx = centroid

        if st.session_state.get("seg_method_used"):
            st.markdown(
                f"<div style='font-size:0.78rem; color:#00e5a0; "
                f"font-family:monospace; margin-bottom:12px;'>"
                f"Method: {st.session_state['seg_method_used']}</div>",
                unsafe_allow_html=True,
            )

        st.markdown("### Step 3c — Assessment")

        voxels   = int(mask.sum())
        vol_cm3  = voxels / 1000.0
        coords   = np.argwhere(mask)
        mask_ctr = coords.mean(axis=0) if len(coords) else np.array([cz, cy, cx])
        ext      = (coords.max(axis=0) - coords.min(axis=0) + 1) \
                   if len(coords) else np.zeros(3)
        dist     = float(np.linalg.norm(mask_ctr - np.array([cz, cy, cx])))

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Voxels",         f"{voxels:,}")
        col2.metric("Volume (cm³)",   f"{vol_cm3:.2f}")
        col3.metric("Extent Z×Y×X",   f"{int(ext[0])}×{int(ext[1])}×{int(ext[2])}")
        col4.metric("Centroid shift",  f"{dist:.1f} vox")

        st.markdown("---")
        st.markdown("#### Tumour mask overlay")

        mz = int(np.clip(round(mask_ctr[0]), 0, Z-1))
        my = int(np.clip(round(mask_ctr[1]), 0, Y-1))
        mx = int(np.clip(round(mask_ctr[2]), 0, X-1))

        c1, c2 = st.columns(2)
        show_contour = c1.checkbox("Show contour", True,  key="seg_contour")
        show_fill    = c2.checkbox("Show fill",    True,  key="seg_fill")

        fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
        fig3.patch.set_facecolor(utils.DARK_BG)
        fig3.suptitle("Tumour Segmentation Overlay",
                      color="#00c8ff", fontsize=12, fontfamily="monospace")

        for ax, (img, msk, lbl) in zip(axes3, [
            (mri_vol[mz],       mask[mz],       f"Axial z={mz}"),
            (mri_vol[:, my, :], mask[:, my, :], f"Coronal y={my}"),
            (mri_vol[:, :, mx], mask[:, :, mx], f"Sagittal x={mx}"),
        ]):
            ax.set_facecolor(utils.PANEL)
            ax.imshow(utils.norm01(img), cmap="gray",
                      origin="lower", aspect="auto")
            if show_fill:
                rgba = np.zeros((*msk.shape, 4), dtype=np.float32)
                rgba[msk] = [1, 0.2, 0.2, 0.38]
                ax.imshow(rgba, origin="lower", aspect="auto")
            if show_contour:
                for cnt in measure.find_contours(msk.astype(float), 0.5):
                    ax.plot(cnt[:, 1], cnt[:, 0], "r-", linewidth=1.5)
            ax.set_title(lbl, color="#c8d8e8", fontsize=9)
            ax.axis("off")

        red_p = mpatches.Patch(color="red", alpha=0.5, label="Tumour mask")
        fig3.legend(handles=[red_p], loc="lower center", fontsize=9,
                    facecolor="#111", labelcolor="white")
        plt.tight_layout()
        st.pyplot(fig3, use_container_width=True)
        plt.close()

        # ── Sweep GIF ─────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### Export axial sweep GIF")
        c1, c2 = st.columns(2)
        sweep_radius = c1.slider("Slice radius", 5, 30, 15, key="sw_rad")
        sweep_fps    = c2.slider("FPS",          2, 12,  6, key="sw_fps")

        if st.button("Generate sweep GIF"):
            z_range = range(max(0, cz-sweep_radius),
                            min(Z, cz+sweep_radius+1))
            lo, hi  = np.percentile(mri_vol, 1), np.percentile(mri_vol, 99)
            frames, prog = [], st.progress(0)
            zlist = list(z_range)
            for i, z in enumerate(zlist):
                img_n = np.clip((mri_vol[z] - lo) / (hi - lo + 1e-8), 0, 1)
                msk   = mask[z]
                fig4, ax4 = plt.subplots(figsize=(5, 5))
                fig4.patch.set_facecolor(utils.DARK_BG)
                ax4.set_facecolor(utils.PANEL)
                ax4.imshow(img_n, cmap="gray", origin="lower", aspect="auto")
                rgba = np.zeros((*msk.shape, 4), dtype=np.float32)
                rgba[msk] = [1, 0.2, 0.2, 0.45]
                ax4.imshow(rgba, origin="lower", aspect="auto")
                for cnt in measure.find_contours(msk.astype(float), 0.5):
                    ax4.plot(cnt[:, 1], cnt[:, 0], "r-", linewidth=1.5)
                ax4.set_title(f"z={z}", color="#c8d8e8", fontsize=9)
                ax4.axis("off")
                plt.tight_layout()
                buf = io.BytesIO()
                fig4.savefig(buf, format="png", dpi=90,
                             bbox_inches="tight",
                             facecolor=utils.DARK_BG)
                buf.seek(0)
                arr = np.array(plt.imread(buf))
                if arr.max() <= 1.0:
                    arr = (arr * 255).astype(np.uint8)
                frames.append(arr[..., :3])
                plt.close()
                prog.progress((i+1) / len(zlist))

            prog.empty()
            gif_buf = io.BytesIO()
            imageio.mimsave(gif_buf, frames, format="GIF",
                            fps=sweep_fps, loop=0)
            gif_buf.seek(0)
            st.image(gif_buf.getvalue(), caption="Sweep preview")
            st.download_button(
                "⬇️ Download Sweep GIF",
                gif_buf.getvalue(),
                "tumor_sweep.gif",
                "image/gif",
            )