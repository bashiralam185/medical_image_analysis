"""
Script 03 - 3D Tumour Segmentation (MONAI + Geodesic + Random Walker)
======================================================================
Objectives:
  3a) Manually find tumour centre and bounding box from PET last frame
  3b) MONAI-assisted segmentation:
        - MONAI ScaleIntensityRange + GaussianSmooth (normalisation)
        - Geodesic distance map from centroid seed
        - Otsu automatic threshold
        - Random Walker boundary refinement (scikit-image)
        - 3D morphological cleaning
  3c) Visualise and numerically assess the result

Dependencies:
    pip install monai scikit-image scipy numpy matplotlib imageio

Usage:
    # With centroid and bbox supplied directly:
    python 03_segmentation.py --centroid 45,128,140 --bbox 35,110,120,55,145,160

    # Interactive mode (shows PET montage, then prompts for coordinates):
    python 03_segmentation.py
"""

import argparse
import json
import warnings
from pathlib import Path

import imageio
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
from skimage import filters, measure
from skimage import segmentation as skiseg

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# 1. LOAD DATA FROM PREVIOUS SCRIPTS
# ─────────────────────────────────────────────────────────────

def load_outputs(out_dir: Path):
    """Load volumes produced by scripts 01 and 02."""
    pet_last  = np.load(out_dir / "pet_last.npy")
    mri_vol   = np.load(out_dir / "mri_vol.npy")
    pet_coreg = np.load(out_dir / "pet_coreg.npy")
    # Ensure 3D
    pet_last  = np.squeeze(pet_last)
    mri_vol   = np.squeeze(mri_vol)
    pet_coreg = np.squeeze(pet_coreg)
    print(f"  MRI shape      : {mri_vol.shape}")
    print(f"  PET last shape : {pet_last.shape}")
    return mri_vol, pet_coreg, pet_last


# ─────────────────────────────────────────────────────────────
# 2. INTERACTIVE TUMOUR LOCALISATION
# ─────────────────────────────────────────────────────────────

def pick_tumour_interactive(pet_last: np.ndarray):
    """
    Show a montage of all PET axial slices, then let the user enter
    centroid and bounding box coordinates manually.
    """
    Z, Y, X = pet_last.shape
    cols = 8
    rows = (Z + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 2.2))
    axes = axes.flatten()
    vmin, vmax = np.percentile(pet_last, 1), np.percentile(pet_last, 99)
    for i, ax in enumerate(axes):
        if i < Z:
            ax.imshow(pet_last[i], cmap="hot", vmin=vmin, vmax=vmax, origin="lower")
            ax.set_title(str(i), fontsize=7, color="white")
        ax.axis("off")
        ax.set_facecolor("#0a0a0a")
    fig.patch.set_facecolor("#0a0a0a")
    fig.suptitle(
        "PET last frame — all axial slices\n"
        "Find the brightest hot-spot (tumour). Note the slice number. Close to continue.",
        color="white", fontsize=10
    )
    plt.tight_layout()
    plt.show()

    print("\nEnter tumour location in MRI voxel coordinates.")
    print("(MRI and PET have different shapes — use MRI indices)\n")

    cz = int(input("  Centroid Z: ").strip())
    cy = int(input("  Centroid Y: ").strip())
    cx = int(input("  Centroid X: ").strip())
    centroid = (cz, cy, cx)

    print("\nEnter bounding box (tight box around tumour in MRI space):")
    z0 = int(input("  Z min: ").strip())
    y0 = int(input("  Y min: ").strip())
    x0 = int(input("  X min: ").strip())
    z1 = int(input("  Z max: ").strip())
    y1 = int(input("  Y max: ").strip())
    x1 = int(input("  X max: ").strip())
    bbox = ((z0, y0, x0), (z1, y1, x1))

    print(f"\n  Centroid : {centroid}")
    print(f"  Bbox     : {bbox}")
    return centroid, bbox


# ─────────────────────────────────────────────────────────────
# 3. MONAI AI SEGMENTATION PIPELINE
# ─────────────────────────────────────────────────────────────

def segment_monai(mri_vol: np.ndarray,
                  centroid: tuple,
                  bbox: tuple) -> tuple:
    """
    MONAI-assisted 3D tumour segmentation.

    Pipeline:
      1. MONAI ScaleIntensityRange — normalise MRI to [0,1]
      2. MONAI GaussianSmooth     — reduce noise (sigma=0.8mm)
      3. Geodesic distance map    — from centroid seed
      4. Otsu threshold           — automatic boundary on geodesic map
      5. Random Walker            — graph-based boundary refinement
      6. Morphological cleaning   — closing + hole-fill + component selection

    Returns:
        mask_3d     : np.ndarray bool (Z, Y, X)
        method_note : str describing what was used
    """

    # ── Try to import MONAI transforms ───────────────────────
    try:
        from monai.transforms import ScaleIntensityRange, GaussianSmooth
        MONAI_OK = True
        print("  MONAI found — using ScaleIntensityRange + GaussianSmooth")
    except ImportError:
        MONAI_OK = False
        print("  MONAI not installed — using percentile normalisation fallback")
        print("  Install with: pip install monai")

    Z, Y, X = mri_vol.shape
    cz, cy, cx = centroid
    (z0, y0, x0), (z1, y1, x1) = bbox

    # ── Step 1 + 2: Normalise with MONAI ─────────────────────
    vol = mri_vol.astype(np.float32)

    if MONAI_OK:
        scaler = ScaleIntensityRange(
            a_min=float(np.percentile(vol, 1)),
            a_max=float(np.percentile(vol, 99)),
            b_min=0.0, b_max=1.0, clip=True,
        )
        smoother = GaussianSmooth(sigma=0.8)
        # MONAI expects channel-first: (C, Z, Y, X)
        vol_t = scaler(vol[np.newaxis])
        vol_t = smoother(vol_t)
        vol_n = vol_t[0].numpy()
        method_note = "MONAI ScaleIntensityRange + GaussianSmooth"
    else:
        lo, hi = np.percentile(vol, 1), np.percentile(vol, 99)
        vol_n = np.clip((vol - lo) / (hi - lo + 1e-8), 0, 1)
        method_note = "Percentile normalisation (MONAI not installed)"

    print(f"  Normalised range: {vol_n.min():.3f} – {vol_n.max():.3f}")

    # ── Step 3: Geodesic distance map ────────────────────────
    # Build a small seed sphere around the centroid
    seed_mask = np.zeros((Z, Y, X), dtype=bool)
    zz, yy, xx = np.ogrid[:Z, :Y, :X]
    seed_mask[(zz - cz)**2 + (yy - cy)**2 + (xx - cx)**2 <= 9] = True

    seed_intensity = vol_n[cz, cy, cx]
    intensity_diff = np.abs(vol_n - seed_intensity)

    # Euclidean distance from seed (normalised to [0,1])
    euc_dist = distance_transform_edt(~seed_mask).astype(np.float32)
    euc_dist_norm = euc_dist / (euc_dist.max() + 1e-8)

    # Geodesic = 50% spatial + 50% intensity
    geodesic = 0.5 * euc_dist_norm + 0.5 * intensity_diff
    print(f"  Geodesic map computed — range: {geodesic.min():.3f} – {geodesic.max():.3f}")

    # ── Step 4: Otsu threshold within bounding box ───────────
    roi_geo = geodesic[z0:z1+1, y0:y1+1, x0:x1+1]
    roi_vol = vol_n[z0:z1+1,    y0:y1+1, x0:x1+1]

    try:
        thresh = filters.threshold_otsu(roi_geo)
        print(f"  Otsu threshold: {thresh:.4f}")
    except Exception:
        thresh = roi_geo.mean()
        print(f"  Otsu fallback (mean): {thresh:.4f}")

    binary_roi = roi_geo < thresh   # low geodesic = likely tumour

    # ── Step 5: Random Walker boundary refinement ─────────────
    try:
        markers = np.zeros(roi_geo.shape, dtype=np.int32)
        markers[roi_geo < np.percentile(roi_geo, 20)] = 1   # tumour seeds
        markers[roi_geo > np.percentile(roi_geo, 80)] = 2   # background seeds

        print("  Running Random Walker...")
        rw_labels = skiseg.random_walker(
            roi_vol.astype(np.float64),
            markers,
            beta=10,
            mode='cg',
            multichannel=False,
        )
        binary_roi = rw_labels == 1
        print("  Random Walker complete")
    except Exception as e:
        print(f"  Random Walker failed ({e}) — using Otsu result")

    # ── Step 6: Morphological cleaning ───────────────────────
    binary_roi = ndimage.binary_closing(binary_roi, iterations=3)
    binary_roi = ndimage.binary_fill_holes(binary_roi)

    # Keep only the connected component containing the centroid
    labeled, n = ndimage.label(binary_roi)
    print(f"  Connected components found: {n}")

    if n > 0:
        seed_z = max(0, min(cz - z0, binary_roi.shape[0] - 1))
        seed_y = max(0, min(cy - y0, binary_roi.shape[1] - 1))
        seed_x = max(0, min(cx - x0, binary_roi.shape[2] - 1))
        seed_label = labeled[seed_z, seed_y, seed_x]

        if seed_label == 0:
            # Centroid fell in background — pick largest component
            sizes = ndimage.sum(binary_roi, labeled, range(1, n + 1))
            seed_label = int(np.argmax(sizes)) + 1
            print(f"  Centroid in background — using largest component ({seed_label})")
        else:
            print(f"  Using component {seed_label} (contains centroid)")

        binary_roi = labeled == seed_label

    # Place ROI mask back into full volume
    mask_3d = np.zeros((Z, Y, X), dtype=bool)
    mask_3d[z0:z1+1, y0:y1+1, x0:x1+1] = binary_roi

    return mask_3d, method_note


# ─────────────────────────────────────────────────────────────
# 4. CLASSICAL REGION-GROWING BASELINE
# ─────────────────────────────────────────────────────────────

def segment_region_growing(mri_vol: np.ndarray,
                            centroid: tuple,
                            bbox: tuple) -> np.ndarray:
    """
    Classical 3D region-growing baseline for comparison.
    Intensity threshold + connected components within bounding box.
    """
    print("  Running classical region-growing baseline...")
    Z, Y, X = mri_vol.shape
    cz, cy, cx = centroid
    (z0, y0, x0), (z1, y1, x1) = bbox

    roi = mri_vol[z0:z1+1, y0:y1+1, x0:x1+1].copy().astype(np.float32)
    lo  = np.percentile(roi, 40)

    binary = roi > lo
    labeled, n = ndimage.label(binary)

    if n == 0:
        return np.zeros((Z, Y, X), dtype=bool)

    sz = max(0, min(cz - z0, binary.shape[0] - 1))
    sy = max(0, min(cy - y0, binary.shape[1] - 1))
    sx = max(0, min(cx - x0, binary.shape[2] - 1))
    seed_label = labeled[sz, sy, sx]

    if seed_label == 0:
        sizes = ndimage.sum(binary, labeled, range(1, n + 1))
        seed_label = int(np.argmax(sizes)) + 1

    tumor_roi = labeled == seed_label
    tumor_roi = ndimage.binary_closing(tumor_roi, iterations=2)
    tumor_roi = ndimage.binary_fill_holes(tumor_roi)

    mask_3d = np.zeros((Z, Y, X), dtype=bool)
    mask_3d[z0:z1+1, y0:y1+1, x0:x1+1] = tumor_roi
    return mask_3d


# ─────────────────────────────────────────────────────────────
# 5. ASSESSMENT & VISUALISATION
# ─────────────────────────────────────────────────────────────

def assess_segmentation(mask_3d: np.ndarray,
                         mri_vol: np.ndarray,
                         centroid: tuple,
                         out_dir: Path,
                         method: str = ""):
    """Numerical metrics and visualisation figures."""

    print("\n--- Segmentation assessment ---")
    voxels = int(mask_3d.sum())
    print(f"  Segmented voxels : {voxels:,}")

    if voxels == 0:
        print("  WARNING: mask is empty — check centroid and bbox coordinates")
        return {}

    # Volume (assumes 1mm³ voxels — update with real spacing if available)
    voxel_vol_mm3 = 1.0
    volume_cm3    = voxels * voxel_vol_mm3 / 1000
    print(f"  Estimated volume : {volume_cm3:.2f} cm³  (assuming 1mm³/voxel)")

    coords = np.argwhere(mask_3d)
    mins   = coords.min(axis=0)
    maxs   = coords.max(axis=0)
    extent = (maxs - mins + 1)
    print(f"  Extent (Z,Y,X)   : {extent}")

    mask_centroid = coords.mean(axis=0)
    dist = float(np.linalg.norm(mask_centroid - np.array(centroid)))
    print(f"  User centroid     : {centroid}")
    print(f"  Mask centroid     : ({mask_centroid[0]:.1f}, {mask_centroid[1]:.1f}, {mask_centroid[2]:.1f})")
    print(f"  Centroid shift    : {dist:.2f} voxels")
    if dist > 20:
        print("  ⚠️  Large centroid shift — check that coordinates are in MRI space, not PET space")

    # ── Orthogonal overlay ────────────────────────────────────
    Z, Y, X = mri_vol.shape
    mz = int(np.clip(round(mask_centroid[0]), 0, Z-1))
    my = int(np.clip(round(mask_centroid[1]), 0, Y-1))
    mx = int(np.clip(round(mask_centroid[2]), 0, X-1))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#0d0d0d")
    fig.suptitle(
        f"Tumour Segmentation — {method}",
        color="white", fontsize=12
    )

    for ax, (img, msk, lbl) in zip(axes, [
        (mri_vol[mz],       mask_3d[mz],       f"Axial z={mz}"),
        (mri_vol[:, my, :], mask_3d[:, my, :], f"Coronal y={my}"),
        (mri_vol[:, :, mx], mask_3d[:, :, mx], f"Sagittal x={mx}"),
    ]):
        lo_v, hi_v = np.percentile(img, 1), np.percentile(img, 99)
        img_n = np.clip((img - lo_v) / (hi_v - lo_v + 1e-8), 0, 1)
        ax.set_facecolor("#0d0d0d")
        ax.imshow(img_n, cmap="gray", origin="lower")
        for cnt in measure.find_contours(msk.astype(float), 0.5):
            ax.plot(cnt[:, 1], cnt[:, 0], "r-", linewidth=1.5)
        rgba = np.zeros((*msk.shape, 4), dtype=np.float32)
        rgba[msk] = [1, 0.2, 0.2, 0.35]
        ax.imshow(rgba, origin="lower")
        ax.set_title(lbl, color="white", fontsize=10)
        ax.axis("off")

    red_p = mpatches.Patch(color="red", alpha=0.5, label="Segmented tumour")
    fig.legend(handles=[red_p], loc="lower center",
               fontsize=10, facecolor="#222", labelcolor="white")
    plt.tight_layout()
    seg_path = str(out_dir / "segmentation" / "tumor_segmentation.png")
    plt.savefig(seg_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  Segmentation overlay → {seg_path}")

    # ── 3D surface render ─────────────────────────────────────
    try:
        verts, faces, _, _ = measure.marching_cubes(
            mask_3d.astype(np.float32), level=0.5
        )
        fig3d = plt.figure(figsize=(7, 7))
        ax3d  = fig3d.add_subplot(111, projection="3d")
        ax3d.plot_trisurf(
            verts[:, 2], verts[:, 1], verts[:, 0],
            triangles=faces, color="salmon", alpha=0.65, linewidth=0,
        )
        ax3d.set_title("3D Tumour Surface (Marching Cubes)", color="white")
        ax3d.set_facecolor("#0a0a0a")
        fig3d.patch.set_facecolor("#0a0a0a")
        surf_path = str(out_dir / "segmentation" / "tumor_3d_surface.png")
        fig3d.savefig(surf_path, dpi=120, bbox_inches="tight",
                      facecolor=fig3d.get_facecolor())
        plt.close()
        print(f"  3D surface          → {surf_path}")
        n_verts, n_faces = len(verts), len(faces)
    except Exception as e:
        print(f"  3D surface skipped  : {e}")
        n_verts, n_faces = 0, 0

    # ── Metrics JSON ──────────────────────────────────────────
    metrics = {
        "method":              method,
        "n_voxels":            voxels,
        "volume_cm3":          round(volume_cm3, 3),
        "centroid_z":          round(float(mask_centroid[0]), 2),
        "centroid_y":          round(float(mask_centroid[1]), 2),
        "centroid_x":          round(float(mask_centroid[2]), 2),
        "centroid_shift_vox":  round(dist, 2),
        "extent_z":            int(extent[0]),
        "extent_y":            int(extent[1]),
        "extent_x":            int(extent[2]),
        "surface_vertices":    n_verts,
        "surface_faces":       n_faces,
    }
    metrics_path = str(out_dir / "segmentation" / "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics JSON        → {metrics_path}")
    return metrics


def make_segmentation_gif(mri_vol: np.ndarray,
                           mask_3d: np.ndarray,
                           centroid: tuple,
                           out_path: str,
                           fps: int = 8,
                           radius: int = 15):
    """Axial sweep GIF around the tumour region."""
    cz = centroid[0]
    Z  = mri_vol.shape[0]
    lo, hi = np.percentile(mri_vol, 1), np.percentile(mri_vol, 99)
    frames = []
    z_range = range(max(0, cz - radius), min(Z, cz + radius + 1))
    print(f"  Rendering {len(list(z_range))} frames for sweep GIF...")

    for z in z_range:
        img_n = np.clip((mri_vol[z] - lo) / (hi - lo + 1e-8), 0, 1)
        msk   = mask_3d[z]

        fig, ax = plt.subplots(figsize=(5, 5))
        fig.patch.set_facecolor("#0a0a0a")
        ax.set_facecolor("#0a0a0a")
        ax.imshow(img_n, cmap="gray", origin="lower")

        rgba = np.zeros((*msk.shape, 4), dtype=np.float32)
        rgba[msk] = [1, 0.2, 0.2, 0.45]
        ax.imshow(rgba, origin="lower")

        for cnt in measure.find_contours(msk.astype(float), 0.5):
            ax.plot(cnt[:, 1], cnt[:, 0], "r-", linewidth=1.5)

        ax.set_title(f"Axial slice z={z}", color="white", fontsize=9)
        ax.axis("off")
        plt.tight_layout()

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        frame = np.frombuffer(fig.canvas.tostring_rgb(),
                              dtype=np.uint8).reshape(h, w, 3)
        frames.append(frame)
        plt.close()

    imageio.mimsave(out_path, frames, fps=fps, loop=0)
    print(f"  Sweep GIF → {out_path}")


# ─────────────────────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────────────────────

def main(out_dir: str, centroid_arg, bbox_arg, baseline: bool):

    out_dir = Path(out_dir)
    (out_dir / "segmentation").mkdir(parents=True, exist_ok=True)
    (out_dir / "animations").mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("STEP 3 — Tumour Segmentation (MONAI pipeline)")
    print("="*60)

    mri_vol, pet_coreg, pet_last = load_outputs(out_dir)

    # ── 3a) Tumour location ───────────────────────────────────
    if centroid_arg and bbox_arg:
        cz, cy, cx = [int(v) for v in centroid_arg.split(",")]
        bv = [int(v) for v in bbox_arg.split(",")]
        centroid = (cz, cy, cx)
        bbox     = ((bv[0], bv[1], bv[2]), (bv[3], bv[4], bv[5]))
        print(f"  Centroid : {centroid}")
        print(f"  Bbox     : {bbox}")
    else:
        centroid, bbox = pick_tumour_interactive(pet_last)

    # ── 3b) Segmentation ─────────────────────────────────────
    print("\nRunning MONAI segmentation pipeline...")
    mask_3d, method_note = segment_monai(mri_vol, centroid, bbox)

    np.save(str(out_dir / "tumor_mask.npy"), mask_3d)
    print(f"  Mask saved → {out_dir}/tumor_mask.npy")

    # Optional: also run baseline for comparison
    if baseline:
        print("\nRunning classical region-growing baseline for comparison...")
        mask_baseline = segment_region_growing(mri_vol, centroid, bbox)
        np.save(str(out_dir / "tumor_mask_baseline.npy"), mask_baseline)
        print(f"  Baseline mask saved → {out_dir}/tumor_mask_baseline.npy")

    # ── 3c) Assessment ────────────────────────────────────────
    metrics = assess_segmentation(
        mask_3d, mri_vol, centroid, out_dir,
        method=method_note,
    )

    make_segmentation_gif(
        mri_vol, mask_3d, centroid,
        out_path=str(out_dir / "animations" / "tumor_sweep.gif"),
    )

    print("\n✓ Script 03 complete.")
    print("\nMetrics summary:")
    if metrics:
        for k, v in metrics.items():
            print(f"  {k:<28} : {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="3D Tumour Segmentation — MONAI geodesic + Random Walker"
    )
    parser.add_argument("--out_dir",   default="outputs",
                        help="Output directory (same as scripts 01 and 02)")
    parser.add_argument("--centroid",  default=None,
                        help="Tumour centroid in MRI space: z,y,x  e.g. 45,128,140")
    parser.add_argument("--bbox",      default=None,
                        help="Bounding box in MRI space: z0,y0,x0,z1,y1,x1")
    parser.add_argument("--baseline",  action="store_true",
                        help="Also run classical region-growing for comparison")
    args = parser.parse_args()
    main(args.out_dir, args.centroid, args.bbox, args.baseline)