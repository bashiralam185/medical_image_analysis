"""
Script 03 - 3D AI Tumor Segmentation (MedSAM2 / SAM2)
=======================================================
Objectives:
  3a) Manually define tumor centre and bounding box in the MR image
  3b) Use MedSAM2 (or SAM2) with a bounding box / centroid prompt
      to produce a semi-automatic 3D tumour segmentation
  3c) Visualise and numerically assess the result

Usage:
    python 03_segmentation.py --out_dir outputs [--checkpoint path/to/sam2.pt]

    Run in interactive mode (default) to click on the tumor centre,
    or pass --bbox z0,y0,x0,z1,y1,x1 and --centroid z,y,x directly.

    Example (hard-coded values from visual inspection of the dataset):
    python 03_segmentation.py --centroid 45,128,140 --bbox 35,110,120,55,145,160
"""

import argparse
import warnings
from pathlib import Path

import imageio
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage import measure

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# 1. HELPER: LOAD DATA FROM PREVIOUS SCRIPTS
# ─────────────────────────────────────────────────────────────

def load_outputs(out_dir: Path):
    """Load volumes produced by scripts 01 and 02."""
    pet_last  = np.load(out_dir / "pet_last.npy")          # last PET frame (Z,Y,X)
    mri_vol   = np.load(out_dir / "mri_vol.npy")           # MRI (Z,Y,X)
    pet_coreg = np.load(out_dir / "pet_coreg.npy")         # co-reg PET (Z,Y,X)
    print(f"  MRI shape      : {mri_vol.shape}")
    print(f"  PET last shape : {pet_last.shape}")
    return mri_vol, pet_coreg, pet_last


# ─────────────────────────────────────────────────────────────
# 2. INTERACTIVE / MANUAL TUMOUR LOCALISATION
# ─────────────────────────────────────────────────────────────

class TumorPicker:
    """
    Simple matplotlib-based interactive tool to pick the tumour centre
    and bounding box on the PET last-frame axial slice.
    """
    def __init__(self, pet_last: np.ndarray):
        self.pet    = pet_last
        self.center = None
        self.bbox   = []      # [(z, y, x) clicks]

    def pick_on_axial(self, z_slice: int = None):
        Z, Y, X = self.pet.shape
        z_slice = z_slice if z_slice is not None else Z // 2

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(self.pet[z_slice], cmap="hot", origin="lower")
        ax.set_title(
            f"Click tumour CENTRE (left), then TWO corners of bbox (left).\n"
            f"Right-click to finish. Slice z={z_slice}/{Z}.",
            fontsize=9,
        )
        clicks = []

        def on_click(event):
            if event.inaxes != ax:
                return
            if event.button == 1:
                y, x = int(event.ydata), int(event.xdata)
                clicks.append((z_slice, y, x))
                ax.plot(x, y, "go" if len(clicks) == 1 else "rs", markersize=8)
                fig.canvas.draw()
            elif event.button == 3:
                plt.close()

        fig.canvas.mpl_connect("button_press_event", on_click)
        plt.tight_layout()
        plt.show()

        if len(clicks) >= 1:
            self.center = clicks[0]
        if len(clicks) >= 3:
            pts = np.array(clicks[1:])
            self.bbox = [
                (pts[:, 0].min(), pts[:, 1].min(), pts[:, 2].min()),
                (pts[:, 0].max(), pts[:, 1].max(), pts[:, 2].max()),
            ]
        return self.center, self.bbox


# ─────────────────────────────────────────────────────────────
# 3. SAM2 / MedSAM2 SEGMENTATION
# ─────────────────────────────────────────────────────────────

def segment_with_sam2(mri_vol: np.ndarray,
                      centroid: tuple,
                      bbox: tuple,
                      checkpoint: str,
                      model_cfg: str = "sam2.1_hiera_large.yaml") -> np.ndarray:
    """
    Run SAM2 / MedSAM2 slice-by-slice on the MRI volume using a
    2-D bounding box on the key axial slice, then propagate through
    the volume with the video predictor.

    centroid : (z, y, x)
    bbox     : ((z0,y0,x0), (z1,y1,x1))   3-D bounding box corners

    Returns:
        mask_3d : np.ndarray bool (Z, Y, X)
    """
    try:
        from sam2.build_sam import build_sam2_video_predictor
        import torch
    except ImportError:
        raise ImportError(
            "SAM2 not installed. Run:\n"
            "  pip install git+https://github.com/facebookresearch/sam2.git\n"
            "and download a checkpoint."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device}")

    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)

    Z, Y, X = mri_vol.shape
    z0, y0, x0 = bbox[0]
    z1, y1, x1 = bbox[1]
    cz, cy, cx = centroid

    # Normalise MRI to uint8 for SAM2 (it expects RGB images)
    lo, hi = np.percentile(mri_vol, 1), np.percentile(mri_vol, 99)
    mri_norm = np.clip((mri_vol - lo) / (hi - lo + 1e-8), 0, 1)

    def slice_to_rgb(sl):
        gray = (sl * 255).astype(np.uint8)
        return np.stack([gray, gray, gray], axis=-1)

    # ── Build "video" from axial slices ──────────────────────
    # SAM2's video predictor treats the Z axis as time.
    with predictor.init_state_from_frames(
        frames_iter=(slice_to_rgb(mri_norm[z]) for z in range(Z)),
        video_length=Z,
    ) as state:
        # Add bounding box prompt on the centre axial slice
        box_2d = np.array([x0, y0, x1, y1], dtype=np.float32)  # SAM2 uses (x0,y0,x1,y1)
        _, _, out_masks = predictor.add_new_prompts(
            inference_state=state,
            frame_idx=cz,
            obj_id=1,
            boxes=box_2d[None],          # (1, 4)
        )

        # Propagate forward and backward
        mask_3d = np.zeros((Z, Y, X), dtype=bool)

        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
            m = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze()
            mask_3d[out_frame_idx] = m

        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
            state, start_frame_idx=cz, reverse=True
        ):
            m = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze()
            mask_3d[out_frame_idx] |= m

    return mask_3d


def segment_fallback(mri_vol: np.ndarray,
                     centroid: tuple,
                     bbox: tuple) -> np.ndarray:
    """
    Fallback segmentation when SAM2 is not available:
    Region-growing (flood fill) from the centroid within the bounding box,
    applied to the MRI intensity.

    This is a reasonable classical baseline — replace with SAM2 in practice.
    """
    print("  [Fallback] Using intensity-based region growing (no SAM2 checkpoint found)")
    Z, Y, X = mri_vol.shape
    cz, cy, cx = centroid
    (z0, y0, x0), (z1, y1, x1) = bbox

    # Crop ROI
    roi = mri_vol[z0:z1+1, y0:y1+1, x0:x1+1].copy()
    lo  = np.percentile(roi, 40)    # local intensity threshold
    hi  = np.percentile(roi, 100)

    # Binary seed mask from high intensity
    seed_z = cz - z0
    seed_y = cy - y0
    seed_x = cx - x0

    binary = roi > lo
    # Label connected components and pick the one containing the centroid
    labeled, n = ndimage.label(binary)
    if n == 0:
        return np.zeros((Z, Y, X), dtype=bool)

    seed_label = labeled[seed_z, seed_y, seed_x]
    if seed_label == 0:
        # Seed is background — pick largest component
        sizes = ndimage.sum(binary, labeled, range(1, n+1))
        seed_label = int(np.argmax(sizes)) + 1

    tumor_roi = labeled == seed_label

    # Apply morphological closing to smooth the mask
    tumor_roi = ndimage.binary_closing(tumor_roi, iterations=2)
    tumor_roi = ndimage.binary_fill_holes(tumor_roi)

    mask_3d = np.zeros((Z, Y, X), dtype=bool)
    mask_3d[z0:z1+1, y0:y1+1, x0:x1+1] = tumor_roi
    return mask_3d


# ─────────────────────────────────────────────────────────────
# 4. ASSESSMENT & VISUALISATION
# ─────────────────────────────────────────────────────────────

def assess_segmentation(mask_3d: np.ndarray,
                        mri_vol:  np.ndarray,
                        centroid: tuple,
                        out_dir:  Path):
    """Compute numerical metrics and produce visualisation figures."""

    print("\n--- Segmentation assessment ---")
    voxels = int(mask_3d.sum())
    print(f"  Segmented voxels : {voxels:,}")

    # Volume estimation (requires voxel size — use 1mm³ placeholder)
    voxel_vol_mm3 = 1.0   # update with actual voxel dimensions if available
    volume_cm3    = voxels * voxel_vol_mm3 / 1000
    print(f"  Estimated volume : {volume_cm3:.2f} cm³  (assuming 1 mm³/voxel)")

    # Bounding box of the mask
    coords = np.argwhere(mask_3d)
    if len(coords):
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        extents = (maxs - mins + 1)
        print(f"  Mask bounding box: z={mins[0]}:{maxs[0]}  "
              f"y={mins[1]}:{maxs[1]}  x={mins[2]}:{maxs[2]}")
        print(f"  Extent (Z,Y,X)   : {extents}")
    else:
        print("  WARNING: mask is empty")
        return

    # Centroid of mask vs. user-supplied centroid
    mask_centroid = coords.mean(axis=0)
    cz, cy, cx    = centroid
    dist = np.linalg.norm(mask_centroid - np.array([cz, cy, cx]))
    print(f"  User centroid     : {centroid}")
    print(f"  Mask centroid     : ({mask_centroid[0]:.1f}, {mask_centroid[1]:.1f}, {mask_centroid[2]:.1f})")
    print(f"  Centroid distance : {dist:.2f} voxels")

    # ── Visualise: 3 orthogonal planes through mask centroid ──
    mz, my, mx = [int(round(v)) for v in mask_centroid]
    Z, Y, X = mri_vol.shape

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#0d0d0d")
    fig.suptitle("Tumor Segmentation — Orthogonal Views", color="white", fontsize=13)

    planes = [
        (mri_vol[mz, :, :],   mask_3d[mz, :, :],   f"Axial z={mz}"),
        (mri_vol[:, my, :],   mask_3d[:, my, :],   f"Coronal y={my}"),
        (mri_vol[:, :, mx],   mask_3d[:, :, mx],   f"Sagittal x={mx}"),
    ]
    for ax, (img, msk, lbl) in zip(axes, planes):
        lo, hi = np.percentile(img, 1), np.percentile(img, 99)
        img_n  = np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)
        ax.set_facecolor("#0d0d0d")
        ax.imshow(img_n, cmap="gray", origin="lower")
        # Overlay mask contour
        contours = measure.find_contours(msk.astype(float), 0.5)
        for cnt in contours:
            ax.plot(cnt[:, 1], cnt[:, 0], "r-", linewidth=1.5)
        # Overlay mask as semi-transparent fill
        mask_rgba = np.zeros((*msk.shape, 4), dtype=np.float32)
        mask_rgba[msk] = [1, 0.2, 0.2, 0.35]
        ax.imshow(mask_rgba, origin="lower")
        ax.set_title(lbl, color="white", fontsize=10)
        ax.axis("off")

    red_patch = mpatches.Patch(color="red", alpha=0.5, label="Segmented tumour")
    fig.legend(handles=[red_patch], loc="lower center",
               ncol=1, fontsize=10, facecolor="#222", labelcolor="white")
    plt.tight_layout()
    seg_path = str(out_dir / "segmentation" / "tumor_segmentation.png")
    plt.savefig(seg_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  Segmentation figure → {seg_path}")

    # ── 3D surface render (matplotlib) ───────────────────────
    try:
        verts, faces, _, _ = measure.marching_cubes(mask_3d.astype(np.float32), level=0.5)
        fig3d = plt.figure(figsize=(7, 7))
        ax3d  = fig3d.add_subplot(111, projection="3d")
        ax3d.plot_trisurf(verts[:, 2], verts[:, 1], verts[:, 0],
                          triangles=faces, color="salmon", alpha=0.6, linewidth=0)
        ax3d.set_title("3D Tumor Surface", color="white")
        ax3d.set_facecolor("#0a0a0a")
        fig3d.patch.set_facecolor("#0a0a0a")
        surf_path = str(out_dir / "segmentation" / "tumor_3d_surface.png")
        fig3d.savefig(surf_path, dpi=120, bbox_inches="tight",
                      facecolor=fig3d.get_facecolor())
        plt.close()
        print(f"  3D surface figure  → {surf_path}")
    except Exception as e:
        print(f"  (3D surface skipped: {e})")

    # ── Metrics summary ───────────────────────────────────────
    metrics = {
        "n_voxels":        voxels,
        "volume_cm3":      round(volume_cm3, 3),
        "centroid_z":      round(float(mask_centroid[0]), 2),
        "centroid_y":      round(float(mask_centroid[1]), 2),
        "centroid_x":      round(float(mask_centroid[2]), 2),
        "centroid_dist_vox": round(float(dist), 2),
        "extent_z":        int(extents[0]),
        "extent_y":        int(extents[1]),
        "extent_x":        int(extents[2]),
    }
    import json
    metrics_path = str(out_dir / "segmentation" / "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics JSON       → {metrics_path}")
    return metrics


def make_segmentation_gif(mri_vol: np.ndarray,
                           mask_3d: np.ndarray,
                           centroid: tuple,
                           out_path: str,
                           fps: int = 8):
    """GIF sweeping through axial slices around the tumour."""
    cz, cy, cx = centroid
    Z = mri_vol.shape[0]
    z_range = range(max(0, cz - 15), min(Z, cz + 16))
    lo, hi  = np.percentile(mri_vol, 1), np.percentile(mri_vol, 99)

    frames_gif = []
    for z in z_range:
        img   = np.clip((mri_vol[z] - lo) / (hi - lo + 1e-8), 0, 1)
        msk   = mask_3d[z]

        fig, ax = plt.subplots(figsize=(6, 6))
        fig.patch.set_facecolor("#0a0a0a")
        ax.set_facecolor("#0a0a0a")
        ax.imshow(img, cmap="gray", origin="lower")

        mask_rgba = np.zeros((*msk.shape, 4), dtype=np.float32)
        mask_rgba[msk] = [1, 0.2, 0.2, 0.45]
        ax.imshow(mask_rgba, origin="lower")

        contours = measure.find_contours(msk.astype(float), 0.5)
        for cnt in contours:
            ax.plot(cnt[:, 1], cnt[:, 0], "r-", linewidth=1.5)

        ax.set_title(f"Axial slice z={z}", color="white", fontsize=10)
        ax.axis("off")
        plt.tight_layout()
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
        frames_gif.append(frame)
        plt.close()

    imageio.mimsave(out_path, frames_gif, fps=fps, loop=0)
    print(f"  Sweep GIF → {out_path}")


# ─────────────────────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────────────────────

def main(out_dir: str, checkpoint,
         centroid_arg, bbox_arg):

    out_dir = Path(out_dir)
    (out_dir / "segmentation").mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("STEP 3 — Tumor Segmentation")
    print("="*60)

    # ── Load volumes ──────────────────────────────────────────
    mri_vol, pet_coreg, pet_last = load_outputs(out_dir)

    # ── 3a) Tumour location ───────────────────────────────────
    if centroid_arg and bbox_arg:
        # Passed via CLI, e.g. --centroid 45,128,140 --bbox 35,110,120,55,145,160
        cz, cy, cx = [int(v) for v in centroid_arg.split(",")]
        bvals = [int(v) for v in bbox_arg.split(",")]
        bbox = ((bvals[0], bvals[1], bvals[2]), (bvals[3], bvals[4], bvals[5]))
        centroid = (cz, cy, cx)
        print(f"  Using supplied centroid : {centroid}")
        print(f"  Using supplied bbox     : {bbox}")
    else:
        # Interactive picking on PET last frame
        print("\n[Interactive] A window will open — click to set tumour centroid & bbox.")
        print("  → Inspect the PET last-frame first to find the hot spot (tumour).")

        Z = pet_last.shape[0]
        # Show a montage to help the user find the right slice
        fig, axes = plt.subplots(4, Z // 4 + 1, figsize=(16, 8))
        axes = axes.flatten()
        vmin, vmax = np.percentile(pet_last, 1), np.percentile(pet_last, 99)
        for i, ax in enumerate(axes):
            if i < Z:
                ax.imshow(pet_last[i], cmap="hot", vmin=vmin, vmax=vmax, origin="lower")
                ax.set_title(str(i), fontsize=6, color="white")
            ax.axis("off")
            ax.set_facecolor("#0a0a0a")
        fig.patch.set_facecolor("#0a0a0a")
        fig.suptitle("PET last frame — all axial slices. Close to proceed.", color="white")
        plt.tight_layout()
        plt.show()

        z_input = input("  Enter the axial slice z to pick tumour on: ").strip()
        z_slice = int(z_input) if z_input.isdigit() else Z // 2

        picker = TumorPicker(pet_last)
        centroid, bbox_pts = picker.pick_on_axial(z_slice)

        if centroid is None:
            print("  No centroid picked. Using image centre as fallback.")
            Z, Y, X = mri_vol.shape
            centroid = (Z // 2, Y // 2, X // 2)

        if not bbox_pts:
            # Build a default 20-voxel box around centroid
            cz, cy, cx = centroid
            d = 20
            bbox = ((max(0, cz-d), max(0, cy-d), max(0, cx-d)),
                    (min(Z-1, cz+d), min(Y-1, cy+d), min(X-1, cx+d)))
        else:
            bbox = tuple(bbox_pts[:2])

        print(f"  Centroid : {centroid}")
        print(f"  Bbox     : {bbox}")

    # ── 3b) Segmentation ──────────────────────────────────────
    print("\nRunning segmentation …")
    if checkpoint and Path(checkpoint).exists():
        print(f"  Loading SAM2 checkpoint: {checkpoint}")
        mask_3d = segment_with_sam2(mri_vol, centroid, bbox, checkpoint)
    else:
        if checkpoint:
            print(f"  WARNING: checkpoint not found at '{checkpoint}'")
        mask_3d = segment_fallback(mri_vol, centroid, bbox)

    # Save mask
    np.save(str(out_dir / "tumor_mask.npy"), mask_3d)
    print(f"  Mask saved → {out_dir}/tumor_mask.npy")

    # ── 3c) Assessment & visualisation ───────────────────────
    metrics = assess_segmentation(mask_3d, mri_vol, centroid, out_dir)

    # Sweep GIF
    make_segmentation_gif(
        mri_vol, mask_3d, centroid,
        out_path=str(out_dir / "animations" / "tumor_sweep.gif"),
    )

    print("\n✓ Script 03 complete. All outputs written to:", out_dir)
    print("\nSummary of metrics:")
    if metrics:
        for k, v in metrics.items():
            print(f"  {k:<25} : {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D AI Tumor Segmentation")
    parser.add_argument("--out_dir",    default="outputs")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to SAM2 / MedSAM2 .pt checkpoint file")
    parser.add_argument("--centroid",   default=None,
                        help="Tumour centroid as z,y,x  e.g. 45,128,140")
    parser.add_argument("--bbox",       default=None,
                        help="Bounding box as z0,y0,x0,z1,y1,x1  e.g. 35,110,120,55,145,160")
    args = parser.parse_args()
    main(args.out_dir, args.checkpoint, args.centroid, args.bbox)