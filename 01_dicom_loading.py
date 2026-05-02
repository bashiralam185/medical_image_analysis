"""
Script 01 - DICOM Loading & Visualization
==========================================
Objectives:
  1c) Load dynamic PET DICOM, rearrange pixel array using headers
  1d) Visualize last frame and average of all frames
  1e) Create GIF animation of the 3 median planes across frames

Usage:
    python 01_dicom_loading.py --pet_dir data/pet --out_dir outputs
"""

import argparse
import os
import warnings
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pydicom
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────────────────────
# 1. DICOM LOADING & PIXEL ARRAY REARRANGEMENT
# ─────────────────────────────────────────────────────────────

def load_dynamic_pet(pet_dir: str) -> dict:
    """
    Load a dynamic PET DICOM study and rearrange the pixel array.

    The Siemens enhanced DICOM format stores all frames in a single file
    (or a small set of files). The flat pixel_array shape is:
        (num_frames * num_slices, rows, cols)
    We rearrange it to:
        (num_frames, num_slices, rows, cols)

    Returns a dict with the 4-D array and all relevant metadata.
    """
    pet_dir = Path(pet_dir)
    dcm_files = sorted(pet_dir.glob("**/*.dcm"))

    if not dcm_files:
        raise FileNotFoundError(f"No .dcm files found in {pet_dir}")

    print(f"Found {len(dcm_files)} DICOM file(s) in {pet_dir}")

    # ── Try enhanced (single-file) DICOM first ──────────────────
    # Enhanced PET stores everything in one file; classic PET uses
    # one file per slice per frame.
    if len(dcm_files) == 1:
        ds = pydicom.dcmread(str(dcm_files[0]))
        return _parse_enhanced_pet(ds)
    else:
        return _parse_classic_pet(dcm_files)


def _parse_enhanced_pet(ds: pydicom.Dataset) -> dict:
    """Parse an enhanced (multi-frame) PET DICOM file."""

    # ── Mandatory header tags ────────────────────────────────────
    n_frames_total = int(ds[0x0028, 0x0008].value)  # Number of Frames
    rows            = int(ds[0x0028, 0x0010].value)  # Rows
    cols            = int(ds[0x0028, 0x0011].value)  # Columns

    print(f"  Total frames in file : {n_frames_total}")
    print(f"  Rows × Cols          : {rows} × {cols}")

    # ── Siemens private tags ─────────────────────────────────────
    frame_positions  = _safe_tag(ds, 0x0055, 0x1002)  # Frame Positions Vector
    frame_start_times= _safe_tag(ds, 0x0055, 0x1001)  # Frame Start Times (ms)
    frame_durations  = _safe_tag(ds, 0x0055, 0x1004)  # Frame Durations (ms)

    # ── Spatial resolution ───────────────────────────────────────
    pixel_spacing = [float(v) for v in ds[0x0028, 0x0030].value]   # [row_sp, col_sp] mm
    try:
        slice_thickness = float(ds[0x0018, 0x0088].value)           # Spacing Between Slices
    except KeyError:
        slice_thickness = float(ds[0x0050, 0x0010].value) if (0x0050, 0x0010) in ds else 1.0

    # ── Determine num_time_frames & num_slices ───────────────────
    # Strategy: use Frame Positions Vector to count unique z positions
    if frame_positions is not None:
        fp = np.array(frame_positions, dtype=float)
        # Shape might be (3*n_total,) or (n_total, 3)
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
        # Fall back: assume square-root split (unlikely but safe)
        n_slices = int(np.sqrt(n_frames_total))
        n_time   = n_frames_total // n_slices

    print(f"  Time frames          : {n_time}")
    print(f"  Slices per frame     : {n_slices}")

    # ── Raw pixel array ──────────────────────────────────────────
    raw = ds.pixel_array  # shape: (n_frames_total, rows, cols)

    # Apply rescale slope / intercept if present
    slope     = float(getattr(ds, "RescaleSlope",     1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    raw       = raw.astype(np.float32) * slope + intercept

    # ── Rearrange to (n_time, n_slices, rows, cols) ──────────────
    # Siemens typically interleaves: frame0_slice0, frame0_slice1, ..., frame1_slice0, ...
    pet_4d = raw.reshape(n_time, n_slices, rows, cols)

    print(f"  Rearranged shape     : {pet_4d.shape}  (time, slices, rows, cols)")

    return {
        "pet_4d":          pet_4d,          # (T, Z, Y, X) float32
        "n_time":          n_time,
        "n_slices":        n_slices,
        "rows":            rows,
        "cols":            cols,
        "pixel_spacing":   pixel_spacing,   # [row_mm, col_mm]
        "slice_thickness": slice_thickness, # mm
        "frame_start_times": frame_start_times,
        "frame_durations":   frame_durations,
        "frame_positions":   frame_positions,
        "dataset":         ds,
    }


def _parse_classic_pet(dcm_files: list) -> dict:
    """
    Parse a classic PET study where each slice/frame is a separate file.
    Files are sorted by InstanceNumber; we then reshape by reading
    the temporal position and slice location from each header.
    """
    slices = []
    for f in tqdm(dcm_files, desc="Reading DICOM files"):
        ds = pydicom.dcmread(str(f))
        slices.append(ds)

    # Sort by TemporalPositionIdentifier then ImagePositionPatient Z
    def sort_key(ds):
        t = int(getattr(ds, "TemporalPositionIdentifier", 1))
        z = float(ds.ImagePositionPatient[2]) if hasattr(ds, "ImagePositionPatient") else 0.0
        return (t, z)

    slices.sort(key=sort_key)

    rows = int(slices[0][0x0028, 0x0010].value)
    cols = int(slices[0][0x0028, 0x0011].value)

    # Count unique time frames and unique z positions
    t_ids = sorted(set(int(getattr(s, "TemporalPositionIdentifier", 1)) for s in slices))
    z_pos = sorted(set(round(float(s.ImagePositionPatient[2]), 2)
                        for s in slices if hasattr(s, "ImagePositionPatient")))
    n_time   = len(t_ids)
    n_slices = len(z_pos)

    # Build 4D array
    pet_4d = np.zeros((n_time, n_slices, rows, cols), dtype=np.float32)
    for ds in slices:
        t  = t_ids.index(int(getattr(ds, "TemporalPositionIdentifier", 1)))
        zv = round(float(ds.ImagePositionPatient[2]), 2)
        z  = z_pos.index(zv)
        sl = ds.pixel_array.astype(np.float32)
        slope     = float(getattr(ds, "RescaleSlope",     1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        pet_4d[t, z] = sl * slope + intercept

    pixel_spacing   = [float(v) for v in slices[0][0x0028, 0x0030].value]
    try:
        slice_thickness = float(slices[0][0x0018, 0x0088].value)
    except Exception:
        slice_thickness = 1.0

    print(f"  Rearranged shape: {pet_4d.shape}  (time, slices, rows, cols)")

    return {
        "pet_4d":          pet_4d,
        "n_time":          n_time,
        "n_slices":        n_slices,
        "rows":            rows,
        "cols":            cols,
        "pixel_spacing":   pixel_spacing,
        "slice_thickness": slice_thickness,
        "frame_start_times": None,
        "frame_durations":   None,
        "frame_positions":   None,
        "dataset":           slices[0],
    }


def _safe_tag(ds, group, element):
    """Return the value of a DICOM tag or None if not present."""
    try:
        return list(ds[group, element].value)
    except (KeyError, TypeError):
        return None


# ─────────────────────────────────────────────────────────────
# 2. VISUALIZATION HELPERS
# ─────────────────────────────────────────────────────────────

def normalize(arr: np.ndarray, vmin=None, vmax=None) -> np.ndarray:
    """Clip and normalize a float array to [0, 1]."""
    vmin = vmin if vmin is not None else np.percentile(arr, 1)
    vmax = vmax if vmax is not None else np.percentile(arr, 99)
    return np.clip((arr - vmin) / (vmax - vmin + 1e-8), 0, 1)


def plot_three_planes(volume: np.ndarray, title: str, out_path: str,
                      cmap="hot", vmin=None, vmax=None):
    """
    Plot the three median orthogonal planes (axial, coronal, sagittal)
    of a 3-D volume  (Z, Y, X).
    """
    Z, Y, X = volume.shape
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#0d0d0d")
    for ax in axes:
        ax.set_facecolor("#0d0d0d")

    planes = [
        (volume[Z // 2, :, :], f"Axial  (z={Z//2})"),
        (volume[:, Y // 2, :], f"Coronal (y={Y//2})"),
        (volume[:, :, X // 2], f"Sagittal (x={X//2})"),
    ]
    for ax, (plane, label) in zip(axes, planes):
        im = ax.imshow(plane, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
        ax.set_title(label, color="white", fontsize=11)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title, color="white", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved → {out_path}")


# ─────────────────────────────────────────────────────────────
# 3. ANIMATION
# ─────────────────────────────────────────────────────────────

def make_planes_animation(pet_4d: np.ndarray, out_path: str,
                           fps: int = 4, cmap: str = "hot"):
    """
    Create a GIF showing the 3 median orthogonal planes across all time frames.
    pet_4d shape: (T, Z, Y, X)
    """
    T, Z, Y, X = pet_4d.shape
    vmin = np.percentile(pet_4d,  1)
    vmax = np.percentile(pet_4d, 99)

    frames_gif = []
    print(f"\nRendering {T} frames for animation …")

    for t in tqdm(range(T)):
        vol = pet_4d[t]           # (Z, Y, X)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.patch.set_facecolor("#0d0d0d")
        for ax in axes:
            ax.set_facecolor("#0d0d0d")

        planes = [
            (vol[Z // 2, :, :], "Axial"),
            (vol[:, Y // 2, :], "Coronal"),
            (vol[:, :, X // 2], "Sagittal"),
        ]
        for ax, (plane, label) in zip(axes, planes):
            ax.imshow(plane, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
            ax.set_title(label, color="white", fontsize=10)
            ax.axis("off")

        fig.suptitle(f"Dynamic PET — Time frame {t+1}/{T}",
                     color="white", fontsize=12)
        plt.tight_layout()

        # Render to numpy array
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(h, w, 3)
        frames_gif.append(img)
        plt.close()

    imageio.mimsave(out_path, frames_gif, fps=fps, loop=0)
    print(f"  Animation saved → {out_path}")


# ─────────────────────────────────────────────────────────────
# 4. MAIN
# ─────────────────────────────────────────────────────────────

def main(pet_dir: str, out_dir: str):
    out_dir = Path(out_dir)
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)
    (out_dir / "animations").mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 1 — Loading dynamic PET DICOM")
    print("="*60)
    data = load_dynamic_pet(pet_dir)
    pet_4d = data["pet_4d"]           # (T, Z, Y, X)
    T, Z, Y, X = pet_4d.shape

    # ── Print metadata ────────────────────────────────────────
    print(f"\nMetadata summary:")
    print(f"  Shape        : {pet_4d.shape}  (frames, slices, rows, cols)")
    print(f"  Pixel spacing: {data['pixel_spacing']} mm")
    print(f"  Slice spacing: {data['slice_thickness']} mm")
    if data["frame_durations"]:
        durs = np.array(data["frame_durations"])
        print(f"  Frame durations (ms): min={durs.min():.0f}  max={durs.max():.0f}")
    if data["frame_start_times"]:
        times = np.array(data["frame_start_times"])
        print(f"  Frame start times (ms): {times[:5]} …")

    # ── 1d) Last frame ────────────────────────────────────────
    print("\nSTEP 1d — Visualizing last frame and temporal average")
    last_frame = pet_4d[-1]           # (Z, Y, X)
    avg_frame  = pet_4d.mean(axis=0)  # (Z, Y, X)

    vmin = np.percentile(pet_4d,  1)
    vmax = np.percentile(pet_4d, 99)

    plot_three_planes(
        last_frame, "PET — Last Time Frame",
        str(out_dir / "frames" / "last_frame.png"),
        vmin=vmin, vmax=vmax,
    )
    plot_three_planes(
        avg_frame, "PET — Average of All Frames",
        str(out_dir / "frames" / "average_frame.png"),
        vmin=vmin, vmax=vmax,
    )

    # ── Side-by-side comparison ───────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.patch.set_facecolor("#0d0d0d")
    fig.suptitle("PET: Average vs Last Frame (axial / coronal / sagittal)",
                 color="white", fontsize=13)

    for row, (vol, label) in enumerate([(avg_frame, "Average"), (last_frame, "Last frame")]):
        planes = [
            vol[Z // 2, :, :],
            vol[:, Y // 2, :],
            vol[:, :, X // 2],
        ]
        plane_labels = ["Axial", "Coronal", "Sagittal"]
        for col, (plane, pl) in enumerate(zip(planes, plane_labels)):
            ax = axes[row, col]
            ax.set_facecolor("#0d0d0d")
            im = ax.imshow(plane, cmap="hot", vmin=vmin, vmax=vmax, origin="lower")
            ax.set_title(f"{label} — {pl}", color="white", fontsize=10)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    comp_path = str(out_dir / "frames" / "comparison_avg_vs_last.png")
    plt.savefig(comp_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved → {comp_path}")

    # ── 1e) Animation ─────────────────────────────────────────
    print("\nSTEP 1e — Creating planes animation GIF")
    make_planes_animation(
        pet_4d,
        out_path=str(out_dir / "animations" / "pet_planes_dynamic.gif"),
        fps=4,
    )

    # ── Save the 4D array for next scripts ───────────────────
    np.save(str(out_dir / "pet_4d.npy"), pet_4d)
    np.save(str(out_dir / "pet_avg.npy"), avg_frame)
    np.save(str(out_dir / "pet_last.npy"), last_frame)

    # Save metadata
    np.save(str(out_dir / "pet_meta.npy"), {
        "pixel_spacing":   data["pixel_spacing"],
        "slice_thickness": data["slice_thickness"],
        "n_time":          T,
        "n_slices":        Z,
        "rows":            Y,
        "cols":            X,
    }, allow_pickle=True)

    print("\n✓ Script 01 complete. Outputs saved to:", out_dir)
    print("  → Run python 02_coregistration.py next\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DICOM PET Loading & Visualization")
    parser.add_argument("--pet_dir", default="data/pet",
                        help="Directory containing PET DICOM files")
    parser.add_argument("--out_dir", default="outputs",
                        help="Root output directory")
    args = parser.parse_args()
    main(args.pet_dir, args.out_dir)