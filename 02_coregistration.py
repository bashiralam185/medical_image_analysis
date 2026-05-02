"""
Script 02 - 3D Rigid Coregistration (PET average → MRI)
=========================================================
Objectives:
  2a) Coregister PET average to MRI using SimpleITK (Mattes MI + gradient descent)
  2b) Create rotating MIP GIF (coronal-sagittal) for:
        i)  Reference MRI
        ii) Co-registered PET
        iii) Alpha-fusion of both

Usage:
    python 02_coregistration.py --mri_dir data/mri --out_dir outputs
"""

import argparse
import os
import warnings
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pydicom
import SimpleITK as sitk
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# 1. MRI LOADING
# ─────────────────────────────────────────────────────────────

def load_mri_dicom(mri_dir: str) -> sitk.Image:
    """Load an MRI DICOM series into a SimpleITK image."""
    mri_dir = Path(mri_dir)
    reader  = sitk.ImageSeriesReader()

    dcm_names = reader.GetGDCMSeriesFileNames(str(mri_dir))
    if not dcm_names:
        # Fallback: grab all .dcm files recursively
        dcm_names = sorted([str(f) for f in mri_dir.glob("**/*.dcm")])

    if not dcm_names:
        raise FileNotFoundError(f"No DICOM series found in {mri_dir}")

    reader.SetFileNames(dcm_names)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    img = reader.Execute()
    print(f"  MRI loaded: size={img.GetSize()}  spacing={img.GetSpacing()}")
    return img


def numpy_to_sitk(arr: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> sitk.Image:
    """Convert a numpy array (Z, Y, X) to a SimpleITK image with given spacing."""
    # SimpleITK expects (X, Y, Z) order
    img = sitk.GetImageFromArray(arr.astype(np.float32))
    img.SetSpacing(tuple(float(s) for s in spacing))
    return img


# ─────────────────────────────────────────────────────────────
# 2. COREGISTRATION
# ─────────────────────────────────────────────────────────────

def coregister_pet_to_mri(pet_avg: np.ndarray,
                           pet_spacing: tuple,
                           mri_sitk: sitk.Image,
                           out_dir: Path) -> tuple:
    """
    Perform 3D rigid coregistration: PET average (moving) → MRI (fixed).

    Strategy:
      - Metric   : Mattes Mutual Information  (ideal for multimodal)
      - Optimizer: Gradient Descent w/ line search
      - Transform: Euler3DTransform (rigid: 3 rotations + 3 translations)
      - Init     : Geometry-based centre alignment

    Returns:
        (coreg_pet_sitk, final_transform)
    """
    print("\n" + "="*60)
    print("STEP 2a — 3D Rigid Coregistration")
    print("="*60)

    # Build SimpleITK images
    pet_sitk = numpy_to_sitk(pet_avg, spacing=pet_spacing)
    def _to_float32(sitk_img):
        arr = sitk.GetArrayFromImage(sitk_img).astype(np.float32).squeeze()
        if arr.ndim == 2:
            arr = arr[np.newaxis]
        out = sitk.GetImageFromArray(arr)
        spacing   = sitk_img.GetSpacing()
        origin    = sitk_img.GetOrigin()
        direction = sitk_img.GetDirection()
        out.SetSpacing(spacing[:3])
        out.SetOrigin(origin[:3])
        if len(direction) == 9:
            out.SetDirection(direction)
        return out

    fixed    = _to_float32(mri_sitk)
    moving   = _to_float32(pet_sitk)

    # ── Initialisation ────────────────────────────────────────
    initial_tf = sitk.CenteredTransformInitializer(
        fixed, moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    print("  Initial transform set (geometry-based centre alignment)")

    # ── Registration method ───────────────────────────────────
    reg = sitk.ImageRegistrationMethod()

    # Metric
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=64)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.15)

    # Interpolator
    reg.SetInterpolator(sitk.sitkLinear)

    # Optimizer
    reg.SetOptimizerAsGradientDescentLineSearch(
        learningRate=1.0,
        numberOfIterations=200,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    reg.SetOptimizerScalesFromPhysicalShift()

    # Multi-resolution pyramid
    reg.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    reg.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    reg.SetInitialTransform(initial_tf, inPlace=False)

    # ── Progress callback ─────────────────────────────────────
    iteration_log = []
    def iteration_callback():
        it  = reg.GetOptimizerIteration()
        val = reg.GetMetricValue()
        iteration_log.append((it, val))
        if it % 25 == 0:
            print(f"    iter {it:4d}  metric={val:.5f}")

    reg.AddCommand(sitk.sitkIterationEvent, iteration_callback)

    print("  Running registration (this may take 1–3 min) …")
    final_tf = reg.Execute(fixed, moving)
    print(f"  ✓ Converged  metric={reg.GetMetricValue():.5f}  "
          f"stop={reg.GetOptimizerStopConditionDescription()}")

    # ── Resample moving into fixed space ──────────────────────
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(final_tf)
    coreg_pet = resampler.Execute(moving)

    # ── Save transform ────────────────────────────────────────
    tf_path = str(out_dir / "coregistration" / "rigid_transform.tfm")
    sitk.WriteTransform(final_tf, tf_path)
    print(f"  Transform saved → {tf_path}")

    # ── Save registration convergence plot ────────────────────
    if iteration_log:
        iters, vals = zip(*iteration_log)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(iters, vals, color="#e07b39", linewidth=1.5)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Mattes MI (negative)")
        ax.set_title("Coregistration convergence")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        conv_path = str(out_dir / "coregistration" / "convergence.png")
        fig.savefig(conv_path, dpi=120)
        plt.close()
        print(f"  Convergence plot → {conv_path}")

    return coreg_pet, final_tf


# ─────────────────────────────────────────────────────────────
# 3. MAXIMUM INTENSITY PROJECTION UTILS
# ─────────────────────────────────────────────────────────────

def compute_mip(volume: np.ndarray, axis: int) -> np.ndarray:
    """Return the Maximum Intensity Projection along the given axis."""
    return volume.max(axis=axis)


def rotate_volume(volume: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Rotate a 3D volume (Z, Y, X) around the Z axis by angle_deg degrees
    using scipy ndimage for speed.
    """
    from scipy.ndimage import rotate as nd_rotate
    return nd_rotate(volume, angle_deg, axes=(1, 2),
                     reshape=False, order=1, cval=0.0)


def norm01(arr: np.ndarray, plo=1, phi=99) -> np.ndarray:
    lo, hi = np.percentile(arr, plo), np.percentile(arr, phi)
    return np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1)


def alpha_fusion(mri_norm: np.ndarray, pet_norm: np.ndarray,
                 alpha: float = 0.45) -> np.ndarray:
    """
    Alpha-blend MRI (gray) and PET (hot colormap) into an RGB image.
    mri_norm, pet_norm : 2D arrays normalised to [0,1]
    """
    mri_rgb = plt.cm.gray(mri_norm)[..., :3]           # (H,W,3)
    pet_rgb = plt.cm.hot(pet_norm)[..., :3]            # (H,W,3)
    fused   = (1 - alpha) * mri_rgb + alpha * pet_rgb  # (H,W,3)
    return np.clip(fused, 0, 1)


# ─────────────────────────────────────────────────────────────
# 4. ROTATING MIP ANIMATION
# ─────────────────────────────────────────────────────────────

def make_rotating_mip_gif(mri_vol:  np.ndarray,
                           pet_vol:  np.ndarray,
                           out_path: str,
                           n_angles: int = 36,
                           fps:      int = 10):
    """
    Create a GIF with three panels per frame (MRI MIP / PET MIP / fused),
    rotating through coronal-sagittal angles.
    """
    angles = np.linspace(0, 360, n_angles, endpoint=False)

    frames_gif = []
    print(f"\nRendering {n_angles} rotation angles …")

    mri_glo = (np.percentile(mri_vol, 1), np.percentile(mri_vol, 99))
    pet_glo = (np.percentile(pet_vol, 1), np.percentile(pet_vol, 99))

    for angle in tqdm(angles):
        # Rotate both volumes around Z (coronal↔sagittal sweep)
        mri_rot = rotate_volume(mri_vol, angle)
        pet_rot = rotate_volume(pet_vol, angle)

        # MIP along Y (coronal projection after rotation = varies view angle)
        mri_mip = compute_mip(mri_rot, axis=1)   # (Z, X)
        pet_mip = compute_mip(pet_rot, axis=1)   # (Z, X)

        mri_n = np.clip((mri_mip - mri_glo[0]) / (mri_glo[1] - mri_glo[0] + 1e-8), 0, 1)
        pet_n = np.clip((pet_mip - pet_glo[0]) / (pet_glo[1] - pet_glo[0] + 1e-8), 0, 1)
        fused = alpha_fusion(mri_n, pet_n, alpha=0.5)

        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
        fig.patch.set_facecolor("#0a0a0a")

        axes[0].imshow(np.flipud(mri_n),  cmap="gray",  origin="upper")
        axes[0].set_title("MRI (reference)", color="white", fontsize=10)

        axes[1].imshow(np.flipud(pet_n),  cmap="hot",   origin="upper")
        axes[1].set_title("PET (co-registered)", color="white", fontsize=10)

        axes[2].imshow(np.flipud(fused),               origin="upper")
        axes[2].set_title("Fusion (α-blend)", color="white", fontsize=10)

        for ax in axes:
            ax.set_facecolor("#0a0a0a")
            ax.axis("off")

        fig.suptitle(f"Rotating MIP — {angle:.0f}°", color="white", fontsize=11)
        plt.tight_layout(pad=0.5)

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
        frames_gif.append(img)
        plt.close()

    imageio.mimsave(out_path, frames_gif, fps=fps, loop=0)
    print(f"  Rotating MIP GIF saved → {out_path}")


# ─────────────────────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────────────────────

def main(mri_dir: str, out_dir: str):
    out_dir = Path(out_dir)
    (out_dir / "coregistration").mkdir(parents=True, exist_ok=True)
    (out_dir / "animations").mkdir(parents=True, exist_ok=True)

    # ── Load PET outputs from script 01 ──────────────────────
    print("\nLoading PET average from previous step …")
    pet_avg  = np.load(str(out_dir / "pet_avg.npy"))
    pet_meta = np.load(str(out_dir / "pet_meta.npy"), allow_pickle=True).item()
    ps  = pet_meta["pixel_spacing"]   # [row_mm, col_mm]
    st  = pet_meta["slice_thickness"]
    # SimpleITK spacing order: (col_mm, row_mm, slice_mm) = (X, Y, Z)
    pet_spacing = (ps[1], ps[0], st)

    print(f"  PET avg shape  : {pet_avg.shape}")
    print(f"  PET spacing    : {pet_spacing}")

    # ── Load MRI ──────────────────────────────────────────────
    print("\nLoading MRI DICOM …")
    mri_sitk = load_mri_dicom(mri_dir)

    # ── Coregister ────────────────────────────────────────────
    coreg_pet_sitk, tf = coregister_pet_to_mri(
        pet_avg, pet_spacing, mri_sitk, out_dir
    )

    # ── Convert back to numpy for visualisation ───────────────
    mri_vol      = sitk.GetArrayFromImage(mri_sitk).astype(np.float32)   # (Z,Y,X)
    coreg_pet_vol = sitk.GetArrayFromImage(coreg_pet_sitk).astype(np.float32)

    # Save co-registered PET for script 03
    np.save(str(out_dir / "pet_coreg.npy"), coreg_pet_vol)
    np.save(str(out_dir / "mri_vol.npy"),   mri_vol)

    print(f"\n  MRI volume shape        : {mri_vol.shape}")
    print(f"  Co-registered PET shape : {coreg_pet_vol.shape}")

    # ── Static coreg comparison (3 planes) ───────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.patch.set_facecolor("#0a0a0a")
    fig.suptitle("Coregistration — 3 orthogonal planes", color="white", fontsize=13)

    Z, Y, X = mri_vol.shape
    mri_n   = norm01(mri_vol)
    pet_n   = norm01(coreg_pet_vol)

    for col, (mip_fn, label) in enumerate([
        (lambda v: v[Z//2, :, :], "Axial"),
        (lambda v: v[:, Y//2, :], "Coronal"),
        (lambda v: v[:, :, X//2], "Sagittal"),
    ]):
        axes[0, col].imshow(mip_fn(mri_n), cmap="gray", origin="lower")
        axes[0, col].set_title(f"MRI {label}", color="white", fontsize=10)
        axes[0, col].axis("off")
        axes[0, col].set_facecolor("#0a0a0a")

        fused = alpha_fusion(mip_fn(mri_n), mip_fn(pet_n))
        axes[1, col].imshow(fused, origin="lower")
        axes[1, col].set_title(f"Fusion {label}", color="white", fontsize=10)
        axes[1, col].axis("off")
        axes[1, col].set_facecolor("#0a0a0a")

    plt.tight_layout()
    static_path = str(out_dir / "coregistration" / "coreg_static.png")
    plt.savefig(static_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Static comparison saved → {static_path}")

    # ── Rotating MIP animation ────────────────────────────────
    print("\nSTEP 2b — Rotating MIP animation")
    make_rotating_mip_gif(
        mri_vol, coreg_pet_vol,
        out_path=str(out_dir / "animations" / "rotating_mip.gif"),
        n_angles=36,
        fps=8,
    )

    print("\n✓ Script 02 complete.")
    print("  → Run python 03_segmentation.py next\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Rigid Coregistration PET→MRI")
    parser.add_argument("--mri_dir", default="data/mri",
                        help="Directory containing MRI DICOM files")
    parser.add_argument("--out_dir", default="outputs",
                        help="Root output directory (same as script 01)")
    args = parser.parse_args()
    main(args.mri_dir, args.out_dir)