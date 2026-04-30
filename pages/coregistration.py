"""pages/coregistration.py — 3D Rigid coregistration PET → MRI"""
import io
import warnings

import imageio
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from scipy.ndimage import rotate as nd_rotate

import utils

warnings.filterwarnings("ignore")


def coregister(pet_avg, pet_spacing, mri_sitk):
    import SimpleITK as sitk

    def np2sitk(arr, spacing):
        """Convert numpy float32 array to a SimpleITK float32 image."""
        img = sitk.GetImageFromArray(arr.astype(np.float32))
        img.SetSpacing(tuple(float(s) for s in spacing))
        return img

    def sitk_to_float32(sitk_img):
        """
        Safely convert any SimpleITK image to float32 without using Cast or
        CopyInformation (both fail on int16 with certain SimpleITK builds).
        Manually transfers spacing, origin and direction instead.
        """
        arr = sitk.GetArrayFromImage(sitk_img).astype(np.float32)
        # Ensure exactly 3D — squeeze any trailing/leading size-1 dims
        arr = arr.squeeze()
        if arr.ndim == 2:
            arr = arr[np.newaxis]   # single-slice fallback
        out = sitk.GetImageFromArray(arr)
        # Manually copy spatial metadata — safer than CopyInformation
        spacing = sitk_img.GetSpacing()
        origin  = sitk_img.GetOrigin()
        direction = sitk_img.GetDirection()
        # Truncate to 3D if the source image had extra dims
        out.SetSpacing(spacing[:3])
        out.SetOrigin(origin[:3])
        if len(direction) == 9:
            out.SetDirection(direction)
        return out

    fixed  = sitk_to_float32(mri_sitk)
    moving = np2sitk(pet_avg, pet_spacing)   # already float32, no cast needed

    init_tf = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=64)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.15)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsGradientDescentLineSearch(
        learningRate=1.0, numberOfIterations=200,
        convergenceMinimumValue=1e-6, convergenceWindowSize=10,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(init_tf, inPlace=False)

    log = []
    progress_bar = st.progress(0, text="Registering…")
    total_iters  = 200

    def cb():
        it  = reg.GetOptimizerIteration()
        val = reg.GetMetricValue()
        log.append((it, val))
        progress_bar.progress(min(it / total_iters, 1.0),
                              text=f"iter {it}  MI={val:.5f}")

    reg.AddCommand(sitk.sitkIterationEvent, cb)
    final_tf = reg.Execute(fixed, moving)
    progress_bar.empty()

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(final_tf)
    coreg = resampler.Execute(moving)

    return sitk.GetArrayFromImage(coreg).astype(np.float32), log


def render():
    utils.try_restore_from_cache()
    utils.sidebar_status()

    st.title("🔗 Coregistration")
    st.markdown("<p style='color:#5a7080;'>3D rigid coregistration of PET average → MRI "
                "using Mattes Mutual Information.</p>", unsafe_allow_html=True)

    if not utils.has("pet_avg"):
        st.warning("⚠️ Load PET data first (📂 Load DICOM).")
        return
    if not utils.has("mri_vol"):
        st.warning("⚠️ Load MRI data first (📂 Load DICOM).")
        return

    pet_avg = utils.get("pet_avg")
    mri_vol = utils.get("mri_vol")
    meta    = utils.get("pet_meta") or {}
    ps  = meta.get("pixel_spacing", [1.0, 1.0])
    st_  = meta.get("slice_thickness", 1.0)
    pet_spacing = (ps[1], ps[0], st_)   # (X, Y, Z) for SimpleITK

    tabs = st.tabs(["⚙️ Run Registration", "🔍 Alignment Viewer", "🌀 Rotating MIP"])

    # ══════════════════════════════════════════════════════════
    # TAB 1 — Run registration
    # ══════════════════════════════════════════════════════════
    with tabs[0]:
        if utils.has("pet_coreg"):
            st.success("✓ Coregistration already computed. Results cached.")
            if st.button("Re-run coregistration"):
                utils.put("pet_coreg", None)
                st.rerun()
        else:
            st.info("Registration uses SimpleITK (Mattes MI, 3-level pyramid, ~1–3 min).")
            if st.button("▶  Run Coregistration", type="primary"):
                try:
                    import SimpleITK as sitk
                    reader = sitk.ImageSeriesReader()
                    dcm_names = reader.GetGDCMSeriesFileNames(str(utils.MRI_DIR))
                    if not dcm_names:
                        dcm_names = sorted([str(f) for f in utils.MRI_DIR.glob("**/*.dcm")])
                    reader.SetFileNames(dcm_names)
                    mri_sitk = reader.Execute()

                    with st.spinner("Running registration…"):
                        coreg_vol, log = coregister(pet_avg, pet_spacing, mri_sitk)
                    coreg_vol = utils.ensure_3d(coreg_vol)

                    utils.put("pet_coreg", coreg_vol)
                    utils.save_cache("pet_coreg", coreg_vol)

                    # Plot convergence
                    if log:
                        iters, vals = zip(*log)
                        fig, ax = plt.subplots(figsize=(10, 3))
                        fig.patch.set_facecolor(utils.DARK_BG)
                        ax.set_facecolor(utils.PANEL)
                        ax.plot(iters, vals, color="#00c8ff", linewidth=1.4)
                        ax.set_xlabel("Iteration", color="#5a7080")
                        ax.set_ylabel("Mattes MI (neg)", color="#5a7080")
                        ax.tick_params(colors="#5a7080")
                        for sp in ax.spines.values(): sp.set_edgecolor("#1e2d42")
                        ax.set_title("Registration convergence", color="#c8d8e8", fontsize=10)
                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                        plt.close()

                    st.success("✓ Coregistration complete!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    # ══════════════════════════════════════════════════════════
    # TAB 2 — Alignment viewer
    # ══════════════════════════════════════════════════════════
    with tabs[1]:
        if not utils.has("pet_coreg"):
            st.info("Run coregistration first.")
        else:
            st.markdown("### Alignment quality — orthogonal overlay")
            coreg = utils.get("pet_coreg")
            Z, Y, X = mri_vol.shape

            c1, c2, c3 = st.columns(3)
            z_sl = c1.slider("Z (axial)",  0, Z-1, Z//2, key="av_z")
            y_sl = c2.slider("Y (coronal)",0, Y-1, Y//2, key="av_y")
            x_sl = c3.slider("X (sagittal)",0,X-1, X//2, key="av_x")
            alpha = st.slider("Fusion alpha (PET weight)", 0.0, 1.0, 0.5, 0.05, key="av_alpha")

            mri_n  = utils.norm01(mri_vol)
            pet_n  = utils.norm01(coreg)

            fig, axes = plt.subplots(3, 3, figsize=(15, 15))
            fig.patch.set_facecolor(utils.DARK_BG)
            fig.suptitle("MRI | PET (co-reg) | Fusion", color="#00c8ff",
                         fontsize=12, fontfamily="monospace")

            view_planes = [
                (mri_n[z_sl],        pet_n[z_sl],        "Axial"),
                (mri_n[:, y_sl, :],  pet_n[:, y_sl, :],  "Coronal"),
                (mri_n[:, :, x_sl],  pet_n[:, :, x_sl],  "Sagittal"),
            ]
            for row, (mp, pp, lbl) in enumerate(view_planes):
                fused = utils.alpha_fusion(mp, pp, alpha=alpha)
                for col, (img, cmap, title) in enumerate([
                    (mp,    "gray", f"MRI — {lbl}"),
                    (pp,    "hot",  f"PET co-reg — {lbl}"),
                    (fused, None,   f"Fusion — {lbl}"),
                ]):
                    ax = axes[row, col]
                    ax.set_facecolor(utils.PANEL)
                    if cmap:
                        ax.imshow(img, cmap=cmap, origin="lower", aspect="auto")
                    else:
                        ax.imshow(img, origin="lower", aspect="auto")
                    ax.set_title(title, color="#c8d8e8", fontsize=9)
                    ax.axis("off")

            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

            # Download static figure
            buf = io.BytesIO()
            fig2, _ = plt.subplots(); plt.close(fig2)
            st.download_button("⬇️ Download alignment figure",
                               data=buf.getvalue(), file_name="alignment.png",
                               mime="image/png", disabled=True)

    # ══════════════════════════════════════════════════════════
    # TAB 3 — Rotating MIP
    # ══════════════════════════════════════════════════════════
    with tabs[2]:
        if not utils.has("pet_coreg"):
            st.info("Run coregistration first.")
        else:
            st.markdown("### Rotating Maximum Intensity Projection")
            c1, c2 = st.columns(2)
            n_angles = c1.slider("Number of angles", 12, 72, 36, 6, key="mip_angles")
            mip_fps  = c2.slider("GIF FPS", 4, 20, 8, key="mip_fps")
            alpha2   = st.slider("Fusion alpha", 0.0, 1.0, 0.5, 0.05, key="mip_alpha")

            if st.button("Generate Rotating MIP GIF", key="gen_mip"):
                coreg = utils.get("pet_coreg")
                angles = np.linspace(0, 360, n_angles, endpoint=False)

                mri_glo = (np.percentile(mri_vol, 1), np.percentile(mri_vol, 99))
                pet_glo = (np.percentile(coreg,   1), np.percentile(coreg,   99))

                gif_frames = []
                prog = st.progress(0)
                for i, angle in enumerate(angles):
                    mri_r = nd_rotate(mri_vol, angle, axes=(1,2), reshape=False, order=1, cval=0)
                    pet_r = nd_rotate(coreg,   angle, axes=(1,2), reshape=False, order=1, cval=0)
                    mri_m = mri_r.max(axis=1)
                    pet_m = pet_r.max(axis=1)
                    mn = np.clip((mri_m - mri_glo[0]) / (mri_glo[1] - mri_glo[0] + 1e-8), 0, 1)
                    pn = np.clip((pet_m - pet_glo[0]) / (pet_glo[1] - pet_glo[0] + 1e-8), 0, 1)
                    fused = utils.alpha_fusion(mn, pn, alpha=alpha2)

                    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
                    fig.patch.set_facecolor(utils.DARK_BG)
                    for ax, (img, cmap, lbl) in zip(axes, [
                        (np.flipud(mn),    "gray", "MRI"),
                        (np.flipud(pn),    "hot",  "PET co-reg"),
                        (np.flipud(fused), None,   "Fusion"),
                    ]):
                        ax.set_facecolor(utils.PANEL)
                        if cmap: ax.imshow(img, cmap=cmap, aspect="auto")
                        else:    ax.imshow(img, aspect="auto")
                        ax.set_title(lbl, color="#c8d8e8", fontsize=10)
                        ax.axis("off")
                    fig.suptitle(f"MIP rotation {angle:.0f}°", color="#00c8ff",
                                 fontsize=10, fontfamily="monospace")
                    plt.tight_layout()

                    buf2 = io.BytesIO()
                    fig.savefig(buf2, format="png", dpi=90, bbox_inches="tight",
                                facecolor=utils.DARK_BG)
                    buf2.seek(0)
                    img_arr = np.array(plt.imread(buf2))
                    if img_arr.max() <= 1.0:
                        img_arr = (img_arr * 255).astype(np.uint8)
                    gif_frames.append(img_arr[..., :3])
                    plt.close()
                    prog.progress((i+1)/n_angles)

                prog.empty()
                gif_buf = io.BytesIO()
                imageio.mimsave(gif_buf, gif_frames, format="GIF", fps=mip_fps, loop=0)
                gif_buf.seek(0)
                st.image(gif_buf.getvalue(), caption="Rotating MIP preview")
                st.download_button("⬇️ Download Rotating MIP GIF",
                                   data=gif_buf.getvalue(),
                                   file_name="rotating_mip.gif",
                                   mime="image/gif")