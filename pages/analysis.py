"""pages/analysis.py — Integrated analysis dashboard"""
import io
import warnings

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import streamlit as st
from skimage import measure

import utils

warnings.filterwarnings("ignore")


def render():
    utils.try_restore_from_cache()
    utils.sidebar_status()

    st.title("📊 Analysis Dashboard")
    st.markdown(
        "<p style='color:#5a7080;'>Quantitative metrics, time-activity curves, "
        "intensity histograms, and 3D visualisations — all in one place.</p>",
        unsafe_allow_html=True,
    )

    has_pet  = utils.has("pet_4d")
    has_mri  = utils.has("mri_vol")
    has_seg  = utils.has("tumor_mask")
    has_coreg= utils.has("pet_coreg")

    if not has_pet and not has_mri:
        st.warning("⚠️ No data loaded yet. Go to **📂 Load DICOM** first.")
        return

    tabs = st.tabs([
        "⏱️ Time-Activity Curve",
        "📈 Intensity Histograms",
        "🗺️ Multi-modal Overview",
        "🧊 3D Surface",
        "📋 Summary Report",
    ])

    # ══════════════════════════════════════════════════════════
    # TAB 1 — Time-activity curves
    # ══════════════════════════════════════════════════════════
    with tabs[0]:
        if not has_pet:
            st.info("Load PET data first.")
        else:
            pet_4d = utils.get("pet_4d")
            meta   = utils.get("pet_meta") or {}
            T, Z, Y, X = pet_4d.shape

            st.markdown("### Time-Activity Curve (TAC)")
            st.markdown(
                "Select a VOI (Volume of Interest) to extract the mean PET signal "
                "over time. Compare tumour vs. background regions."
            )

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**VOI 1 — Tumour / ROI**")
                z1 = st.slider("Z",  0, Z-1, Z//2,  key="tac_z1")
                y1 = st.slider("Y",  0, Y-1, Y//2,  key="tac_y1")
                x1 = st.slider("X",  0, X-1, X//2,  key="tac_x1")
                r1 = st.slider("Radius (vox)", 1, 15, 5, key="tac_r1")
            with c2:
                st.markdown("**VOI 2 — Background / Contralateral**")
                z2 = st.slider("Z",  0, Z-1, Z//2,  key="tac_z2")
                y2 = st.slider("Y",  0, Y-1, Y//2,  key="tac_y2")
                x2 = st.slider("X",  0, X-1, X//2+30, key="tac_x2")
                r2 = st.slider("Radius (vox)", 1, 15, 5, key="tac_r2")

            def sphere_mask(shape, center, radius):
                Z_, Y_, X_ = shape
                cz, cy, cx = center
                z_, y_, x_ = np.ogrid[:Z_, :Y_, :X_]
                return ((z_-cz)**2 + (y_-cy)**2 + (x_-cx)**2) <= radius**2

            mask1 = sphere_mask(pet_4d.shape[1:], (z1,y1,x1), r1)
            mask2 = sphere_mask(pet_4d.shape[1:], (z2,y2,x2), r2)

            tac1 = [pet_4d[t][mask1].mean() for t in range(T)]
            tac2 = [pet_4d[t][mask2].mean() for t in range(T)]

            # Time axis
            if meta.get("frame_start_times"):
                times = np.array(meta["frame_start_times"]) / 1000.0   # → seconds
            else:
                times = np.arange(T)
            xlabel = "Time (s)" if meta.get("frame_start_times") else "Frame index"

            fig, ax = plt.subplots(figsize=(11, 4))
            fig.patch.set_facecolor(utils.DARK_BG)
            ax.set_facecolor(utils.PANEL)
            ax.plot(times, tac1, color="#ff6b35", linewidth=2, marker="o",
                    markersize=4, label=f"VOI 1 ({z1},{y1},{x1}) r={r1}")
            ax.plot(times, tac2, color="#00c8ff", linewidth=2, marker="s",
                    markersize=4, label=f"VOI 2 ({z2},{y2},{x2}) r={r2}")
            ax.set_xlabel(xlabel, color="#5a7080")
            ax.set_ylabel("Mean PET intensity", color="#5a7080")
            ax.tick_params(colors="#5a7080")
            for sp in ax.spines.values(): sp.set_edgecolor("#1e2d42")
            ax.legend(fontsize=9, facecolor="#0e1520", labelcolor="#c8d8e8")
            ax.set_title("Time-Activity Curves", color="#c8d8e8", fontsize=11)
            ax.grid(alpha=0.15)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

            # Tumour TAC if mask exists
            if has_seg:
                tumor_mask = utils.get("tumor_mask")
                if tumor_mask.shape == pet_4d.shape[1:]:
                    tac_tumor = [pet_4d[t][tumor_mask].mean() for t in range(T)]
                    fig_t, ax_t = plt.subplots(figsize=(11, 3.5))
                    fig_t.patch.set_facecolor(utils.DARK_BG)
                    ax_t.set_facecolor(utils.PANEL)
                    ax_t.plot(times, tac_tumor, color="#00e5a0", linewidth=2,
                              marker="D", markersize=4, label="Tumour mask mean")
                    ax_t.set_xlabel(xlabel, color="#5a7080")
                    ax_t.set_ylabel("Mean SUV / intensity", color="#5a7080")
                    ax_t.tick_params(colors="#5a7080")
                    for sp in ax_t.spines.values(): sp.set_edgecolor("#1e2d42")
                    ax_t.legend(fontsize=9, facecolor="#0e1520", labelcolor="#c8d8e8")
                    ax_t.set_title("Tumour Segmentation TAC", color="#c8d8e8", fontsize=10)
                    ax_t.grid(alpha=0.15)
                    plt.tight_layout()
                    st.pyplot(fig_t, use_container_width=True)
                    plt.close()
                else:
                    st.info("Tumour mask shape doesn't match PET shape (different space). "
                            "TAC from mask skipped.")

    # ══════════════════════════════════════════════════════════
    # TAB 2 — Histograms
    # ══════════════════════════════════════════════════════════
    with tabs[1]:
        st.markdown("### Intensity Histograms")
        n_bins = st.slider("Number of bins", 20, 200, 80, key="hist_bins")

        fig, axes = plt.subplots(1, 2 if has_mri else 1, figsize=(14, 4))
        fig.patch.set_facecolor(utils.DARK_BG)
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        if has_pet:
            pet_4d = utils.get("pet_4d")
            ax = axes[0]
            ax.set_facecolor(utils.PANEL)
            flat = pet_4d.flatten()
            lo, hi = np.percentile(flat, 1), np.percentile(flat, 99.5)
            ax.hist(flat[(flat >= lo) & (flat <= hi)], bins=n_bins,
                    color="#ff6b35", alpha=0.85, edgecolor="none", density=True)

            if has_seg:
                tumor_mask = utils.get("tumor_mask")
                pet_last   = utils.get("pet_last")
                if pet_last is not None and tumor_mask.shape == pet_last.shape:
                    t_vals = pet_last[tumor_mask]
                    ax.hist(t_vals, bins=n_bins, color="#00e5a0", alpha=0.7,
                            edgecolor="none", density=True, label="Tumour voxels")
                    ax.legend(fontsize=8, facecolor="#0e1520", labelcolor="#c8d8e8")

            ax.set_xlabel("PET intensity", color="#5a7080")
            ax.set_ylabel("Density", color="#5a7080")
            ax.tick_params(colors="#5a7080")
            for sp in ax.spines.values(): sp.set_edgecolor("#1e2d42")
            ax.set_title("PET intensity distribution", color="#c8d8e8", fontsize=10)

        if has_mri and len(axes) > 1:
            mri_vol = utils.get("mri_vol")
            ax = axes[1]
            ax.set_facecolor(utils.PANEL)
            flat = mri_vol.flatten()
            lo, hi = np.percentile(flat, 2), np.percentile(flat, 99)
            ax.hist(flat[(flat >= lo) & (flat <= hi)], bins=n_bins,
                    color="#00c8ff", alpha=0.85, edgecolor="none", density=True)

            if has_seg:
                tumor_mask = utils.get("tumor_mask")
                if tumor_mask.shape == mri_vol.shape:
                    t_vals = mri_vol[tumor_mask]
                    ax.hist(t_vals, bins=n_bins, color="#ffb547", alpha=0.7,
                            edgecolor="none", density=True, label="Tumour voxels")
                    ax.legend(fontsize=8, facecolor="#0e1520", labelcolor="#c8d8e8")

            ax.set_xlabel("MRI intensity", color="#5a7080")
            ax.set_ylabel("Density", color="#5a7080")
            ax.tick_params(colors="#5a7080")
            for sp in ax.spines.values(): sp.set_edgecolor("#1e2d42")
            ax.set_title("MRI T1 intensity distribution", color="#c8d8e8", fontsize=10)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ══════════════════════════════════════════════════════════
    # TAB 3 — Multi-modal overview
    # ══════════════════════════════════════════════════════════
    with tabs[2]:
        st.markdown("### Multi-modal overview")
        if not (has_mri and has_pet):
            st.info("Load both PET and MRI to see the overlay.")
        else:
            mri_vol  = utils.get("mri_vol")
            pet_last = utils.get("pet_last")
            coreg    = utils.get("pet_coreg") if has_coreg else None
            Z, Y, X  = mri_vol.shape

            c1, c2 = st.columns(2)
            z_ov = c1.slider("Axial slice", 0, Z-1, Z//2, key="ov_z")
            a_ov = c2.slider("Fusion alpha", 0.0, 1.0, 0.5, 0.05, key="ov_alpha")

            pet_display = utils.norm01(coreg[z_ov]) if coreg is not None else utils.norm01(pet_last[min(z_ov, pet_last.shape[0]-1)])
            mri_display = utils.norm01(mri_vol[z_ov])
            fused       = utils.alpha_fusion(mri_display, pet_display, alpha=a_ov)

            n_panels = 4 if has_seg else 3
            fig, axes = plt.subplots(1, n_panels, figsize=(5*n_panels, 5))
            fig.patch.set_facecolor(utils.DARK_BG)
            fig.suptitle(f"Axial z={z_ov}", color="#00c8ff", fontfamily="monospace")

            panels = [
                (mri_display,  "gray", "MRI T1"),
                (pet_display,  "hot",  "PET" + (" co-reg" if coreg is not None else " last")),
                (fused,        None,   "Fusion"),
            ]
            if has_seg:
                tumor_mask = utils.get("tumor_mask")
                seg_panel = np.zeros((*mri_display.shape, 4), dtype=np.float32)
                seg_panel[tumor_mask[z_ov]] = [1, 0.2, 0.2, 0.6]
                panels.append((seg_panel, None, "Tumour mask"))

            for ax, (img, cmap, lbl) in zip(axes, panels):
                ax.set_facecolor(utils.PANEL)
                if cmap:
                    ax.imshow(img, cmap=cmap, origin="lower", aspect="auto")
                else:
                    ax.imshow(img, origin="lower", aspect="auto")
                ax.set_title(lbl, color="#c8d8e8", fontsize=9)
                ax.axis("off")

            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

    # ══════════════════════════════════════════════════════════
    # TAB 4 — 3D Surface
    # ══════════════════════════════════════════════════════════
    with tabs[3]:
        st.markdown("### 3D Surface Render")
        if not has_seg:
            st.info("Run segmentation first to see the 3D tumour surface.")
        else:
            tumor_mask = utils.get("tumor_mask")
            try:
                verts, faces, _, _ = measure.marching_cubes(
                    tumor_mask.astype(np.float32), level=0.5)
                st.markdown(f"Surface mesh: **{len(verts):,} vertices**, **{len(faces):,} faces**")

                elev = st.slider("Elevation", -90, 90, 25, key="surf_elev")
                azim = st.slider("Azimuth",   0,  360, 45, key="surf_azim")
                surf_color = st.color_picker("Surface colour", "#ff6b35", key="surf_col")

                fig3d = plt.figure(figsize=(8, 8))
                ax3d  = fig3d.add_subplot(111, projection="3d")
                ax3d.plot_trisurf(verts[:, 2], verts[:, 1], verts[:, 0],
                                  triangles=faces, color=surf_color,
                                  alpha=0.75, linewidth=0, antialiased=True)
                ax3d.set_xlabel("X", color="#5a7080")
                ax3d.set_ylabel("Y", color="#5a7080")
                ax3d.set_zlabel("Z", color="#5a7080")
                ax3d.view_init(elev=elev, azim=azim)
                ax3d.set_facecolor("#0e1520")
                ax3d.tick_params(colors="#5a7080")
                fig3d.patch.set_facecolor(utils.DARK_BG)
                ax3d.set_title("3D Tumour Surface", color="#c8d8e8", fontsize=11)
                plt.tight_layout()
                st.pyplot(fig3d, use_container_width=True)
                plt.close()
            except Exception as e:
                st.error(f"Could not compute surface: {e}")

    # ══════════════════════════════════════════════════════════
    # TAB 5 — Summary report
    # ══════════════════════════════════════════════════════════
    with tabs[4]:
        st.markdown("### Summary Report")

        report_lines = ["# PET/MRI Analysis Report\n"]

        if has_pet:
            meta = utils.get("pet_meta") or {}
            pet_4d = utils.get("pet_4d")
            T, Z, Y, X = pet_4d.shape
            report_lines += [
                "## PET Study",
                f"- Shape: {pet_4d.shape} (frames × slices × rows × cols)",
                f"- Time frames: {T}",
                f"- Pixel spacing: {meta.get('pixel_spacing')} mm",
                f"- Slice thickness: {meta.get('slice_thickness')} mm",
                f"- Intensity range: {pet_4d.min():.1f} – {pet_4d.max():.1f}",
                "",
            ]

        if has_mri:
            mri_vol = utils.get("mri_vol")
            report_lines += [
                "## MRI Study",
                f"- Shape: {mri_vol.shape} (Z × Y × X)",
                f"- Intensity range: {mri_vol.min():.1f} – {mri_vol.max():.1f}",
                "",
            ]

        if has_coreg:
            report_lines += ["## Coregistration", "- Method: 3D Rigid (Euler3D)", 
                             "- Metric: Mattes Mutual Information (64 bins)",
                             "- Optimizer: Gradient Descent w/ line search",
                             "- Multi-resolution: 3 levels (σ = 2,1,0 mm)", ""]

        if has_seg:
            tumor_mask = utils.get("tumor_mask")
            mri_vol    = utils.get("mri_vol") if has_mri else None
            voxels     = int(tumor_mask.sum())
            vol_cm3    = voxels / 1000.0
            coords     = np.argwhere(tumor_mask)
            if len(coords):
                mask_ctr = coords.mean(axis=0)
                ext      = coords.max(axis=0) - coords.min(axis=0) + 1
            else:
                mask_ctr = [0,0,0]; ext = [0,0,0]

            report_lines += [
                "## Tumour Segmentation",
                f"- Voxel count: {voxels:,}",
                f"- Estimated volume: {vol_cm3:.2f} cm³",
                f"- Centroid: ({mask_ctr[0]:.1f}, {mask_ctr[1]:.1f}, {mask_ctr[2]:.1f})",
                f"- Bounding box extent: Z={ext[0]}, Y={ext[1]}, X={ext[2]} voxels",

            ]

        report_text = "\n".join(report_lines)
        st.markdown(report_text)

        st.download_button(
            "⬇️ Download Report (.md)",
            data=report_text,
            file_name="pet_mri_analysis_report.md",
            mime="text/markdown",
        )
