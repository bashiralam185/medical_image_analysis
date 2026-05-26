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




# ─────────────────────────────────────────────────────────────────────────────
# Mutual Information computation (same metric used by SimpleITK coregistration)
# ─────────────────────────────────────────────────────────────────────────────

def compute_mi(vol_a, vol_b, bins=64):
    """
    Compute Normalised Mutual Information between two 3D volumes.
    NMI = (H(A) + H(B)) / H(A,B)  — range [1, 2], higher = better alignment.
    Also returns standard MI = H(A) + H(B) - H(A,B).
    Both volumes are flattened and must have the same shape.
    """
    assert vol_a.shape == vol_b.shape, "Volumes must have the same shape"
    a = vol_a.flatten().astype(np.float64)
    b = vol_b.flatten().astype(np.float64)

    # Normalise to [0, 1] for histogram stability
    a = (a - a.min()) / (a.max() - a.min() + 1e-10)
    b = (b - b.min()) / (b.max() - b.min() + 1e-10)

    # Joint histogram
    hist_2d, _, _ = np.histogram2d(a, b, bins=bins)
    hist_2d = hist_2d / hist_2d.sum()   # normalise to joint PDF

    # Marginal histograms
    p_a = hist_2d.sum(axis=1)
    p_b = hist_2d.sum(axis=0)

    # Entropies (ignore zero bins)
    h_a  = -np.sum(p_a[p_a > 0] * np.log2(p_a[p_a > 0]))
    h_b  = -np.sum(p_b[p_b > 0] * np.log2(p_b[p_b > 0]))
    h_ab = -np.sum(hist_2d[hist_2d > 0] * np.log2(hist_2d[hist_2d > 0]))

    mi  = h_a + h_b - h_ab
    nmi = (h_a + h_b) / (h_ab + 1e-10)   # Normalised MI
    return {"MI": round(mi, 4), "NMI": round(nmi, 4),
            "H_A": round(h_a, 4), "H_B": round(h_b, 4), "H_AB": round(h_ab, 4)}

def render():
    utils.try_restore_from_cache()
    utils.sidebar_status()

    st.title("📊 Analysis Dashboard")
    # st.markdown(
    #     "<p style='color:#5a7080;'>Quantitative metrics, time-activity curves, "
    #     "intensity histograms, and 3D visualisations — all in one place.</p>",
    #     unsafe_allow_html=True,
    # )

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
        "🔗 Coregistration Quality",
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
    # TAB 3 — Coregistration quality (NEW — inserted before old tab 2)
    # ══════════════════════════════════════════════════════════
    with tabs[2]:
        st.markdown("### Coregistration Quality Assessment")
        st.markdown(
            "Mutual Information (MI) measures how statistically dependent the two images are. "
            "A higher MI after registration vs. before means the alignment improved."
        )

        if not has_mri or not has_pet:
            st.info("Load both PET and MRI first.")
        else:
            mri_vol  = utils.ensure_3d(utils.get("mri_vol"))
            pet_avg  = utils.ensure_3d(utils.get("pet_avg"))
            pet_coreg= utils.ensure_3d(utils.get("pet_coreg")) if has_coreg else None

            # ── Resample PET avg to MRI space for fair before/after comparison ──
            # Before: crop/pad PET to MRI shape naively (no registration)
            def match_shape(src, ref_shape):
                """Crop or zero-pad src to match ref_shape."""
                out = np.zeros(ref_shape, dtype=src.dtype)
                slices_src = tuple(slice(0, min(s, r)) for s, r in zip(src.shape, ref_shape))
                slices_out = tuple(slice(0, min(s, r)) for s, r in zip(src.shape, ref_shape))
                out[slices_out] = src[slices_src]
                return out

            mri_shape = mri_vol.shape

            col1, col2 = st.columns(2)

            # ── Before registration ───────────────────────────────────────────
            with col1:
                st.markdown("#### Before registration")
                st.markdown(
                    "<div style='font-size:0.8rem; color:#5a7080;'>"
                    "PET average naively placed in MRI space (no alignment)</div>",
                    unsafe_allow_html=True
                )
                if st.button("Compute MI (before)", key="btn_mi_before"):
                    with st.spinner("Computing…"):
                        pet_unregistered = match_shape(pet_avg, mri_shape)
                        metrics_before = compute_mi(mri_vol, pet_unregistered)
                        st.session_state["mi_before_result"] = metrics_before

                if "mi_before_result" in st.session_state and isinstance(st.session_state["mi_before_result"], dict):
                    m = st.session_state["mi_before_result"]
                    st.metric("Mutual Information", m["MI"])
                    st.metric("Normalised MI (NMI)", m["NMI"])
                    with st.expander("Entropy breakdown"):
                        st.write(f"H(MRI) = {m['H_A']} bits")
                        st.write(f"H(PET) = {m['H_B']} bits")
                        st.write(f"H(MRI,PET) joint = {m['H_AB']} bits")

            # ── After registration ────────────────────────────────────────────
            with col2:
                st.markdown("#### After registration")
                st.markdown(
                    "<div style='font-size:0.8rem; color:#5a7080;'>"
                    "Co-registered PET resampled into MRI space</div>",
                    unsafe_allow_html=True
                )
                if not has_coreg:
                    st.info("Run coregistration first.")
                else:
                    if st.button("Compute MI (after)", key="btn_mi_after"):
                        with st.spinner("Computing…"):
                            metrics_after = compute_mi(mri_vol, pet_coreg)
                            st.session_state["mi_after_result"] = metrics_after

                    if "mi_after_result" in st.session_state and isinstance(st.session_state["mi_after_result"], dict):
                        m = st.session_state["mi_after_result"]
                        st.metric("Mutual Information", m["MI"])
                        st.metric("Normalised MI (NMI)", m["NMI"])
                        with st.expander("Entropy breakdown"):
                            st.write(f"H(MRI) = {m['H_A']} bits")
                            st.write(f"H(PET) = {m['H_B']} bits")
                            st.write(f"H(MRI,PET) joint = {m['H_AB']} bits")

            # ── Improvement summary ───────────────────────────────────────────
            if ("mi_before_result" in st.session_state and
                    "mi_after_result" in st.session_state and
                    isinstance(st.session_state["mi_before_result"], dict) and
                    isinstance(st.session_state["mi_after_result"], dict)):
                st.markdown("---")
                st.markdown("#### Improvement")
                mb = st.session_state["mi_before_result"]
                ma = st.session_state["mi_after_result"]
                mi_gain  = round(ma["MI"]  - mb["MI"],  4)
                nmi_gain = round(ma["NMI"] - mb["NMI"], 4)
                pct_gain = round((mi_gain / (mb["MI"] + 1e-10)) * 100, 1)

                c1, c2, c3 = st.columns(3)
                c1.metric("MI gain",     f"+{mi_gain}",  delta=str(mi_gain))
                c2.metric("NMI gain",    f"+{nmi_gain}", delta=str(nmi_gain))
                c3.metric("Improvement", f"{pct_gain}%",
                          delta=f"{pct_gain}% better" if pct_gain > 0 else "no improvement")

                # Bar chart comparison
                fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
                fig.patch.set_facecolor(utils.DARK_BG)
                for ax in axes:
                    ax.set_facecolor(utils.PANEL)
                    ax.tick_params(colors="#5a7080")
                    for sp in ax.spines.values():
                        sp.set_edgecolor("#1e2d42")

                # MI comparison
                axes[0].bar(["Before", "After"],
                            [mb["MI"], ma["MI"]],
                            color=["#ff6b35", "#00e5a0"],
                            width=0.5, alpha=0.85)
                axes[0].set_title("Mutual Information", color="#c8d8e8", fontsize=10)
                axes[0].set_ylabel("MI (bits)", color="#5a7080")

                # NMI comparison
                axes[1].bar(["Before", "After"],
                            [mb["NMI"], ma["NMI"]],
                            color=["#ff6b35", "#00e5a0"],
                            width=0.5, alpha=0.85)
                axes[1].set_title("Normalised MI", color="#c8d8e8", fontsize=10)
                axes[1].set_ylabel("NMI", color="#5a7080")
                axes[1].set_ylim(0, 2)

                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

                # Interpretation guide
                good = mi_gain > 0
                color = "#00e5a0" if good else "#ff6b35"
                msg   = (f"MI increased by {mi_gain:.4f} bits ({pct_gain}%) — "
                         "the registration improved alignment."
                         if good else
                         "MI did not improve — check alignment visually and consider re-running.")
                st.markdown(
                    f"<div style='background:#0e1520; border-left:3px solid {color}; "
                    f"padding:10px 14px; border-radius:4px; font-size:0.82rem; color:#c8d8e8;'>"
                    f"{msg}</div>",
                    unsafe_allow_html=True
                )

            # ── Joint histogram (visual MI check) ────────────────────────────
            st.markdown("---")
            st.markdown("#### Joint intensity histogram")
            st.markdown(
                "A well-registered pair shows tight, well-defined clusters in the joint histogram. "
                "A misaligned pair shows a diffuse, scattered pattern."
            )
            show_joint = st.selectbox(
                "Show joint histogram for:",
                ["Before registration (unaligned)", "After registration (co-registered)"]
                if has_coreg else ["Before registration (unaligned)"],
                key="joint_hist_sel"
            )

            if st.button("Generate joint histogram", key="gen_joint"):
                mri_flat = mri_vol.flatten().astype(np.float64)
                if "After" in show_joint and pet_coreg is not None:
                    pet_flat = pet_coreg.flatten().astype(np.float64)
                    title = "Joint histogram — After registration"
                else:
                    pet_flat = match_shape(pet_avg, mri_shape).flatten().astype(np.float64)
                    title = "Joint histogram — Before registration"

                # Percentile clip for cleaner vis
                mri_lo, mri_hi = np.percentile(mri_flat, 1), np.percentile(mri_flat, 99)
                pet_lo, pet_hi = np.percentile(pet_flat, 1), np.percentile(pet_flat, 99)
                mri_flat = np.clip(mri_flat, mri_lo, mri_hi)
                pet_flat = np.clip(pet_flat, pet_lo, pet_hi)

                hist_2d, xedges, yedges = np.histogram2d(mri_flat, pet_flat, bins=64)
                hist_2d = np.log1p(hist_2d)   # log scale for visibility

                fig2, ax2 = plt.subplots(figsize=(6, 5))
                fig2.patch.set_facecolor(utils.DARK_BG)
                ax2.set_facecolor(utils.PANEL)
                im = ax2.imshow(hist_2d.T, origin="lower", aspect="auto",
                                cmap="hot",
                                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
                ax2.set_xlabel("MRI intensity", color="#5a7080")
                ax2.set_ylabel("PET intensity", color="#5a7080")
                ax2.set_title(title, color="#c8d8e8", fontsize=10)
                ax2.tick_params(colors="#5a7080")
                for sp in ax2.spines.values():
                    sp.set_edgecolor("#1e2d42")
                plt.colorbar(im, ax=ax2, label="log(count)")
                plt.tight_layout()
                st.pyplot(fig2, use_container_width=True)
                plt.close()

    # ══════════════════════════════════════════════════════════
    # TAB 2 — Histograms (renumbered to tabs[1])
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
    with tabs[3]:
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
    with tabs[4]:
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
    with tabs[5]:
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