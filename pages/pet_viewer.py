"""pages/pet_viewer.py — Interactive dynamic PET exploration  [v2]"""
import io
import warnings

import imageio
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import utils

warnings.filterwarnings("ignore")


def render():
    utils.try_restore_from_cache()
    utils.sidebar_status()

    st.title("🖼️ PET Viewer")

    if not utils.has("pet_4d"):
        st.warning("⚠️ No PET data loaded. Go to **📂 Load DICOM** first.")
        return

    pet_4d = utils.ensure_4d(utils.get("pet_4d"))
    T, Z, Y, X = pet_4d.shape
    meta = utils.get("pet_meta") or {}

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Time frames", T)
    col2.metric("Slices (Z)",  Z)
    col3.metric("Rows × Cols", f"{Y}×{X}")
    col4.metric("Value range", f"{pet_4d.min():.0f} – {pet_4d.max():.0f}")

    st.markdown("---")

    tabs = st.tabs(["🎞️ Frame Explorer", "📐 Orthogonal Planes", "⏱️ Avg vs Last", "🎬 Export GIF"])

    # ══════════════════════════════════════════════════════════
    # TAB 1 — Frame Explorer
    # ══════════════════════════════════════════════════════════
    with tabs[0]:
        st.markdown("### Interactive frame & slice browser")
        c1, c2, c3 = st.columns(3)
        t_idx  = c1.slider("Time frame",  0, T-1, T-1, key="fe_t")
        z_idx  = c2.slider("Axial slice", 0, Z-1, Z//2, key="fe_z")
        cmap   = c3.selectbox("Colormap",
                    ["hot","inferno","viridis","plasma","gray","turbo"], key="fe_cmap")

        vmin_pct = st.slider("Min percentile (window)", 0, 50,  1, key="fe_vmin")
        vmax_pct = st.slider("Max percentile (window)", 50, 100, 99, key="fe_vmax")

        vol  = pet_4d[t_idx]
        vmin = np.percentile(pet_4d, vmin_pct)
        vmax = np.percentile(pet_4d, vmax_pct)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.patch.set_facecolor(utils.DARK_BG)
        for ax in axes:
            ax.set_facecolor(utils.PANEL)

        for ax, (plane, label) in zip(axes, [
            (vol[z_idx],         f"Axial z={z_idx}"),
            (vol[:, Y//2, :],    f"Coronal y={Y//2}"),
            (vol[:, :, X//2],    f"Sagittal x={X//2}"),
        ]):
            im = ax.imshow(plane, cmap=cmap, vmin=vmin, vmax=vmax,
                           origin="lower", aspect="auto")
            ax.set_title(label, color="#c8d8e8", fontsize=10)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.suptitle(f"Dynamic PET — frame {t_idx+1}/{T}",
                     color="#00c8ff", fontsize=12, fontfamily="monospace")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown("#### Signal at current voxel across time")
        signal = pet_4d[:, z_idx, Y//2, X//2]
        fig2, ax2 = plt.subplots(figsize=(10, 2.5))
        fig2.patch.set_facecolor(utils.DARK_BG)
        ax2.set_facecolor(utils.PANEL)
        ax2.plot(range(T), signal, color="#00c8ff", linewidth=1.5,
                 marker="o", markersize=3)
        ax2.axvline(t_idx, color="#ff6b35", linewidth=1.2,
                    linestyle="--", label=f"frame {t_idx}")
        ax2.set_xlabel("Time frame", color="#5a7080")
        ax2.set_ylabel("Intensity",  color="#5a7080")
        ax2.tick_params(colors="#5a7080")
        for sp in ax2.spines.values():
            sp.set_edgecolor("#1e2d42")
        ax2.legend(fontsize=8)
        ax2.set_title(f"Time-activity at voxel ({z_idx}, {Y//2}, {X//2})",
                      color="#c8d8e8", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close()

    # ══════════════════════════════════════════════════════════
    # TAB 2 — Orthogonal planes navigator
    # ══════════════════════════════════════════════════════════
    with tabs[1]:
        st.markdown("### Navigate all three planes simultaneously")
        c1, c2, c3, c4 = st.columns(4)
        t2    = c1.slider("Time frame",   0, T-1, T-1,  key="op_t")
        z2    = c2.slider("Z (axial)",    0, Z-1, Z//2, key="op_z")
        y2    = c3.slider("Y (coronal)",  0, Y-1, Y//2, key="op_y")
        x2    = c4.slider("X (sagittal)", 0, X-1, X//2, key="op_x")
        cmap2 = st.selectbox("Colormap",
                    ["hot","inferno","viridis","gray","plasma","turbo"], key="op_cmap")

        vol2  = pet_4d[t2]
        vmin2, vmax2 = np.percentile(vol2, 1), np.percentile(vol2, 99)

        fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
        fig3.patch.set_facecolor(utils.DARK_BG)

        for ax, (plane, label, crosshair) in zip(axes3, [
            (vol2[z2],        f"Axial z={z2}",    (y2, x2)),
            (vol2[:, y2, :],  f"Coronal y={y2}",  (z2, x2)),
            (vol2[:, :, x2],  f"Sagittal x={x2}", (z2, y2)),
        ]):
            ax.set_facecolor(utils.PANEL)
            ax.imshow(plane, cmap=cmap2, vmin=vmin2, vmax=vmax2,
                      origin="lower", aspect="auto")
            cy_ch, cx_ch = crosshair
            ax.axhline(cy_ch, color="#00c8ff", linewidth=0.6, alpha=0.7)
            ax.axvline(cx_ch, color="#00c8ff", linewidth=0.6, alpha=0.7)
            ax.set_title(label, color="#c8d8e8", fontsize=10)
            ax.axis("off")

        fig3.suptitle(f"PET frame {t2+1}/{T} — crosshair at ({z2},{y2},{x2})",
                      color="#00c8ff", fontsize=11, fontfamily="monospace")
        plt.tight_layout()
        st.pyplot(fig3, use_container_width=True)
        plt.close()

    # ══════════════════════════════════════════════════════════
    # TAB 3 — Average vs Last frame
    # ══════════════════════════════════════════════════════════
    with tabs[2]:
        st.markdown("### Temporal average vs. last frame comparison")

        # ✅ Safe: explicit has() check instead of `or` on a numpy array
        pet_avg  = utils.ensure_3d(utils.get("pet_avg"))  \
                   if utils.has("pet_avg")  else utils.ensure_3d(pet_4d.mean(axis=0))
        pet_last = utils.ensure_3d(utils.get("pet_last")) \
                   if utils.has("pet_last") else utils.ensure_3d(pet_4d[-1])

        c1, c2 = st.columns(2)
        z3    = c1.slider("Axial slice", 0, Z-1, Z//2, key="al_z")
        cmap3 = c2.selectbox("Colormap",
                    ["hot","inferno","viridis","gray"], key="al_cmap")

        vmin3 = np.percentile(pet_4d, 1)
        vmax3 = np.percentile(pet_4d, 99)

        fig4, axes4 = plt.subplots(2, 3, figsize=(15, 10))
        fig4.patch.set_facecolor(utils.DARK_BG)
        fig4.suptitle("Temporal Average (top) vs. Last Frame (bottom)",
                      color="#00c8ff", fontsize=12, fontfamily="monospace")

        for row, (vol_r, label_r) in enumerate([
            (pet_avg,  "Average"),
            (pet_last, "Last frame"),
        ]):
            for col, (plane, lbl) in enumerate([
                (vol_r[z3],          f"{label_r} — Axial z={z3}"),
                (vol_r[:, Y//2, :],  f"{label_r} — Coronal"),
                (vol_r[:, :, X//2],  f"{label_r} — Sagittal"),
            ]):
                ax = axes4[row, col]
                ax.set_facecolor(utils.PANEL)
                im = ax.imshow(plane, cmap=cmap3, vmin=vmin3, vmax=vmax3,
                               origin="lower", aspect="auto")
                ax.set_title(lbl, color="#c8d8e8", fontsize=9)
                ax.axis("off")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        st.pyplot(fig4, use_container_width=True)
        plt.close()

        st.markdown("#### Difference map (Last − Average)")
        diff = pet_last - pet_avg
        fig5, axes5 = plt.subplots(1, 3, figsize=(15, 4))
        fig5.patch.set_facecolor(utils.DARK_BG)
        vlim = np.percentile(np.abs(diff), 98)

        for ax, (plane, lbl) in zip(axes5, [
            (diff[z3],          f"Axial z={z3}"),
            (diff[:, Y//2, :],  "Coronal"),
            (diff[:, :, X//2],  "Sagittal"),
        ]):
            ax.set_facecolor(utils.PANEL)
            im = ax.imshow(plane, cmap="RdBu_r", vmin=-vlim, vmax=vlim,
                           origin="lower", aspect="auto")
            ax.set_title(lbl, color="#c8d8e8", fontsize=9)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        st.pyplot(fig5, use_container_width=True)
        plt.close()

    # ══════════════════════════════════════════════════════════
    # TAB 4 — Export GIF
    # ══════════════════════════════════════════════════════════
    with tabs[3]:
        st.markdown("### Export animated GIF")

        c1, c2, c3 = st.columns(3)
        gif_fps   = c1.slider("FPS", 1, 15, 4, key="gif_fps")
        gif_plane = c2.selectbox("Plane",
                        ["Axial","Coronal","Sagittal","All 3"], key="gif_plane")
        gif_cmap  = c3.selectbox("Colormap",
                        ["hot","inferno","viridis","gray","plasma"], key="gif_cmap")

        if st.button("Generate GIF", key="gen_gif"):
            with st.spinner("Rendering frames…"):
                vmin_g = np.percentile(pet_4d, 1)
                vmax_g = np.percentile(pet_4d, 99)
                gif_frames = []

                for t in range(T):
                    vol = pet_4d[t]
                    if gif_plane == "All 3":
                        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                        fig.patch.set_facecolor(utils.DARK_BG)
                        for ax, (plane, lbl) in zip(axes, [
                            (vol[Z//2],        "Axial"),
                            (vol[:, Y//2, :],  "Coronal"),
                            (vol[:, :, X//2],  "Sagittal"),
                        ]):
                            ax.set_facecolor(utils.PANEL)
                            ax.imshow(plane, cmap=gif_cmap,
                                      vmin=vmin_g, vmax=vmax_g,
                                      origin="lower", aspect="auto")
                            ax.set_title(lbl, color="#c8d8e8", fontsize=9)
                            ax.axis("off")
                    else:
                        plane_map = {
                            "Axial":    vol[Z//2],
                            "Coronal":  vol[:, Y//2, :],
                            "Sagittal": vol[:, :, X//2],
                        }
                        fig, ax = plt.subplots(figsize=(5, 5))
                        fig.patch.set_facecolor(utils.DARK_BG)
                        ax.set_facecolor(utils.PANEL)
                        ax.imshow(plane_map[gif_plane], cmap=gif_cmap,
                                  vmin=vmin_g, vmax=vmax_g,
                                  origin="lower", aspect="auto")
                        ax.axis("off")

                    fig.suptitle(f"PET frame {t+1}/{T}",
                                 color="#00c8ff", fontsize=10,
                                 fontfamily="monospace")
                    plt.tight_layout()
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=100,
                                bbox_inches="tight",
                                facecolor=fig.get_facecolor())
                    buf.seek(0)
                    img_arr = np.array(plt.imread(buf))
                    if img_arr.max() <= 1.0:
                        img_arr = (img_arr * 255).astype(np.uint8)
                    gif_frames.append(img_arr[..., :3])
                    plt.close()

                gif_buf = io.BytesIO()
                imageio.mimsave(gif_buf, gif_frames,
                                format="GIF", fps=gif_fps, loop=0)
                gif_buf.seek(0)

                st.image(gif_buf.getvalue(),
                         caption="Preview", use_container_width=True)
                st.download_button(
                    "⬇️  Download GIF",
                    data=gif_buf.getvalue(),
                    file_name="pet_animation.gif",
                    mime="image/gif",
                )