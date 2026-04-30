"""
utils.py — Shared state management, loaders, and plot helpers
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

warnings.filterwarnings("ignore")

# ── Project paths ─────────────────────────────────────────────────────────────
DATA_DIR    = Path("data")
PET_DIR     = DATA_DIR / "pet"
MRI_DIR     = DATA_DIR / "mri"
OUTPUTS_DIR = Path("outputs")
CACHE_DIR   = OUTPUTS_DIR / ".cache"

for d in [PET_DIR, MRI_DIR, OUTPUTS_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Session state keys ────────────────────────────────────────────────────────
KEYS = {
    "pet_4d":       "pet_4d",
    "pet_avg":      "pet_avg",
    "pet_last":     "pet_last",
    "pet_meta":     "pet_meta",
    "mri_vol":      "mri_vol",
    "pet_coreg":    "pet_coreg",
    "tumor_mask":   "tumor_mask",
    "tumor_center": "tumor_center",
}


def get(key):
    return st.session_state.get(KEYS[key])


def put(key, value):
    st.session_state[KEYS[key]] = value


def has(key):
    return KEYS[key] in st.session_state and st.session_state[KEYS[key]] is not None


# ── Shape sanitization ────────────────────────────────────────────────────────
def ensure_3d(arr):
    """
    Squeeze a volume to exactly 3 dimensions (Z, Y, X).
    Handles shapes like (Z,Y,X,1), (1,Z,Y,X), or plain (Z,Y,X).
    """
    if arr is None:
        return None
    arr = np.squeeze(arr)
    if arr.ndim == 3:
        return arr
    if arr.ndim == 4:
        if arr.shape[-1] <= 4:
            return arr.mean(axis=-1)
        if arr.shape[0] <= 4:
            return arr.mean(axis=0)
    if arr.ndim == 2:
        return arr[np.newaxis]
    raise ValueError("Cannot reduce array of shape {} to 3D".format(arr.shape))


def ensure_4d(arr):
    """Ensure a PET array is exactly 4 dimensions (T, Z, Y, X)."""
    if arr is None:
        return None
    arr = np.squeeze(arr)
    if arr.ndim == 4:
        return arr
    if arr.ndim == 3:
        return arr[np.newaxis]
    raise ValueError("Cannot reduce array of shape {} to 4D".format(arr.shape))


# ── Disk-backed cache ─────────────────────────────────────────────────────────
def save_cache(key, arr):
    np.save(str(CACHE_DIR / "{}.npy".format(key)), arr)


def load_cache(key):
    p = CACHE_DIR / "{}.npy".format(key)
    if p.exists():
        return np.load(str(p), allow_pickle=True)
    return None


def try_restore_from_cache():
    """Auto-load previously computed arrays from disk into session state."""
    for key in KEYS:
        if not has(key):
            arr = load_cache(key)
            if arr is None:
                continue
            if arr.ndim == 0 or arr.dtype == object:
                put(key, arr.item())
            else:
                if key in ("mri_vol", "pet_avg", "pet_last", "pet_coreg"):
                    arr = ensure_3d(arr)
                elif key == "pet_4d":
                    arr = ensure_4d(arr)
                elif key == "tumor_mask":
                    arr = ensure_3d(arr).astype(bool)
                put(key, arr)


# ── Plotting helpers ──────────────────────────────────────────────────────────
DARK_BG = "#080c10"
PANEL   = "#0e1520"


def norm01(arr, plo=1, phi=99):
    lo, hi = np.percentile(arr, plo), np.percentile(arr, phi)
    return np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1)


def show_plane(ax, plane, cmap="hot", label="", vmin=None, vmax=None):
    ax.imshow(plane, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(label, color="#c8d8e8", fontsize=9, pad=4)
    ax.axis("off")


def alpha_fusion(mri_n, pet_n, alpha=0.5):
    mri_rgb = plt.cm.gray(mri_n)[..., :3]
    pet_rgb = plt.cm.hot(pet_n)[..., :3]
    return np.clip((1 - alpha) * mri_rgb + alpha * pet_rgb, 0, 1)


def status_badge(label, ready):
    color  = "#00e5a0" if ready else "#ff6b35"
    symbol = "u25cf" if ready else "u25cb"
    st.markdown(
        "<span style='font-family:Space Mono,monospace; font-size:0.75rem; "
        "color:{};'>{} {}</span>".format(color, symbol, label),
        unsafe_allow_html=True,
    )


def sidebar_status():
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<div style='font-size:0.7rem; color:#5a7080; letter-spacing:1px;'>DATA STATUS</div>",
        unsafe_allow_html=True,
    )
    status_badge("PET loaded",       has("pet_4d"))
    status_badge("MRI loaded",       has("mri_vol"))
    status_badge("Coregistered",     has("pet_coreg"))
    status_badge("Tumour segmented", has("tumor_mask"))