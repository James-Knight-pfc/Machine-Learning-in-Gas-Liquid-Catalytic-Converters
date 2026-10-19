"""
Shared segmentation code used by the HSV tuner (and later the batch analysis).

Pixel classes inside the channel:
  pillar  - from pillar_mask.png (reactor structure, never counted)
  fluid   - blue, selected by the HSV threshold
  bubble  - everything else in the open (non-pillar) channel area,
            i.e. the gas plus its dark film edges
"""
import glob
import json
import os

import cv2
import numpy as np

# ---- folder layout (all paths relative to this file, so scripts work from any directory)
#   Image Recognition/
#     Videos/Raw/                    raw videos (input to process_videos.py)
#     Videos/<name>_every10/         processed frames + video_info.json
#     Results/                       area CSVs
#     hsv_settings.json, pillar_mask.png   shared by every video
HERE          = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR    = os.path.join(HERE, "Videos")
RESULTS_DIR   = os.path.join(HERE, "Results")
SETTINGS_FILE = os.path.join(HERE, "hsv_settings.json")
PILLAR_FILE   = os.path.join(HERE, "pillar_mask.png")


def _frame_key(path):
    digits = "".join(c for c in os.path.basename(path) if c.isdigit())
    return int(digits) if digits else 0


def list_frames(video_dir):
    """Sorted list of frame_*.png paths in a processed video folder."""
    return sorted(glob.glob(os.path.join(video_dir, "frame_*.png")), key=_frame_key)


def list_videos():
    """Processed video folders in Videos/ (any sub-folder containing frame_*.png), sorted by name."""
    if not os.path.isdir(VIDEOS_DIR):
        return []
    out = []
    for name in sorted(os.listdir(VIDEOS_DIR)):
        d = os.path.join(VIDEOS_DIR, name)
        if name != "Raw" and os.path.isdir(d) and glob.glob(os.path.join(d, "frame_*.png")):
            out.append(d)
    return out


def video_label(video_dir):
    """Short name, e.g. foam2d_w10_n02__20191109_182042"""
    name = os.path.basename(os.path.normpath(video_dir))
    return name.rsplit("_every", 1)[0] if "_every" in name else name


def video_info(video_dir):
    """fps / step for a processed video (falls back to 30 fps, every 10th frame)."""
    info = {"fps": 30.0, "step": 10}
    p = os.path.join(video_dir, "video_info.json")
    if os.path.exists(p):
        with open(p) as f:
            info.update(json.load(f))
    return info


def find_video(query=None, last=None):
    """Pick a video folder by (part of) its name; else the last one used; else the first."""
    vids = list_videos()
    if not vids:
        raise FileNotFoundError(f"No processed videos in {VIDEOS_DIR} - run process_videos.py first")
    if query:
        hits = [v for v in vids if query.lower() in os.path.basename(v).lower()]
        if not hits:
            raise FileNotFoundError(f"No video matching '{query}'. Available: " +
                                    ", ".join(video_label(v) for v in vids))
        return hits[0]
    for v in vids:
        if last and os.path.basename(v) == last:
            return v
    return vids[0]

DEFAULT_SETTINGS = {
    # channel region of interest (frame rows / columns)
    "roi_y0": 44, "roi_y1": 912, "roi_x0": 5, "roi_x1": 376,
    # HSV threshold for the BLUE FLUID (OpenCV: H 0-179, S 0-255, V 0-255)
    "h_min": 88, "h_max": 125,
    "s_min": 100, "s_max": 255,
    "v_min": 40, "v_max": 255,
    # clean-up (kernel sizes in px, 0 = off)
    "open_k": 3,    # removes isolated specks of 'fluid'
    "close_k": 3,   # fills pinholes in the fluid
    "pillar_pad": 1,  # grow the pillar mask by this many px to swallow pillar rims
    # scale (pixels -> real units). mm_per_px = None means "not set, report pixels".
    "mm_per_px": None,
    "scale_unit": "mm",      # unit used for display: "um", "mm" or "cm"
    "scale_source": "",      # how the scale was set (for the record)
}


# ---- scale / units ---------------------------------------------------------
UNITS_MM = {"um": 0.001, "mm": 1.0, "cm": 10.0}      # size of one unit in mm
UNIT_LABEL = {"um": "µm", "mm": "mm", "cm": "cm"}
GRID_FILE = os.path.join(HERE, "pillar_grid.json")


def pillar_spacing_px():
    """Centre-to-centre spacing of pillars along a row, in px (from make_pillar_mask.py)."""
    if os.path.exists(GRID_FILE):
        with open(GRID_FILE) as f:
            return float(json.load(f)["in_row_spacing_px"])
    return 26.66


def unit_per_px(s):
    """(length of one pixel in the display unit, unit label) or (None, 'px') if no scale."""
    if not s.get("mm_per_px"):
        return None, "px"
    u = s.get("scale_unit", "mm")
    return s["mm_per_px"] / UNITS_MM[u], UNIT_LABEL[u]


def fmt_length(px, s):
    k, lab = unit_per_px(s)
    return f"{px:,.0f} px" if k is None else f"{px * k:,.2f} {lab}"


def fmt_area(px2, s):
    k, lab = unit_per_px(s)
    return f"{px2:,.0f} px" if k is None else f"{px2 * k * k:,.2f} {lab}²"


# Personal tuner state (which video/frame you were looking at) is kept in a separate,
# git-ignored file so it doesn't cause merge conflicts in the shared hsv_settings.json.
STATE_FILE = os.path.join(HERE, ".tuner_state.json")
STATE_KEYS = ("last_video", "last_frame")


def load_settings(path=SETTINGS_FILE):
    s = dict(DEFAULT_SETTINGS)
    for p in (path, STATE_FILE):
        if os.path.exists(p):
            with open(p) as f:
                s.update(json.load(f))
    return s


def save_settings(s, path=SETTINGS_FILE):
    shared = {k: v for k, v in s.items() if k not in STATE_KEYS}
    with open(path, "w") as f:
        json.dump(shared, f, indent=2)
    with open(STATE_FILE, "w") as f:
        json.dump({k: s[k] for k in STATE_KEYS if k in s}, f, indent=2)


def load_pillar_mask(shape, path=PILLAR_FILE):
    if not os.path.exists(path):
        print(f"WARNING: {path} not found - run make_pillar_mask.py. Pillars will be counted!")
        return np.zeros(shape[:2], np.uint8)
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def _kernel(k):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)) if k and k > 1 else None


def interface_length(bubble, fluid):
    """Length (in px) of the boundary between bubble and fluid pixels.

    Uses the Cauchy-Crofton formula over 4 directions (0, 45, 90, 135 deg):
        L = pi/8 * (n0 + n90 + (n45 + n135) / sqrt(2))
    where n_theta = number of bubble/fluid neighbour pairs in that direction.
    Unlike simply counting boundary pixels (~27 % too long for curved shapes),
    this is within ~2 % for circles of radius >= 5 px. Boundaries between a bubble
    and a pillar or the channel wall are NOT counted - only bubble <-> liquid.
    """
    B, F = bubble, fluid
    n0   = np.count_nonzero(B[:, :-1] & F[:, 1:]) + np.count_nonzero(F[:, :-1] & B[:, 1:])
    n90  = np.count_nonzero(B[:-1, :] & F[1:, :]) + np.count_nonzero(F[:-1, :] & B[1:, :])
    n45  = np.count_nonzero(B[:-1, 1:] & F[1:, :-1]) + np.count_nonzero(F[:-1, 1:] & B[1:, :-1])
    n135 = np.count_nonzero(B[:-1, :-1] & F[1:, 1:]) + np.count_nonzero(F[:-1, :-1] & B[1:, 1:])
    return float(np.pi / 8 * (n0 + n90 + (n45 + n135) / np.sqrt(2)))


def interface_pixels(bubble, fluid):
    """Bubble pixels that touch fluid (4-neighbour) - used to draw the interface line."""
    f = fluid.astype(np.uint8)
    near = cv2.dilate(f, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))) > 0
    return bubble & near


def segment(frame_bgr, s, pillar_mask):
    """Return dict of boolean masks (full-frame size) and pixel areas."""
    H, W = frame_bgr.shape[:2]
    roi = np.zeros((H, W), bool)
    roi[s["roi_y0"]:s["roi_y1"], s["roi_x0"]:s["roi_x1"]] = True

    pil = pillar_mask > 0
    if s["pillar_pad"] > 0:
        k = 2 * s["pillar_pad"] + 1
        pil = cv2.dilate(pillar_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
    pil &= roi
    open_area = roi & ~pil                      # where fluid or gas can be

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lo = np.array([s["h_min"], s["s_min"], s["v_min"]], np.uint8)
    hi = np.array([s["h_max"], s["s_max"], s["v_max"]], np.uint8)
    fluid = cv2.inRange(hsv, lo, hi)
    if _kernel(s["open_k"]) is not None:
        fluid = cv2.morphologyEx(fluid, cv2.MORPH_OPEN, _kernel(s["open_k"]))
    if _kernel(s["close_k"]) is not None:
        fluid = cv2.morphologyEx(fluid, cv2.MORPH_CLOSE, _kernel(s["close_k"]))
    fluid = (fluid > 0) & open_area
    bubble = open_area & ~fluid

    n_open = int(open_area.sum())
    return {
        "roi": roi, "pillar": pil, "open": open_area, "fluid": fluid, "bubble": bubble,
        "fluid_px": int(fluid.sum()), "bubble_px": int(bubble.sum()), "open_px": n_open,
        "fluid_frac": fluid.sum() / max(n_open, 1),
        "bubble_frac": bubble.sum() / max(n_open, 1),
        "interface_px": interface_length(bubble, fluid),     # bubble-liquid boundary length, px
        "interface": interface_pixels(bubble, fluid),        # mask, for drawing
    }


def overlay(frame_bgr, r, alpha=0.45, show_interface=True):
    """Colour-coded view: fluid=blue, bubble=orange, pillar=grey, interface=green, outside ROI=dark."""
    out = frame_bgr.copy()
    colours = {"fluid": (255, 120, 0), "bubble": (0, 140, 255), "pillar": (128, 128, 128)}
    for key, col in colours.items():
        m = r[key]
        out[m] = (out[m] * (1 - alpha) + np.array(col) * alpha).astype(np.uint8)
    out[~r["roi"]] = (out[~r["roi"]] * 0.3).astype(np.uint8)
    if show_interface and "interface" in r:
        out[r["interface"]] = (80, 255, 80)            # bubble-liquid interface in green
    return out
