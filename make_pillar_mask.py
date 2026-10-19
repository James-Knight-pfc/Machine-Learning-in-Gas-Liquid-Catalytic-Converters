"""
Step 2: Build a mask of the reactor pillars so they are never counted
as fluid or bubble.

The pillars sit on a regular staggered grid. Some are hidden under bubbles
in any one frame, so we:
  1. find clearly-visible round white blobs in frames from EVERY processed
     video in Videos/ (the camera/reactor position is the same for all runs),
  2. fit the grid (row pitch, column pitch, origin) to their centres,
  3. draw a filled circle at EVERY grid position inside the channel.

Outputs (in the Image Recognition folder, shared by every video):
  pillar_mask.png   white = pillar, black = not pillar (same size as frames)
  pillar_check.png  first frame with the fitted pillars outlined in red
"""
import glob
import json
import os

import cv2
import numpy as np

from segmentation import HERE, SETTINGS_FILE as SETTINGS, PILLAR_FILE, list_videos, list_frames

# Channel region (rows / columns of the frame that are inside the reactor).
# Everything outside this box is ignored by all later steps.
DEFAULTS = {"roi_y0": 44, "roi_y1": 912, "roi_x0": 5, "roi_x1": 376}

# Pillar detection: pillars are bright (high V) and nearly colourless (low S)
PILLAR_V_MIN = 200
PILLAR_S_MAX = 90
PILLAR_RADIUS_DRAW = 7   # px; a bit bigger than the ~6 px core to cover the dark rim


def load_roi():
    s = dict(DEFAULTS)
    if os.path.exists(SETTINGS):
        with open(SETTINGS) as f:
            s.update({k: v for k, v in json.load(f).items() if k in DEFAULTS})
    return s


def find_pillar_centres(paths):
    pts = []
    for p in paths:
        hsv = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2HSV)
        m = ((hsv[..., 2] > PILLAR_V_MIN) & (hsv[..., 1] < PILLAR_S_MAX)).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in contours:
            a, per = cv2.contourArea(c), cv2.arcLength(c, True)
            if 80 < a < 220 and per > 0 and 4 * np.pi * a / per ** 2 > 0.8:   # round, pillar-sized
                M = cv2.moments(c)
                pts.append((M["m10"] / M["m00"], M["m01"] / M["m00"]))
    return np.array(pts)


def fit_grid(pts, py0=26.6, px0=13.3):
    """Fit centres to x = x0 + j*px, y = y0 + k*py with (j+k) of fixed parity."""
    y0, x0 = np.median(pts[:, 1] - py0 * np.round(pts[:, 1] / py0)), \
             np.median(pts[:, 0] - px0 * np.round(pts[:, 0] / px0))
    py, px = py0, px0
    for _ in range(5):   # refine by least squares
        k = np.round((pts[:, 1] - y0) / py)
        j = np.round((pts[:, 0] - x0) / px)
        py, y0 = np.polyfit(k, pts[:, 1], 1)
        px, x0 = np.polyfit(j, pts[:, 0], 1)
    k = np.round((pts[:, 1] - y0) / py).astype(int)
    j = np.round((pts[:, 0] - x0) / px).astype(int)
    parity = int(np.round(np.mean((j + k) % 2)))
    return x0, y0, px, py, parity


def main():
    videos = list_videos()
    if not videos:
        raise FileNotFoundError("No processed videos in Videos/ - run process_videos.py first")
    paths = []
    for v in videos:                       # ~40 evenly spaced frames from each video
        fr = list_frames(v)
        paths += fr[::max(1, len(fr) // 40)]
    print(f"Using {len(paths)} frames from {len(videos)} video(s)")
    roi = load_roi()
    first = cv2.imread(paths[0])
    H, W = first.shape[:2]

    pts = find_pillar_centres(paths)
    x0, y0, px, py, parity = fit_grid(pts)
    print(f"Grid fit from {len(pts)} detections: row pitch {py:.2f}px, "
          f"column pitch {px:.2f}px (pillar spacing in a row {2*px:.2f}px)")

    mask = np.zeros((H, W), np.uint8)
    n = 0
    for k in range(-5, int(H / py) + 5):
        for j in range(-5, int(W / px) + 5):
            if (j + k) % 2 != parity:
                continue
            x, y = x0 + j * px, y0 + k * py
            # keep pillars inside the channel, incl. half-pillars built into the side walls
            r = PILLAR_RADIUS_DRAW
            if roi["roi_x0"] - r <= x <= roi["roi_x1"] + r and roi["roi_y0"] + 10 <= y <= roi["roi_y1"] - 10:
                cv2.circle(mask, (int(round(x)), int(round(y))), PILLAR_RADIUS_DRAW, 255, -1)
                n += 1

    cv2.imwrite(PILLAR_FILE, mask)
    with open(os.path.join(HERE, "pillar_grid.json"), "w") as f:
        json.dump({"row_pitch_px": round(py, 3), "in_row_spacing_px": round(2 * px, 3),
                   "column_pitch_px": round(px, 3), "x0": round(x0, 3), "y0": round(y0, 3),
                   "note": "in_row_spacing_px = centre-to-centre distance between neighbouring "
                           "pillars in the same row (used to set the scale)"}, f, indent=2)
    check = first.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(check, contours, -1, (0, 0, 255), 1)
    cv2.rectangle(check, (roi["roi_x0"], roi["roi_y0"]), (roi["roi_x1"], roi["roi_y1"]), (0, 255, 255), 1)
    cv2.imwrite(os.path.join(HERE, "pillar_check.png"), check)
    print(f"Drew {n} pillars -> pillar_mask.png (check it with pillar_check.png)")


if __name__ == "__main__":
    main()
