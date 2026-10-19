"""
Step 3: Interactive HSV threshold tuner.

Drag the sliders until the BLUE overlay covers only the fluid and the
ORANGE overlay covers only the bubbles. Pillars (grey) are excluded
automatically using pillar_mask.png.

Works on the processed videos in Videos/ (made by process_videos.py).
Areas/lengths are shown in real units once a scale is set (python set_scale.py).
Pick the video with the "Video" slider, or start with part of its name:
    python hsv_tuner.py n20

Two windows:
  "HSV tuner - controls"  sliders, live areas, RESET button, key help
  "HSV tuner - view"      original frame | colour-coded result

Your slider values are AUTO-SAVED to hsv_settings.json whenever you change
them, and loaded again next time you open the tuner.

Keys (work with either window selected):
  m        switch view: overlay | fluid mask | bubble mask
  a / d    previous / next frame
  r        reset sliders to the default values (or click the RESET button)
  q / Esc  quit
"""
import os
import time

import cv2
import numpy as np

import sys

from segmentation import (SETTINGS_FILE, list_videos, list_frames, find_video, video_label, video_info,
                               DEFAULT_SETTINGS, load_settings, save_settings,
                               fmt_area, fmt_length, unit_per_px,
                               load_pillar_mask, segment, overlay)

DISPLAY_SCALE = 0.75         # size of the picture window (lower it if too tall for your screen)
WIN_VIEW, WIN_CTRL = "HSV tuner - view", "HSV tuner - controls"
AUTOSAVE_DELAY = 0.4         # s after the last slider move before saving

# slider name -> (settings key, max value)
SLIDERS = [
    ("H min", "h_min", 179), ("H max", "h_max", 179),
    ("S min", "s_min", 255), ("S max", "s_max", 255),
    ("V min", "v_min", 255), ("V max", "v_max", 255),
    ("Open k", "open_k", 15), ("Close k", "close_k", 15),
    ("Pillar pad", "pillar_pad", 5),
]
VIEWS = ["overlay", "fluid mask", "bubble mask"]

PANEL_W = 520                # width of the info panel under the sliders (px)
FONT, FS = cv2.FONT_HERSHEY_SIMPLEX, 0.55
LINE_H = 26
BTN_H = 44


def render_view(frame, r, view):
    """Picture window: original | result."""
    if VIEWS[view] == "overlay":
        right = overlay(frame, r)
    else:
        m = r["fluid"] if VIEWS[view] == "fluid mask" else r["bubble"]
        right = cv2.cvtColor(m.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    canvas = np.hstack([frame, np.full((frame.shape[0], 6, 3), 255, np.uint8), right])
    if DISPLAY_SCALE != 1:
        canvas = cv2.resize(canvas, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE,
                            interpolation=cv2.INTER_AREA)
    return canvas


def render_panel(r, view, video_name, frame_name, status):
    """Info panel shown under the sliders. Returns (image, reset-button rect)."""
    panel = np.full((520, PANEL_W, 3), 35, np.uint8)
    W = PANEL_W
    col2 = W // 2 + 10

    def txt(t, x, y, col=(225, 225, 225), s=FS, th=1):
        cv2.putText(panel, t, (x, y), FONT, s, col, th, cv2.LINE_AA)

    y = 28
    txt(video_name, 14, y, (120, 220, 255), 0.55, 2)
    y += LINE_H
    txt(frame_name, 14, y, (255, 255, 255), 0.55, 1)
    txt(f"View: {VIEWS[view].upper()}", col2 + 40, y, (255, 255, 255), 0.6, 2)
    y += LINE_H + 6
    txt("Fluid (blue)", 14, y, (255, 170, 60), th=2)
    txt("Bubble (orange)", col2, y, (60, 170, 255), th=2)
    y += LINE_H
    sc = r.get("scale", {})
    txt(f"{fmt_area(r['fluid_px'], sc)}   {100*r['fluid_frac']:.1f}%", 14, y, (255, 170, 60))
    txt(f"{fmt_area(r['bubble_px'], sc)}   {100*r['bubble_frac']:.1f}%", col2, y, (60, 170, 255))
    y += 20
    txt("% of open channel area (pillars excluded)", 14, y, (150, 150, 150), 0.45)
    y += LINE_H + 4
    il = r.get("interface_px", 0.0)
    txt("Interface (green):", 14, y, (80, 255, 80), th=2)
    txt(fmt_length(il, sc), 190, y, (80, 255, 80))
    y += 18
    txt("bubble-liquid boundary length (pillar/wall contact excluded)", 14, y, (150, 150, 150), 0.45)
    y += 20
    k, lab = unit_per_px(sc)
    txt(f"Scale: 1 px = {k:.4g} {lab}" if k else "Scale: not set (pixels) - run set_scale.py",
        14, y, (200, 200, 120), 0.45)
    y += LINE_H
    txt(status, 14, y, (120, 255, 120))

    # RESET button
    y += 14
    x0, y0, x1, y1 = 14, y, W - 14, y + BTN_H
    cv2.rectangle(panel, (x0, y0), (x1, y1), (40, 40, 170), -1)
    cv2.rectangle(panel, (x0, y0), (x1, y1), (210, 210, 255), 2)
    label = "RESET TO DEFAULTS"
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.65, 2)
    txt(label, x0 + (x1 - x0 - tw) // 2, y0 + (BTN_H + th) // 2, (255, 255, 255), 0.65, 2)
    y = y1 + 30

    txt("KEYS", 14, y, (255, 255, 255), th=2)
    keys = [("m", "switch view (overlay / fluid / bubble)"),
            ("a / d", "previous / next frame"),
            ("r", "reset to defaults"),
            ("q / Esc", "quit")]
    for k, d in keys:
        y += LINE_H
        txt(k, 14, y, (120, 220, 255), th=2)
        txt(d, 110, y)
    y += LINE_H
    txt("Sliders auto-save when changed.", 14, y, (150, 150, 150), 0.45)
    return panel[:y + 14], (x0, y0, x1, y1)


def main():
    print("HSV tuner v5 (video picker) running from:", os.path.abspath(__file__))
    s = load_settings()
    vids = list_videos()
    start_vid = find_video(sys.argv[1] if len(sys.argv) > 1 else None, s.get("last_video"))
    vi = vids.index(start_vid)
    cur = {"vi": vi, "paths": list_frames(start_vid), "info": video_info(start_vid)}
    frames = {}

    def get_frame(i):
        key = (cur["vi"], i)
        if key not in frames:
            if len(frames) > 40:
                frames.clear()
            frames[key] = cv2.imread(cur["paths"][i])
        return frames[key]

    pillar_mask = load_pillar_mask(get_frame(0).shape)
    keep = os.path.basename(start_vid) == s.get("last_video")
    start_frame = int(min(max(s.get("last_frame", 0), 0), len(cur["paths"]) - 1)) if keep else 0
    s["last_video"] = os.path.basename(start_vid)
    print("Videos:", ", ".join(f"[{i}] {video_label(v)}" for i, v in enumerate(vids)))

    cv2.namedWindow(WIN_VIEW, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(WIN_CTRL, cv2.WINDOW_AUTOSIZE)   # sliders on top, info panel below
    if len(vids) > 1:
        cv2.createTrackbar("Video", WIN_CTRL, vi, len(vids) - 1, lambda v: None)
    cv2.createTrackbar("Frame", WIN_CTRL, start_frame, len(cur["paths"]) - 1, lambda v: None)
    for name, key, vmax in SLIDERS:
        cv2.createTrackbar(name, WIN_CTRL, int(s[key]), vmax, lambda v: None)

    clicks = {"reset": False, "btn": (0, 0, 0, 0)}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            x0, y0, x1, y1 = clicks["btn"]
            if x0 <= x <= x1 and y0 <= y <= y1:
                clicks["reset"] = True
    cv2.setMouseCallback(WIN_CTRL, on_mouse)

    def reset_sliders():
        for name, k, _ in SLIDERS:
            cv2.setTrackbarPos(name, WIN_CTRL, DEFAULT_SETTINGS[k])

    def slider_values():
        return tuple(s[k] for _, k, _ in SLIDERS)

    view, last_state = 0, None
    status = "Loaded saved settings" if os.path.exists(SETTINGS_FILE) else "Using default settings"
    saved_values, last_change = slider_values(), None

    SCALE_KEYS = ("mm_per_px", "scale_unit", "scale_source")
    scale_mtime = [os.path.getmtime(SETTINGS_FILE) if os.path.exists(SETTINGS_FILE) else 0]

    def refresh_scale():
        """Scale is set by set_scale.py / the web tuner - re-read it if the file changed."""
        m = os.path.getmtime(SETTINGS_FILE) if os.path.exists(SETTINGS_FILE) else 0
        if m != scale_mtime[0]:
            scale_mtime[0] = m
            disk = load_settings()
            changed = any(s.get(k) != disk.get(k) for k in SCALE_KEYS)
            for k in SCALE_KEYS:
                s[k] = disk.get(k)
            return changed
        return False

    def save():
        refresh_scale()                      # never overwrite a newer scale
        save_settings(s)
        scale_mtime[0] = os.path.getmtime(SETTINGS_FILE)

    last_check = 0.0
    while True:
        if time.time() - last_check > 1.0:
            last_check = time.time()
            if refresh_scale():
                status, last_state = "Scale updated", None
        # switch video?
        if len(vids) > 1:
            new_vi = cv2.getTrackbarPos("Video", WIN_CTRL)
            if new_vi != cur["vi"]:
                cur.update(vi=new_vi, paths=list_frames(vids[new_vi]), info=video_info(vids[new_vi]))
                cv2.setTrackbarMax("Frame", WIN_CTRL, len(cur["paths"]) - 1)
                cv2.setTrackbarPos("Frame", WIN_CTRL, 0)
                s["last_video"] = os.path.basename(vids[new_vi])
                save()
                status = "Switched to " + video_label(vids[new_vi])
        paths = cur["paths"]
        idx = min(cv2.getTrackbarPos("Frame", WIN_CTRL), len(paths) - 1)
        for name, key, _ in SLIDERS:
            s[key] = cv2.getTrackbarPos(name, WIN_CTRL)
        s["last_frame"] = idx

        # auto-save shortly after the sliders stop moving
        if slider_values() != saved_values:
            if last_change is None or slider_values() != last_change[1]:
                last_change = (time.time(), slider_values())
            elif time.time() - last_change[0] > AUTOSAVE_DELAY:
                save()
                saved_values, last_change = slider_values(), None
                status = "Auto-saved " + time.strftime("%H:%M:%S")

        state = (cur["vi"], idx, view, slider_values(), status)
        if state != last_state:
            frame = get_frame(idx)
            r = segment(frame, s, pillar_mask)
            r["scale"] = s
            cv2.imshow(WIN_VIEW, render_view(frame, r, view))
            t = idx * cur["info"]["step"] / cur["info"]["fps"]
            panel, clicks["btn"] = render_panel(r, view, video_label(vids[cur["vi"]]),
                                                f"{os.path.basename(paths[idx])}  t={t:.2f}s", status)
            cv2.imshow(WIN_CTRL, panel)
            last_state = state

        key = cv2.waitKey(30) & 0xFF
        if clicks["reset"] or key == ord("r"):
            clicks["reset"] = False
            reset_sliders()
            status = "Reset to default values"
        if key in (ord("q"), 27):
            break
        elif key == ord("d"):
            cv2.setTrackbarPos("Frame", WIN_CTRL, min(idx + 1, len(paths) - 1))
        elif key == ord("a"):
            cv2.setTrackbarPos("Frame", WIN_CTRL, max(idx - 1, 0))
        elif key == ord("m"):
            view = (view + 1) % len(VIEWS)
        if (cv2.getWindowProperty(WIN_VIEW, cv2.WND_PROP_VISIBLE) < 1 or
                cv2.getWindowProperty(WIN_CTRL, cv2.WND_PROP_VISIBLE) < 1):
            break   # a window was closed with the X button

    save()   # always save on exit
    print("Settings saved to hsv_settings.json:",
          {k: s[k] for _, k, _ in SLIDERS})
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
