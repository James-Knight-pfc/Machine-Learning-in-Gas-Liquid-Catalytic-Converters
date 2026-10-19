"""
Set the image scale (pixels -> real units). Saved in hsv_settings.json, so every
script, the web tuner and the whole group use the same scale.

Three ways to set it:
  1. Measure a line   - click two points on a frame and type the real distance
  2. Pillar spacing   - type the real centre-to-centre spacing of pillars in a row
                        (their spacing in pixels is known from the pillar grid fit)
  3. Direct           - type the size of one pixel

Usage:
  python set_scale.py                                  interactive menu
  python set_scale.py --measure [--video n20]          click two points on a frame
  python set_scale.py --pillar-spacing 2.0 --unit mm   from pillar spacing
  python set_scale.py --per-px 0.075 --unit mm         one pixel = 0.075 mm
  python set_scale.py --show                           print the current scale
  python set_scale.py --clear                          go back to pixels
Units: um, mm (default) or cm.
"""
import argparse
import math
import sys

import cv2
import numpy as np

from segmentation import (UNITS_MM, UNIT_LABEL, load_settings, save_settings, find_video,
                               list_frames, video_label, pillar_spacing_px, unit_per_px)


def describe(s):
    k, lab = unit_per_px(s)
    if k is None:
        return "No scale set - results are in pixels."
    txt = f"1 px = {k:.5g} {lab}   ({1 / k:.4g} px per {lab})"
    if s.get("scale_source"):
        txt += f"\n   set from: {s['scale_source']}"
    return txt


def apply(s, mm_per_px, unit, source):
    s["mm_per_px"] = float(mm_per_px)
    s["scale_unit"] = unit
    s["scale_source"] = source
    save_settings(s)
    print("\nSaved to hsv_settings.json:\n  " + describe(s))


def ask_float(prompt):
    while True:
        v = input(prompt).strip()
        try:
            v = float(v)
            if v > 0:
                return v
        except ValueError:
            pass
        print("  Please enter a positive number.")


def ask_unit(default="mm"):
    u = input(f"Unit [um / mm / cm] (Enter = {default}): ").strip().lower().replace("µ", "u") or default
    while u not in UNITS_MM:
        u = input("  Please type um, mm or cm: ").strip().lower() or default
    return u


def measure_line(video_query=None):
    """Show a frame; user clicks two points. Returns (length_px, (p1, p2), frame name) or None."""
    s = load_settings()
    vid = find_video(video_query, s.get("last_video"))
    frames = list_frames(vid)
    img = cv2.imread(frames[min(int(s.get("last_frame", 0) or 0), len(frames) - 1)])
    pts = []
    win = "Click two points (Enter = accept, r = redo, Esc = cancel)"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(pts) == 2:
                pts.clear()
            pts.append((x, y))

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)       # resizable; clicks are in image pixels
    h, w = img.shape[:2]
    cv2.resizeWindow(win, int(w * 0.8), int(h * 0.8))
    cv2.setMouseCallback(win, on_mouse)
    print(f"\nVideo: {video_label(vid)}\nClick both ends of something of known length "
          "(e.g. across the channel, or between pillar centres).\n"
          "Tip: maximise the window for more precise clicks. Enter = accept, r = redo, Esc = cancel.")
    result = None
    while True:
        view = img.copy()
        for p in pts:
            cv2.drawMarker(view, p, (0, 255, 255), cv2.MARKER_CROSS, 14, 1)
        if len(pts) == 2:
            cv2.line(view, pts[0], pts[1], (0, 255, 255), 1, cv2.LINE_AA)
            L = math.dist(pts[0], pts[1])
            cv2.putText(view, f"{L:.1f} px", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(win, view)
        k = cv2.waitKey(30) & 0xFF
        if k in (13, 10) and len(pts) == 2:
            result = (math.dist(pts[0], pts[1]), tuple(pts), video_label(vid))
            break
        if k == ord("r"):
            pts.clear()
        if k == 27 or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyWindow(win)
    cv2.waitKey(1)
    return result


def do_measure(video=None, real=None, unit=None):
    m = measure_line(video)
    if not m:
        print("Cancelled - scale unchanged.")
        return
    L, (p1, p2), name = m
    print(f"Line length: {L:.2f} px  ({p1} -> {p2})")
    unit = unit or ask_unit()
    real = real or ask_float(f"Real length of that line in {UNIT_LABEL[unit]}: ")
    apply(load_settings(), real * UNITS_MM[unit] / L, unit,
          f"measured line {L:.1f} px = {real:g} {UNIT_LABEL[unit]} on {name}")


def do_pillar(spacing=None, unit=None):
    px = pillar_spacing_px()
    print(f"\nPillars in the same row are {px:.2f} px apart (centre to centre) in the images.")
    unit = unit or ask_unit()
    spacing = spacing or ask_float(f"Real centre-to-centre pillar spacing in {UNIT_LABEL[unit]}: ")
    apply(load_settings(), spacing * UNITS_MM[unit] / px, unit,
          f"pillar spacing {spacing:g} {UNIT_LABEL[unit]} = {px:.2f} px")


def do_direct(per_px=None, unit=None):
    unit = unit or ask_unit()
    per_px = per_px or ask_float(f"Size of one pixel in {UNIT_LABEL[unit]}: ")
    apply(load_settings(), per_px * UNITS_MM[unit], unit, f"entered directly: 1 px = {per_px:g} {UNIT_LABEL[unit]}")


def do_clear():
    s = load_settings()
    s["mm_per_px"], s["scale_source"] = None, ""
    save_settings(s)
    print("Scale cleared - results are in pixels.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--measure", action="store_true", help="click two points on a frame")
    ap.add_argument("--video", help="part of a video name, for --measure")
    ap.add_argument("--length", type=float, help="real length of the measured line (with --measure)")
    ap.add_argument("--pillar-spacing", type=float, help="real centre-to-centre pillar spacing")
    ap.add_argument("--per-px", type=float, help="real size of one pixel")
    ap.add_argument("--unit", choices=list(UNITS_MM), help="um, mm or cm (default mm)")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--clear", action="store_true")
    a = ap.parse_args()

    if a.show:
        print(describe(load_settings())); return
    if a.clear:
        do_clear(); return
    if a.pillar_spacing:
        do_pillar(a.pillar_spacing, a.unit or "mm"); return
    if a.per_px:
        do_direct(a.per_px, a.unit or "mm"); return
    if a.measure:
        do_measure(a.video, a.length, a.unit); return

    print("Current scale: " + describe(load_settings()))
    print("\nHow do you want to set the scale?\n"
          "  1  Measure a line on a frame (click two points)\n"
          "  2  Pillar spacing (centre to centre, same row)\n"
          "  3  Enter the size of one pixel\n"
          "  4  Clear the scale (back to pixels)\n"
          "  q  Quit")
    c = input("Choice: ").strip().lower()
    {"1": do_measure, "2": do_pillar, "3": do_direct, "4": do_clear}.get(c, lambda: None)()


if __name__ == "__main__":
    main()
