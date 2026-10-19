# Image Recognition — ES410 Fourth Year Project

Measures the **fluid area** (blue liquid), the **bubble area** (gas) and the **bubble–liquid interface length** in 2D reactor videos, frame by frame, using an HSV colour threshold. The reactor's pillars are masked out, so they never count as fluid or bubble.

> Private group repository — please don't make it public or share the data outside the group.

---

## Quick start

```bash
git clone <repo-url>
cd <repo-folder>
pip install -r requirements.txt
```

Then either:

- **Web tuner (easiest):** open `hsv_tuner.html` in **Chrome or Edge**, click **Open folder…** and choose the repo folder.
- **Python tuner:** `python hsv_tuner.py` (or `python hsv_tuner.py n20` to start on a particular video).

The processed frames are already in the repo, so nothing needs to be downloaded or processed first. Every script works only from the frame images in `Videos/`.

---

## Folder layout

```
├── Videos/
│   └── <video>_every10/             every 10th frame, full resolution, lossless PNG
│       ├── frame_0000.png ...       frame_0001 = original frame 10, etc.
│       └── video_info.json          fps, step, frame counts (used for timing)
├── Results/                         <video>_areas.csv from "Analyse all frames"
├── hsv_settings.json                shared threshold settings (used by every script)
├── pillar_mask.png                  mask of the reactor pillars (same for all videos)
├── pillar_grid.json                 fitted pillar spacing in px (used for the scale)
├── hsv_tuner.html              web tuner + batch analysis
├── hsv_tuner.py                     Python (OpenCV) tuner
├── segmentation.py             shared segmentation code + folder paths
├── make_pillar_mask.py              rebuilds pillar_mask.png + pillar_grid.json
└── set_scale.py                     set the pixel -> µm/mm/cm scale
```

### Videos in the repo

| Video | Frames kept | Size |
|---|---:|---:|
| foam2d_w10_n01__20191109_191803 | 69 | 67 MB |
| foam2d_w10_n02__20191109_182042 | 1001 | 985 MB |
| foam2d_w10_n05__20191109_191239 | 1001 | 982 MB |
| foam2d_w10_n10__20191109_180813 | 501 | 498 MB |
| foam2d_w10_n20__20191109_184113 | 1001 | 1.0 GB |
| foam2d_w10_n50__20191109_190122 | 1001 | 998 MB |

All recorded at 30 fps, 384 × 1024 px. Every 10th frame is kept (one frame per 1/3 s).

---

## How it works

Each pixel inside the channel (the yellow box) is put in one of three classes:

| Class | How it's found |
|---|---|
| **Pillar** | `pillar_mask.png`: the regular pillar grid, fitted once from all videos and grown by *Pillar pad* px |
| **Fluid** | HSV threshold on the blue liquid, cleaned with a morphological *open* (removes specks) then *close* (fills pinholes) |
| **Bubble** | everything else in the open channel, i.e. gas plus the dark film edges |
| **Interface** (green) | bubble pixels touching fluid, i.e. the bubble–liquid boundary |

Areas are reported as a **% of the open channel area** (channel minus pillars), and in pixels, or in real units once a scale is set (see below).

### Scale (pixels to real units)

Set the scale once; it's stored in `hsv_settings.json` (`mm_per_px`, `scale_unit`, `scale_source`), so every tool and everyone in the group uses it. Choose µm, mm or cm, then use one of three methods:

| Method | Web tuner (*Scale* section) | Python |
|---|---|---|
| **Measure a line**: click both ends of something of known length (e.g. across the channel) and type its real length | *Measure line*, click two points on the image, type the length, *Set scale* | `python set_scale.py --measure` |
| **Pillar spacing**: type the real centre-to-centre spacing of neighbouring pillars in a row. In the images that's **26.65 px** (from `pillar_grid.json`) | *Pillar spacing* | `python set_scale.py --pillar-spacing 2.0 --unit mm` |
| **Direct**: type the size of one pixel | *Enter directly* | `python set_scale.py --per-px 0.075 --unit mm` |

`python set_scale.py` on its own gives a menu, `--show` prints the current scale, and `--clear` goes back to pixels. With a scale set, the tuners show areas in unit² and interface length in units, and the CSV gets extra real-unit columns. Pixel columns are always kept.

Set the scale in one tool at a time. The Python tuner picks up changes made by `set_scale.py` while it's open.

### Interface length

The **bubble–liquid interface length** is the total length of boundary where bubble pixels touch fluid pixels, drawn as the **green line** on the overlay. Where a bubble touches a pillar or the channel wall, that boundary is **not** counted, because it isn't in contact with liquid.

Simply counting boundary pixels overestimates curved edges by about 27%, so the length is estimated with the **Cauchy–Crofton formula** over four directions:

```
L = π/8 · ( n0 + n90 + (n45 + n135) / √2 )      [px]
```

where *n*θ is the number of bubble/fluid neighbouring-pixel pairs in direction θ. On test circles this is within 2% of the true perimeter for radii ≥ 5 px, and within about 5% for straight edges. It's converted to real units automatically once a scale is set. The code is `interface_length()` in `segmentation.py`, and the web tuner matches it exactly.

Note: the thin liquid films (lamellae) between two touching bubbles are usually too thin to look blue, so they're classed as bubble. Only boundaries against visibly blue liquid count towards the interface.

### CSV columns (`Results/<video>_areas.csv`)

`frame_file, original_frame, time_s, fluid_px, bubble_px, open_px, fluid_frac, bubble_frac, interface_px`, plus `fluid_<unit>2, bubble_<unit>2, interface_<unit>, <unit>_per_px` if a scale is set (e.g. `fluid_mm2`).

The HSV ranges use OpenCV's convention: **H 0–179, S 0–255, V 0–255**. The web tuner reproduces OpenCV's conversion exactly, so both tuners give identical pixel counts.

---

## Workflow

1. **Tune:** open a tuner, drag the sliders until **blue = fluid** and **orange = bubbles** on the overlay. Check several frames and videos. Hover over the image in the web tuner to read a pixel's H/S/V.
2. **Settings auto-save** to `hsv_settings.json`. **Commit it** when you're happy so the rest of the group uses the same thresholds. *RESET TO DEFAULTS* restores the original values.
3. **Analyse:** in the web tuner, pick a video and click **Analyse all frames**. You get plots of fluid/bubble % and interface length against time, and `Results/<video>_areas.csv`.

Keys (both tuners): `m` switch view (overlay / fluid mask / bubble mask), `a` / `d` previous / next frame, `r` reset to defaults, `q` quit (Python).

---

## Adding a new video

The raw videos (6–12 GB each) are too big for GitHub, so they aren't in this repo, and neither is the script that converts them (`process_videos.py`). James has both. To add a video, the person with the raw file:

1. Runs `process_videos.py`. It keeps every 10th frame at full resolution and writes `Videos/<video>_every10/` straight into this repo folder, with a `video_info.json`.
2. Commits the new `Videos/<video>_every10/` folder (see *Pushing large commits* below).

All the code in this repo works only from those frame folders. If the camera position ever changes, rerun `python make_pillar_mask.py`.

---

## Pushing large commits

GitHub rejects a single push over **2 GB** and any file over **100 MB**. Every frame is about 1 MB, so the files are fine, but **commit and push one video folder at a time**:

```bash
git add README.md requirements.txt *.py *.html *.json pillar_mask.png
git commit -m "Code and settings"
git push

git add Videos/foam2d_w10_n01__20191109_191803_every10
git commit -m "Add n01 frames"
git push
# ...repeat for each Videos/<video>_every10 folder
```

The full repo is about **4.6 GB** (4,574 frames), so the first clone takes a while.

**OneDrive:** if the repo folder is inside OneDrive, pause syncing while you commit, or clone the repo to a folder outside OneDrive. OneDrive and git's `.git` folder can clash.

---

## Personal vs shared files

- `hsv_settings.json` is **shared**: thresholds, channel box, scale.
- `.tuner_state.json` (created by the Python tuner) and browser storage (web) are **personal**: the last video and frame you looked at. Don't commit `.tuner_state.json`, `__pycache__/` or `Results/` unless you mean to share them.
