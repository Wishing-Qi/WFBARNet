# OpenCV Court Homography Demo

Traditional OpenCV demo for badminton court white-line detection and homography
tracking. It does not use a deep model.

Pipeline:

1. HSV green-court ROI modeling.
2. Lab/HSV white-line response modeling: local brightness contrast, top-hat ridge response,
   low saturation, and low Lab chroma.
3. Paired green-side support so green mat / floor borders are suppressed.
4. Morphological open/close cleanup.
5. Canny edges.
6. HoughLinesP line detection.
7. Direction clustering and nearby-line merging.
8. Intersections between the court line families.
9. Candidate court quadrilateral search.
10. Confidence scoring and temporal smoothing.
11. Standard 610 x 1340 court template projection.

Run:

```powershell
python tools/opencv_court_homography_demo/run_opencv_court_homography.py
```

## Smooth Pipeline Demo

`run_court_homography_pipeline.py` is a cleaner end-to-end version organized
exactly as:

```text
frame -> HSV white mask -> morphology close -> Canny -> HoughLinesP
-> angle grouping -> line intersections -> badminton geometry filtering
-> Homography -> standard court template projection
```

Run:

```powershell
python tools/opencv_court_homography_demo/run_court_homography_pipeline.py
```

Show intermediate model lines:

```powershell
python tools/opencv_court_homography_demo/run_court_homography_pipeline.py --draw-debug
```

Run with a camera:

```powershell
python tools/opencv_court_homography_demo/run_opencv_court_homography.py --source 0
```

Save output:

```powershell
python tools/opencv_court_homography_demo/run_opencv_court_homography.py `
  --source videos/MVI_0212.MP4 `
  --save-video outputs/opencv_court_homography.mp4
```

Useful tuning flags:

- `--redetect-interval 4.0`
- `--white-s-max 130`
- `--white-v-min 120`
- `--white-chroma-max 96`
- `--line-response-percentile 91`
- `--line-response-min 72`
- `--line-local-bg-ksize 31`
- `--green-s-min 70`
- `--white-green-pair-offset-px 8`
- `--no-green-roi`
- `--keep-all-green-rois`
- `--no-refine-homography`
- `--draw-debug-lines`
- `--point-scheme auto`

Confidence behavior:

- `confidence >= 0.75`: reliable update.
- `0.55 <= confidence < 0.75`: medium update, blended with previous state.
- `confidence < 0.55`: rejected, previous reliable homography is reused.

The output state stores both homographies:

- `court_to_image_h`: standard court coordinates to image pixels.
- `image_to_court_h`: image pixels to standard court coordinates.
