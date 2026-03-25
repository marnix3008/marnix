/*
 * Copyright (C) 2026
 *
 * This file is part of paparazzi
 */
/**
 * @file "modules/gate_detector/gate_detector.h"
 * @author AE4317
 *
 * Detects a blue gate by finding two wide horizontal blue blobs (posts) in the
 * front camera image. The Bebop camera is physically rotated 90°, so posts
 * appear as horizontal stripes (one near top, one near bottom of the image).
 * Works directly in YUV422 — no HSV conversion needed.
 *
 * Algorithm:
 *   1. Subsample the image (every 2nd row and 2nd column) for speed.
 *   2. Build a per-row histogram of blue pixels + track horizontal span per row.
 *   3. Group active rows into "runs" (post candidates).
 *   4. Keep the two highest-scoring runs, validate each with a width/height ratio.
 *   5. If two valid posts exist, gate_detected = true and outputs center + aperture.
 *
 * Blue YUV defaults tuned from the RGB gate-post samples
 *   RGB ~(54,115,196):  Y≈106, Cb≈179, Cr≈91
 */

#ifndef GATE_DETECTOR_H
#define GATE_DETECTOR_H

#include <stdint.h>
#include <stdbool.h>

/* Blue pixel YUV thresholds (tunable via GCS). */
extern uint8_t gd_y_min;    /* minimum luminance                    */
extern uint8_t gd_y_max;    /* maximum luminance                    */
extern uint8_t gd_cb_min;   /* minimum Cb (must be high for blue)   */
extern uint8_t gd_cb_max;   /* maximum Cb                           */
extern uint8_t gd_cr_min;   /* minimum Cr                           */
extern uint8_t gd_cr_max;   /* maximum Cr (must be low for blue)    */

/* Detection parameters (tunable via GCS). */
extern uint8_t gd_min_ratio_x10;  /* min width/height ratio × 10 (e.g. 18 = 1.8) */
extern uint8_t gd_min_col_fill;   /* min % of sub-sampled columns blue in a post row (0-100) */
extern uint8_t gd_min_post_sep;   /* min sub-sampled row gap between the two posts */

/* Draw detected posts / gate on the image when true. */
extern bool gd_draw;

/* Detection results (read by navigation module). */
extern bool     gd_gate_detected;
extern int16_t  gd_gate_x;      /* gate centre x in image pixels */
extern int16_t  gd_gate_y;      /* gate centre y in image pixels */
extern uint16_t gd_gate_width;  /* vertical pixel distance between post centres (gate aperture) */

/* Module API. */
void gate_detector_init(void);
void gate_detector_periodic(void);

#endif /* GATE_DETECTOR_H */
