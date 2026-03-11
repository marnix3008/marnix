/*
 * Copyright (C) 2026
 *
 * This file is part of paparazzi
 */
/**
 * @file "modules/danger_detector/danger_detector.h"
 * @author AE4317
 *
 * Divides the bottom strip of the camera image into columns and scores each
 * column by how many non-green (floor) pixels it contains.
 * Green floor = safe, non-green = obstacle/danger.
 *
 * Results are stored in danger_scores[0..DANGER_DETECTOR_NUM_COLUMNS-1] as
 * values 0-100 (0 = fully safe, 100 = fully blocked).
 */

#ifndef DANGER_DETECTOR_H
#define DANGER_DETECTOR_H

#include <stdint.h>
#include <stdbool.h>

#ifndef DANGER_DETECTOR_NUM_COLUMNS
#define DANGER_DETECTOR_NUM_COLUMNS 12
#endif

/** Danger score per column: 0 (safe) … 100 (blocked). Updated on every
 *  processed camera frame. Access from other modules after calling
 *  danger_detector_get_scores(). */
extern uint8_t danger_scores[DANGER_DETECTOR_NUM_COLUMNS];

/** Green-floor YUV thresholds (tunable via GCS settings). */
extern uint8_t dd_lum_min;
extern uint8_t dd_lum_max;
extern uint8_t dd_cb_max;   /* Cb below this value = green-ish (less blue)  */
extern uint8_t dd_cr_max;   /* Cr below this value = green-ish (less red)   */

/** Fraction of the image height used as the floor detection strip (bottom). */
extern float dd_floor_strip_frac;

/** When true, draw column lines and danger-level colours on the image. */
extern bool dd_draw;

/* Module functions (called by the autopilot). */
extern void danger_detector_init(void);
extern void danger_detector_periodic(void);

/**
 * Copy the latest danger scores into the provided array.
 * @param out  array of at least DANGER_DETECTOR_NUM_COLUMNS uint8_t values.
 * @return index of the safest column (0-based), or -1 if no data yet.
 */
extern int danger_detector_get_scores(uint8_t *out);

#endif /* DANGER_DETECTOR_H */
