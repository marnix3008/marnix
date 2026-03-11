/*
 * Copyright (C) 2026
 *
 * This file is part of paparazzi
 */
/**
 * @file "modules/danger_detector/danger_detector.h"
 * @author AE4317
 *
 * Divides the left strip of the camera image (portrait orientation) into
 * horizontal bands and scores each band by how many non-green (floor) pixels
 * it contains.
 *
 * The camera is mounted in portrait orientation: the floor is visible on the
 * LEFT side of the image.  Each of the DANGER_DETECTOR_NUM_ZONES horizontal
 * bands receives a danger score 0–100:
 *   0–24   low      (mostly safe floor)
 *   25–49  medium
 *   50–74  high
 *   75–100 critical (no floor visible)
 *
 * Results are stored in danger_scores[0..DANGER_DETECTOR_NUM_ZONES-1].
 */

#ifndef DANGER_DETECTOR_H
#define DANGER_DETECTOR_H

#include <stdint.h>
#include <stdbool.h>

#ifndef DANGER_DETECTOR_NUM_ZONES
#define DANGER_DETECTOR_NUM_ZONES 9
#endif

/** Danger score per horizontal zone: 0 (safe) … 100 (blocked). Updated on
 *  every processed camera frame. Access from other modules after calling
 *  danger_detector_get_scores(). */
extern uint8_t danger_scores[DANGER_DETECTOR_NUM_ZONES];

/** Green-floor YUV thresholds (tunable via GCS settings). */
extern uint8_t dd_lum_min;
extern uint8_t dd_lum_max;
extern uint8_t dd_cb_max;   /* Cb below this value = green-ish (less blue)  */
extern uint8_t dd_cr_max;   /* Cr below this value = green-ish (less red)   */

/** Fraction of the image WIDTH used as the floor detection strip (left side). */
extern float dd_floor_strip_frac;

/** When true, draw zone lines and danger-level colours on the image. */
extern bool dd_draw;

/* Module functions (called by the autopilot). */
extern void danger_detector_init(void);
extern void danger_detector_periodic(void);

/**
 * Copy the latest danger scores into the provided array.
 * @param out  array of at least DANGER_DETECTOR_NUM_ZONES uint8_t values.
 * @return index of the safest zone (0-based), or -1 if no data yet.
 */
extern int danger_detector_get_scores(uint8_t *out);

#endif /* DANGER_DETECTOR_H */
