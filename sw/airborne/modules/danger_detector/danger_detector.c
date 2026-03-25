/*
 * Copyright (C) 2026
 *
 * This file is part of paparazzi
 */
/**
 * @file "modules/danger_detector/danger_detector.c"
 * @author AE4317
 *
 * Danger detector for the TU Delft Cyberzoo.
 *
 * The camera is mounted in portrait orientation with the floor visible on the
 * LEFT side of the image.  The module:
 *
 *   1. Takes the left dd_floor_strip_frac fraction of the image width as the
 *      "floor detection strip".
 *   2. Divides the image HEIGHT into DANGER_DETECTOR_NUM_ZONES horizontal bands.
 *   3. For each band, counts non-green pixels inside the strip and computes a
 *      danger score 0–100 (0 = fully safe / green floor, 100 = fully blocked).
 *
 * Danger levels:
 *   score  0–24  → low
 *   score 25–49  → medium
 *   score 50–74  → high
 *   score 75–100 → critical
 *
 * The green-floor detection uses a simple YUV422 threshold:
 *   - Y  (luminance): dd_lum_min … dd_lum_max       (avoid very dark / overexposed)
 *   - Cb (U, blue chroma): below dd_cb_max           (green has low blue chroma)
 *   - Cr (V, red chroma):  below dd_cr_max           (green has low red chroma)
 *
 * Results are available via the global danger_scores[] array and the
 * danger_detector_get_scores() helper. For ABI consumers (e.g.
 * orange_avoider), this module now publishes per-zone blocked flags (0/1)
 * based on dd_danger_threshold.
 *
 * NOTE: This module does NOT yet steer the drone – it only produces scores
 * that other modules can read.
 */

#include "modules/danger_detector/danger_detector.h"
#include "modules/danger_detector/grid_utils.h"
#include "modules/computer_vision/cv.h"
#include "modules/computer_vision/lib/vision/image.h"
#include "modules/core/abi.h"
#include "std.h"

#include <string.h>
#include <stdio.h>
#include <pthread.h>

/* -------------------------------------------------------------------------
 * Compile-time defaults (can be overridden in the airframe file)
 * ---------------------------------------------------------------------- */

#ifndef DANGER_DETECTOR_CAMERA
#define DANGER_DETECTOR_CAMERA front_camera
#endif

#ifndef DANGER_DETECTOR_FPS
#define DANGER_DETECTOR_FPS 0  /* 0 = run at camera rate */
#endif

/* -------------------------------------------------------------------------
 * Tunable settings (accessible via GCS)
 * ---------------------------------------------------------------------- */

uint8_t dd_lum_min        = 40;   /* minimum Y  – ignore very dark pixels   */
uint8_t dd_lum_max        = 200;  /* maximum Y  – ignore overexposed pixels  */
uint8_t dd_cb_max         = 120;  /* max Cb (U) for a pixel to be "green"   */
uint8_t dd_cr_max         = 140;  /* max Cr (V) for a pixel to be "green"   */
uint8_t dd_danger_threshold = 30; /* score >= threshold is sent as blocked=1 */
float   dd_floor_strip_frac = 0.25f; /* left fraction of image width to inspect */
bool    dd_draw           = true; /* overlay zone lines and danger colours   */

/* -------------------------------------------------------------------------
 * Public result array + mutex
 * ---------------------------------------------------------------------- */

uint8_t danger_scores[DANGER_DETECTOR_NUM_ZONES];

/** Internal (camera-thread) copy, protected by mutex. */
static uint8_t _scores_buf[DANGER_DETECTOR_NUM_ZONES];
static bool    _scores_updated = false;
static pthread_mutex_t _mutex;

/* -------------------------------------------------------------------------
 * Camera callback – runs in the video thread
 * ---------------------------------------------------------------------- */

/**
 * is_green_yuv – returns true if the pixel at (y_val, cb_val, cr_val) passes
 * the simple green-floor filter.
 */
static inline bool is_green_yuv(uint8_t y_val, uint8_t cb_val, uint8_t cr_val)
{
  return (y_val  >= dd_lum_min) &&
         (y_val  <= dd_lum_max) &&
         (cb_val <= dd_cb_max)  &&
         (cr_val <= dd_cr_max);
}

static inline void dd_set_yuv422_pixel(uint8_t *buf, uint16_t width, uint16_t height,
                                       int32_t row, int32_t x,
                                       uint8_t y, uint8_t cb, uint8_t cr)
{
  if (!buf || row < 0 || x < 0 || row >= (int32_t)height || x >= (int32_t)width) {
    return;
  }

  uint16_t uy = (uint16_t)row;
  uint16_t ux = (uint16_t)x;
  uint32_t base = (uint32_t)uy * 2u * width + 2u * ux;
  if ((ux & 1u) == 0u) {
    buf[base] = cb;
    buf[base + 1u] = y;
    buf[base + 2u] = cr;
  } else {
    buf[base - 2u] = cb;
    buf[base + 1u] = y;
    buf[base] = cr;
  }
}

static void dd_draw_dot(uint8_t *buf, uint16_t width, uint16_t height,
                        uint16_t cx, uint16_t cy, uint8_t radius,
                        uint8_t y, uint8_t cb, uint8_t cr)
{
  int32_t r = (int32_t)radius;
  int32_t r2 = r * r;
  for (int32_t dy = -r; dy <= r; dy++) {
    for (int32_t dx = -r; dx <= r; dx++) {
      if ((dx * dx + dy * dy) <= r2) {
        dd_set_yuv422_pixel(buf, width, height, (int32_t)cy + dy, (int32_t)cx + dx,
                            y, cb, cr);
      }
    }
  }
}

static struct image_t *danger_detector_cb(struct image_t *img,
                                          uint8_t camera_id __attribute__((unused)))
{
  if (!img || !img->buf) { return img; }

  uint16_t width  = img->w;
  uint16_t height = img->h;

  /* Left strip: leftmost dd_floor_strip_frac of the image width.
   * The camera is in portrait orientation – the floor is on the left side. */
  uint16_t strip_end = (uint16_t)(width * dd_floor_strip_frac);
  if (strip_end == 0)    { strip_end = 1; }
  if (strip_end > width) { strip_end = width; }

  uint8_t *buf = (uint8_t *)img->buf;

  uint32_t non_green[DANGER_DETECTOR_NUM_ZONES];
  uint32_t total[DANGER_DETECTOR_NUM_ZONES];
  memset(non_green, 0, sizeof(non_green));
  memset(total,     0, sizeof(total));

  for (uint16_t row = 0; row < height; row++) {
    /* Map row -> zone index (0 = top band, NUM_ZONES-1 = bottom band). */
    uint8_t zone = cv_grid_zone_from_row(row, height, DANGER_DETECTOR_NUM_ZONES);

    for (uint16_t x = 0; x < strip_end; x++) {
      /* YUV422 (UYVY) layout – each macro-pixel pair is [U, Y0, V, Y1].
       * Even x → first pixel of pair; odd x → second pixel of pair. */
      uint8_t y_val, cb_val, cr_val;
      if (x % 2 == 0) {
        cb_val = buf[row * 2u * width + 2u * x];
        y_val  = buf[row * 2u * width + 2u * x + 1];
        cr_val = buf[row * 2u * width + 2u * x + 2];
      } else {
        cb_val = buf[row * 2u * width + 2u * x - 2];
        y_val  = buf[row * 2u * width + 2u * x + 1];
        cr_val = buf[row * 2u * width + 2u * x];
      }

      total[zone]++;
      if (!is_green_yuv(y_val, cb_val, cr_val)) {
        non_green[zone]++;
      }
    }
  }

  /* Compute danger scores and copy to buffer under the mutex. */
  pthread_mutex_lock(&_mutex);
  for (int z = 0; z < DANGER_DETECTOR_NUM_ZONES; z++) {
    if (total[z] == 0) {
      _scores_buf[z] = 0;
    } else {
      _scores_buf[z] = (uint8_t)((non_green[z] * 100u) / total[z]);
    }
  }
  _scores_updated = true;
  pthread_mutex_unlock(&_mutex);

  /* Draw only a per-zone status dot:
   * green = safe zone, red = unsafe zone. */
  if (dd_draw) {
    uint8_t *dbuf = (uint8_t *)img->buf;
    static const uint8_t green[4] = {44, 220,  21, 220};
    static const uint8_t red[4]   = {85, 100, 255, 100};
    uint16_t dot_x = (strip_end + 10u < width) ? (uint16_t)(strip_end + 10u)
                                                : (uint16_t)(width / 2u);

    for (int z = 0; z < DANGER_DETECTOR_NUM_ZONES; z++) {
      uint8_t score  = (total[z] == 0) ? 0u
                       : (uint8_t)((non_green[z] * 100u) / total[z]);
      bool unsafe = (score >= dd_danger_threshold);
      uint16_t y0 = (uint16_t)((uint32_t)height * (uint32_t)z / DANGER_DETECTOR_NUM_ZONES);
      uint16_t y1 = (uint16_t)((uint32_t)height * (uint32_t)(z + 1u) / DANGER_DETECTOR_NUM_ZONES);
      uint16_t yc = (uint16_t)((y0 + y1) / 2u);
      const uint8_t *c = unsafe ? red : green;
      dd_draw_dot(dbuf, width, height, dot_x, yc, 3u, 200u, c[0], c[2]);
    }
  }

  return img;
}

/* -------------------------------------------------------------------------
 * Module API
 * ---------------------------------------------------------------------- */

void danger_detector_init(void)
{
  memset(danger_scores, 0, sizeof(danger_scores));
  memset(_scores_buf,   0, sizeof(_scores_buf));
  _scores_updated = false;
  pthread_mutex_init(&_mutex, NULL);

#ifdef DANGER_DETECTOR_CAMERA
  cv_add_to_device(&DANGER_DETECTOR_CAMERA, danger_detector_cb,
                   DANGER_DETECTOR_FPS, 0);
#endif
}

void danger_detector_periodic(void)
{
  uint8_t danger_blocked[DANGER_DETECTOR_NUM_ZONES];

  /* Copy camera-thread results to the public array. */
  pthread_mutex_lock(&_mutex);
  if (_scores_updated) {
    memcpy(danger_scores, _scores_buf, sizeof(danger_scores));
    _scores_updated = false;
  }
  pthread_mutex_unlock(&_mutex);

  /* Convert score to blocked/free flags for ABI consumers. */
  for (int z = 0; z < DANGER_DETECTOR_NUM_ZONES; z++) {
    danger_blocked[z] = (danger_scores[z] >= dd_danger_threshold) ? 1u : 0u;
  }

  /* Broadcast blocked flags to other modules via ABI (consumed by orange_avoider). */
  AbiSendMsgPAYLOAD_DATA(DANGER_DETECTOR_SENDER_ID, 0, 1,
                          DANGER_DETECTOR_NUM_ZONES, danger_blocked);

  /* Print scores to stderr for debugging. */
#ifdef DANGER_DETECTOR_VERBOSE
  fprintf(stderr, "[danger_detector] zones:");
  for (int z = 0; z < DANGER_DETECTOR_NUM_ZONES; z++) {
    uint8_t s = danger_scores[z];
    const char *level = (s < 25) ? "low" :
                        (s < 50) ? "med" :
                        (s < 75) ? "high" : "CRIT";
    fprintf(stderr, " %3u(%s,%u)", s, level, danger_blocked[z]);
  }
  fprintf(stderr, "\n");
#endif
}

int danger_detector_get_scores(uint8_t *out)
{
  if (!out) { return -1; }
  memcpy(out, danger_scores, DANGER_DETECTOR_NUM_ZONES * sizeof(uint8_t));

  /* Find safest (lowest danger) zone. */
  int safest = 0;
  for (int z = 1; z < DANGER_DETECTOR_NUM_ZONES; z++) {
    if (out[z] < out[safest]) { safest = z; }
  }
  return safest;
}
