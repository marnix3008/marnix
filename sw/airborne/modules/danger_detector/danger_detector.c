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
 * The cyberzoo floor is green. This module processes the bottom strip of each
 * camera frame, divides it into columns, and scores each column by the fraction
 * of non-green pixels it contains (0 = safe, 100 = blocked).
 *
 * The green-floor detection uses a simple YUV422 threshold:
 *   - Y  (luminance): dd_lum_min … dd_lum_max       (avoid very dark / overexposed)
 *   - Cb (U, blue chroma): below dd_cb_max           (green has low blue chroma)
 *   - Cr (V, red chroma):  below dd_cr_max           (green has low red chroma)
 *
 * Results are available via the global danger_scores[] array and the
 * danger_detector_get_scores() helper.
 *
 * NOTE: This module does NOT yet steer the drone – it only produces scores
 * that other modules can read.
 */

#include "modules/danger_detector/danger_detector.h"
#include "modules/computer_vision/cv.h"
#include "modules/computer_vision/lib/vision/image.h"
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
uint8_t dd_cr_max         = 125;  /* max Cr (V) for a pixel to be "green"   */
float   dd_floor_strip_frac = 0.15f; /* bottom fraction of image to inspect */
bool    dd_draw           = true; /* overlay column lines and danger colours */

/* -------------------------------------------------------------------------
 * Public result array + mutex
 * ---------------------------------------------------------------------- */

uint8_t danger_scores[DANGER_DETECTOR_NUM_COLUMNS];

/** Internal (camera-thread) copy, protected by mutex. */
static uint8_t _scores_buf[DANGER_DETECTOR_NUM_COLUMNS];
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

static struct image_t *danger_detector_cb(struct image_t *img,
                                          uint8_t camera_id __attribute__((unused)))
{
  if (!img || !img->buf) { return img; }

  uint16_t width  = img->w;
  uint16_t height = img->h;

  /* Determine the start row of the bottom strip. */
  uint16_t strip_start = (uint16_t)(height * (1.0f - dd_floor_strip_frac));
  if (strip_start >= height) { strip_start = height - 1; }

  uint16_t strip_rows = height - strip_start;
  if (strip_rows == 0) { return img; }

  uint8_t *buf = (uint8_t *)img->buf;

  /* Column boundaries. */
  uint16_t col_width = width / DANGER_DETECTOR_NUM_COLUMNS;

  uint32_t non_green[DANGER_DETECTOR_NUM_COLUMNS];
  uint32_t total[DANGER_DETECTOR_NUM_COLUMNS];
  memset(non_green, 0, sizeof(non_green));
  memset(total,     0, sizeof(total));

  for (uint16_t row = strip_start; row < height; row++) {
    for (uint16_t x = 0; x < width; x++) {

      /* YUV422 (UYVY) layout – see cv_detect_color_object.c for reference. */
      uint8_t y_val, cb_val, cr_val;
      if (x % 2 == 0) {
        cb_val = buf[row * 2 * width + 2 * x];
        y_val  = buf[row * 2 * width + 2 * x + 1];
        cr_val = buf[row * 2 * width + 2 * x + 2];
      } else {
        cb_val = buf[row * 2 * width + 2 * x - 2];
        y_val  = buf[row * 2 * width + 2 * x + 1];
        cr_val = buf[row * 2 * width + 2 * x];
      }

      /* Map pixel to column index. */
      int col = (int)(x * DANGER_DETECTOR_NUM_COLUMNS / width);
      if (col >= DANGER_DETECTOR_NUM_COLUMNS) { col = DANGER_DETECTOR_NUM_COLUMNS - 1; }

      total[col]++;
      if (!is_green_yuv(y_val, cb_val, cr_val)) {
        non_green[col]++;
      }
    }
  }

  /* Compute danger scores and copy to buffer under the mutex. */
  pthread_mutex_lock(&_mutex);
  for (int c = 0; c < DANGER_DETECTOR_NUM_COLUMNS; c++) {
    if (total[c] == 0) {
      _scores_buf[c] = 0;
    } else {
      _scores_buf[c] = (uint8_t)((non_green[c] * 100u) / total[c]);
    }
  }
  _scores_updated = true;
  pthread_mutex_unlock(&_mutex);

  /* ------------------------------------------------------------------
   * Optional drawing: colour the bottom strip and draw column borders.
   *
   * YUV422 (UYVY) colour format for image_draw_line_color: {U, Y1, V, Y2}
   *   white  = {127, 255, 127, 255}
   *   green  = { 44, 150,  21, 150}   safe    (score <33)
   *   yellow = {  0, 226, 149, 226}   medium  (score 33-65)
   *   red    = { 85,  76, 255,  76}   danger  (score >=66)
   * ---------------------------------------------------------------- */
  if (dd_draw) {
    uint8_t *dbuf = (uint8_t *)img->buf;

    /* 1. Recolour the chroma (U/V) of each pixel in the bottom strip
          based on its column's danger score. Y (brightness) is kept
          untouched so the original image stays visible. */
    for (uint16_t row = strip_start; row < height; row++) {
      for (uint16_t x = 0; x < width; x += 2) {  /* step by 2: one UYVY macro-pixel */
        int col = (int)(x * DANGER_DETECTOR_NUM_COLUMNS / width);
        if (col >= DANGER_DETECTOR_NUM_COLUMNS) { col = DANGER_DETECTOR_NUM_COLUMNS - 1; }

        uint8_t score = (total[col] == 0) ? 0
                        : (uint8_t)((non_green[col] * 100u) / total[col]);

        uint8_t u_tint, v_tint;
        if (score < 33) {
          u_tint = 44;  v_tint = 21;   /* green  */
        } else if (score < 66) {
          u_tint = 0;   v_tint = 149;  /* yellow */
        } else {
          u_tint = 85;  v_tint = 255;  /* red    */
        }

        /* UYVY layout: [U, Y1, V, Y2] per pair of pixels */
        uint32_t base = (uint32_t)row * 2u * width + 2u * x;
        dbuf[base + 0] = u_tint;   /* U  */
        /* dbuf[base + 1] = Y1  – keep original */
        if (x + 1 < width) {
          dbuf[base + 2] = v_tint; /* V  */
          /* dbuf[base + 3] = Y2 – keep original */
        }
      }
    }

    /* 2. Draw a full-height white vertical line at each column border. */
    static const uint8_t white[4] = {127, 255, 127, 255};
    for (int c = 1; c < DANGER_DETECTOR_NUM_COLUMNS; c++) {
      uint16_t x_line = (uint16_t)((uint32_t)width * (uint32_t)c / DANGER_DETECTOR_NUM_COLUMNS);
      struct point_t from = {x_line, 0,          0, 0, 0};
      struct point_t to   = {x_line, height - 1, 0, 0, 0};
      image_draw_line_color(img, &from, &to, white);
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
  /* Copy camera-thread results to the public array. */
  pthread_mutex_lock(&_mutex);
  if (_scores_updated) {
    memcpy(danger_scores, _scores_buf, sizeof(danger_scores));
    _scores_updated = false;
  }
  pthread_mutex_unlock(&_mutex);

  /* Print scores to stderr for debugging. */
#ifdef DANGER_DETECTOR_VERBOSE
  fprintf(stderr, "[danger_detector] scores:");
  for (int c = 0; c < DANGER_DETECTOR_NUM_COLUMNS; c++) {
    fprintf(stderr, " %3u", danger_scores[c]);
  }
  fprintf(stderr, "\n");
#endif
}

int danger_detector_get_scores(uint8_t *out)
{
  if (!out) { return -1; }
  memcpy(out, danger_scores, DANGER_DETECTOR_NUM_COLUMNS * sizeof(uint8_t));

  /* Find safest (lowest danger) column. */
  int safest = 0;
  for (int c = 1; c < DANGER_DETECTOR_NUM_COLUMNS; c++) {
    if (out[c] < out[safest]) { safest = c; }
  }
  return safest;
}
