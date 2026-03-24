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

/* -------------------------------------------------------------------------
 * 7-segment digit overlay helpers
 *
 * Segment bit assignments:
 *   0x01 = a (top)           0x02 = b (upper-right)
 *   0x04 = c (lower-right)   0x08 = d (bottom)
 *   0x10 = e (lower-left)    0x20 = f (upper-left)
 *   0x40 = g (middle)
 * ---------------------------------------------------------------------- */

static const uint8_t _seg7[10] = {
  0x3F, /* 0: abcdef  */  0x06, /* 1: bc      */  0x5B, /* 2: abdeg   */
  0x4F, /* 3: abcdg   */  0x66, /* 4: bcfg    */  0x6D, /* 5: acdfg   */
  0x7D, /* 6: acdefg  */  0x07, /* 7: abc     */  0x7F, /* 8: all     */
  0x6F, /* 9: abcdfg  */
};

#define DD_DIGIT_W   8   /* pixels wide per digit  */
#define DD_DIGIT_H   12  /* pixels tall per digit  */
#define DD_DIGIT_GAP 2   /* gap between digits      */

static void dd_draw_digit(struct image_t *img, int x0, int y0,
                          uint8_t d, const uint8_t *color)
{
  if (d > 9) { return; }
  uint8_t s  = _seg7[d];
  int     Hh = DD_DIGIT_H / 2;
  struct point_t p1, p2;

#define _SEG(ax, ay, bx, by) \
  p1.x = (uint16_t)(ax); p1.y = (uint16_t)(ay); \
  p2.x = (uint16_t)(bx); p2.y = (uint16_t)(by); \
  image_draw_line_color(img, &p1, &p2, color)

  if (s & 0x01) { _SEG(x0,            y0,             x0+DD_DIGIT_W, y0            ); } /* a top          */
  if (s & 0x02) { _SEG(x0+DD_DIGIT_W, y0,             x0+DD_DIGIT_W, y0+Hh         ); } /* b upper-right  */
  if (s & 0x04) { _SEG(x0+DD_DIGIT_W, y0+Hh,          x0+DD_DIGIT_W, y0+DD_DIGIT_H ); } /* c lower-right  */
  if (s & 0x08) { _SEG(x0,            y0+DD_DIGIT_H,  x0+DD_DIGIT_W, y0+DD_DIGIT_H ); } /* d bottom       */
  if (s & 0x10) { _SEG(x0,            y0+Hh,          x0,            y0+DD_DIGIT_H ); } /* e lower-left   */
  if (s & 0x20) { _SEG(x0,            y0,             x0,            y0+Hh         ); } /* f upper-left   */
  if (s & 0x40) { _SEG(x0,            y0+Hh,          x0+DD_DIGIT_W, y0+Hh         ); } /* g middle       */

#undef _SEG
}

static void dd_draw_number(struct image_t *img, int x0, int y0,
                           uint8_t value, const uint8_t *color)
{
  uint8_t digits[3];
  int n = 0;
  if (value >= 100) {
    digits[n++] = 1; digits[n++] = 0; digits[n++] = 0;
  } else if (value >= 10) {
    digits[n++] = value / 10u;
    digits[n++] = value % 10u;
  } else {
    digits[n++] = value;
  }
  for (int i = 0; i < n; i++) {
    dd_draw_digit(img, x0 + i * (DD_DIGIT_W + DD_DIGIT_GAP), y0, digits[i], color);
  }
}

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

  /* ------------------------------------------------------------------
   * Optional drawing:
   *   1. Per-pixel tint in the left strip:
   *        green tint  = safe floor pixel (is_green == true)
   *        red tint    = obstacle pixel   (is_green == false)
   *   2. Full-width white horizontal lines at zone boundaries.
   *
   * YUV422 colour values  {U, V}:
   *   green  = {44,  21}   (safe floor)
   *   red    = {85, 255}   (obstacle)
   *   white  = {127, 127}  (separator line)
   * ---------------------------------------------------------------- */
  if (dd_draw) {
    uint8_t *dbuf = (uint8_t *)img->buf;

    /* 1. Tint the left strip pixel-by-pixel.  Step by 2 in x so we process
          one UYVY macro-pixel [U, Y0, V, Y1] at a time. */
    for (uint16_t row = 0; row < height; row++) {
      for (uint16_t x = 0; x < strip_end; x += 2) {
        uint32_t base  = (uint32_t)row * 2u * width + 2u * x;
        uint8_t  cb    = dbuf[base];       /* U (Cb) */
        uint8_t  y0    = dbuf[base + 1];   /* Y of first pixel  */
        uint8_t  cr    = dbuf[base + 2];   /* V (Cr) */

        bool is_green0 = is_green_yuv(y0, cb, cr);

        /* Same U/V slot covers both x and x+1, so one tint per pair. */
        dbuf[base + 0] = is_green0 ? 44u : 85u;   /* U (Cb) */
        if (x + 1 < strip_end) {
          dbuf[base + 2] = is_green0 ? 21u : 255u; /* V (Cr) */
        }
      }
    }

    /* 2. Draw a full-width white horizontal line at each zone boundary. */
    static const uint8_t white[4] = {127, 255, 127, 255};
    cv_grid_draw_horizontal_boundaries(img, DANGER_DETECTOR_NUM_ZONES, white);

    /* 3. Overlay danger score (0-100) centred in each zone, just right of
          the floor strip.  Colour indicates danger level:
            green  = low     (score  0-24)
            yellow = medium  (score 25-49)
            red    = high/critical (score 50-100)            */
    static const uint8_t col_low[4]  = { 44, 220,  21, 220}; /* green  */
    static const uint8_t col_med[4]  = {  0, 230, 149, 230}; /* yellow */
    static const uint8_t col_high[4] = { 85, 100, 255, 100}; /* red    */

    for (int z = 0; z < DANGER_DETECTOR_NUM_ZONES; z++) {
      uint8_t score  = (total[z] == 0) ? 0u
                       : (uint8_t)((non_green[z] * 100u) / total[z]);
      int zone_y0    = (int)cv_grid_zone_boundary_y(height, (uint8_t)z, DANGER_DETECTOR_NUM_ZONES);
      int zone_y1    = (int)cv_grid_zone_boundary_y(height, (uint8_t)(z + 1), DANGER_DETECTOR_NUM_ZONES);
      int label_y    = (zone_y0 + zone_y1) / 2 - DD_DIGIT_H / 2;
      int label_x    = (int)strip_end + 4;

      if (label_y < 0) { label_y = 0; }
      if (label_y + DD_DIGIT_H >= (int)height) { label_y = (int)height - DD_DIGIT_H - 1; }

      const uint8_t *score_color = (score < 25) ? col_low :
                                   (score < 50) ? col_med : col_high;
      dd_draw_number(img, label_x, label_y, score, score_color);
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
