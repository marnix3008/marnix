/*
 * Copyright (C) 2026
 *
 * This file is part of paparazzi
 */
/**
 * @file "modules/gate_detector/gate_detector.c"
 * @author AE4317
 *
 * Blue gate post detector for the TU Delft Cyberzoo.
 *
 * The camera is physically rotated 90° on the Bebop, so the image buffer is
 * in landscape orientation. Gate posts appear as wide, short HORIZONTAL blobs
 * (one near the top of the image, one near the bottom).
 *
 * Key design choices for the Bebop's limited hardware:
 *   - 2x subsampling in both dimensions  →  ~260×120 = 31 200 pixels per frame
 *   - Row histogram only (no mask image stored)
 *   - Integer arithmetic throughout, no floating point in the hot path
 *   - Static arrays, no dynamic allocation
 */

#include "modules/gate_detector/gate_detector.h"
#include "modules/computer_vision/cv.h"
#include "modules/computer_vision/lib/vision/image.h"
#include "std.h"

#include <string.h>
#include <stdio.h>
#include <pthread.h>

/* -------------------------------------------------------------------------
 * Compile-time defaults (can be overridden in the airframe file)
 * ---------------------------------------------------------------------- */

#ifndef GATE_DETECTOR_CAMERA
#define GATE_DETECTOR_CAMERA front_camera
#endif

#ifndef GATE_DETECTOR_FPS
#define GATE_DETECTOR_FPS 0   /* 0 = run at camera rate */
#endif

/* Maximum sub-sampled dimension (covers images up to 640px in either axis). */
#define GD_MAX_SSIZE 320

/* -------------------------------------------------------------------------
 * Tunable settings (accessible via GCS)
 * ---------------------------------------------------------------------- */

/* Blue YUV defaults derived from the gate-post RGB samples:
 *   RGB (54,115,196) → Y≈106, Cb≈179, Cr≈91                             */
uint8_t gd_y_min          = 50;
uint8_t gd_y_max          = 160;
uint8_t gd_cb_min         = 130;
uint8_t gd_cb_max         = 235;
uint8_t gd_cr_min         = 55;
uint8_t gd_cr_max         = 125;

uint8_t gd_min_ratio_x10  = 18;   /* min width/height ratio × 10 (e.g. 18 = 1.8) */
uint8_t gd_min_col_fill   = 12;   /* ≥ 12 % of sub-sampled columns must be blue in a post row */
uint8_t gd_min_post_sep   = 5;    /* posts must be ≥ 5 sub-sampled rows apart */
bool    gd_draw           = true;

/* -------------------------------------------------------------------------
 * Public result variables
 * ---------------------------------------------------------------------- */

bool     gd_gate_detected = false;
int16_t  gd_gate_x        = 0;
int16_t  gd_gate_y        = 0;
uint16_t gd_gate_width    = 0;

/* Internal copy protected by mutex. */
static bool     _det;
static int16_t  _gx, _gy;
static uint16_t _gw;
static pthread_mutex_t _mutex;

/* -------------------------------------------------------------------------
 * Blue pixel classifier (inline for speed)
 * ---------------------------------------------------------------------- */

static inline bool is_blue_yuv(uint8_t y, uint8_t cb, uint8_t cr)
{
  return (y  >= gd_y_min)  && (y  <= gd_y_max)  &&
         (cb >= gd_cb_min) && (cb <= gd_cb_max)  &&
         (cr >= gd_cr_min) && (cr <= gd_cr_max);
}

/* -------------------------------------------------------------------------
 * Camera callback – runs in the video thread
 * ---------------------------------------------------------------------- */

static struct image_t *gate_detector_cb(struct image_t *img,
                                        uint8_t camera_id __attribute__((unused)))
{
  if (!img || !img->buf) { return img; }

  const uint16_t W = img->w;
  const uint16_t H = img->h;
  const uint8_t *buf = (const uint8_t *)img->buf;

  /* Sub-sampled dimensions (step 2 in each axis). */
  const uint16_t scols = W / 2;
  const uint16_t srows = H / 2;

  if (srows > GD_MAX_SSIZE) { return img; }   /* safety guard */

  /* ---- 1. Build row histogram ---------------------------------------- */
  /* row_count[r] = number of blue pixels in sub-sampled row r.
   * row_xmin/xmax track horizontal span (in original image columns).    */
  static uint16_t row_count[GD_MAX_SSIZE];
  static uint16_t row_xmin[GD_MAX_SSIZE];
  static uint16_t row_xmax[GD_MAX_SSIZE];

  memset(row_count, 0, srows * sizeof(uint16_t));
  for (uint16_t r = 0; r < srows; r++) {
    row_xmin[r] = W;
    row_xmax[r] = 0;
  }

  /* Process only even rows and even columns (2× sub-sampling).
   * For even column x = 2c, UYVY layout:
   *   buf[ row*2W + 4c ]     = U (Cb)
   *   buf[ row*2W + 4c + 1 ] = Y0
   *   buf[ row*2W + 4c + 2 ] = V (Cr)                                   */
  for (uint16_t row = 0; row < H; row += 2) {
    const uint16_t r        = row / 2;
    const uint32_t row_base = (uint32_t)row * 2u * W;

    for (uint16_t c = 0; c < scols; c++) {
      const uint32_t base = row_base + (uint32_t)c * 4u;
      const uint8_t  cb   = buf[base];
      const uint8_t  y    = buf[base + 1];
      const uint8_t  cr   = buf[base + 2];

      if (is_blue_yuv(y, cb, cr)) {
        const uint16_t x = c * 2u;   /* original pixel column */
        row_count[r]++;
        if (x < row_xmin[r]) { row_xmin[r] = x; }
        if (x > row_xmax[r]) { row_xmax[r] = x; }
      }
    }
  }

  /* ---- 2. Threshold rows -------------------------------------------- */
  /* A row is "active" if it has enough blue pixels across its width.    */
  const uint16_t fill_thresh = (uint16_t)((uint32_t)scols * gd_min_col_fill / 100u);
  const uint16_t min_th      = (fill_thresh < 2) ? 2u : fill_thresh;

  /* ---- 3. Find runs of active rows ------------------------------------ */
  /* We keep the two runs with the most total blue pixels.               */
  typedef struct {
    uint16_t start;   /* first sub-sampled row in run  */
    uint16_t end;     /* last  sub-sampled row in run  */
    uint32_t score;   /* total blue pixels in run      */
    uint16_t xmin;    /* leftmost  blue pixel seen     */
    uint16_t xmax;    /* rightmost blue pixel seen     */
  } GdRun;

  GdRun best[2];
  memset(best, 0, sizeof(best));
  best[0].score = 0; best[1].score = 0;

  bool in_run = false;
  GdRun cur;
  memset(&cur, 0, sizeof(cur));
  cur.xmin = W; cur.xmax = 0;

  for (uint16_t r = 0; r <= srows; r++) {
    bool active = (r < srows) && (row_count[r] >= min_th);

    if (active && !in_run) {
      in_run    = true;
      cur.start = r;
      cur.end   = r;
      cur.score = row_count[r];
      cur.xmin  = row_xmin[r];
      cur.xmax  = row_xmax[r];

    } else if (active && in_run) {
      cur.end    = r;
      cur.score += row_count[r];
      if (row_xmin[r] < cur.xmin) { cur.xmin = row_xmin[r]; }
      if (row_xmax[r] > cur.xmax) { cur.xmax = row_xmax[r]; }

    } else if (!active && in_run) {
      in_run = false;
      if (cur.score > best[0].score) {
        best[1] = best[0];
        best[0] = cur;
      } else if (cur.score > best[1].score) {
        best[1] = cur;
      }
      cur.xmin = W; cur.xmax = 0; cur.score = 0;
    }
  }

  /* ---- 4. Validate both posts (width/height ratio) ------------------- */
  bool det = false;
  int16_t gate_x = 0, gate_y = 0;
  uint16_t gate_w = 0;

  if (best[0].score > 0 && best[1].score > 0) {
    /* Sort so that post0 is top, post1 is bottom. */
    GdRun *top = (best[0].start < best[1].start) ? &best[0] : &best[1];
    GdRun *bot = (best[0].start < best[1].start) ? &best[1] : &best[0];

    /* Minimum post separation check (in sub-sampled rows). */
    if ((bot->start - top->end) >= gd_min_post_sep) {

      /* Width/height ratio check for each post (integer, ×10).
       * Run height in original pixels = (end - start + 1) * 2.
       * Width in original pixels      = xmax - xmin.              */
      uint16_t t_h = (uint16_t)((top->end - top->start + 1u) * 2u);
      uint16_t t_w = (top->xmax > top->xmin) ? (top->xmax - top->xmin) : 0u;
      uint16_t b_h = (uint16_t)((bot->end - bot->start + 1u) * 2u);
      uint16_t b_w = (bot->xmax > bot->xmin) ? (bot->xmax - bot->xmin) : 0u;

      /* ratio_x10 = w * 10 / h — avoids float */
      bool t_ok = (t_h > 0) && ((uint32_t)t_w * 10u / t_h >= gd_min_ratio_x10);
      bool b_ok = (b_h > 0) && ((uint32_t)b_w * 10u / b_h >= gd_min_ratio_x10);

      if (t_ok && b_ok) {
        det = true;

        /* Post centres in original pixel coordinates.
         * Sub-sampled row r → original row 2r, so centre = start + end. */
        uint16_t top_cy = top->start + top->end;   /* original row centre */
        uint16_t bot_cy = bot->start + bot->end;
        uint16_t top_cx = (top->xmin + top->xmax) / 2u;
        uint16_t bot_cx = (bot->xmin + bot->xmax) / 2u;

        gate_x = (int16_t)((top_cx + bot_cx) / 2u);
        gate_y = (int16_t)((top_cy + bot_cy) / 2u);
        gate_w = (uint16_t)(bot_cy - top_cy);   /* vertical gate aperture */

        /* ---- 5. Optional drawing ------------------------------------ */
        if (gd_draw) {
          /* Top post:    horizontal cyan  line at top_cy
           * Bottom post: horizontal green line at bot_cy
           * Gate centre: vertical white line at gate_x
           *
           * YUV colour values {U, Y, V, Y}:
           *   cyan  = {166, 200,  16, 200}
           *   green = { 44, 200,  21, 200}
           *   white = {127, 255, 127, 255}                              */
          static const uint8_t col_cyan[4]  = {166, 200,  16, 200};
          static const uint8_t col_green[4] = { 44, 200,  21, 200};
          static const uint8_t col_white[4] = {127, 255, 127, 255};

          struct point_t p1, p2;

          /* Top post — horizontal cyan line */
          p1.x = top->xmin; p1.y = top_cy;
          p2.x = top->xmax; p2.y = top_cy;
          image_draw_line_color(img, &p1, &p2, col_cyan);

          /* Bottom post — horizontal green line */
          p1.x = bot->xmin; p1.y = bot_cy;
          p2.x = bot->xmax; p2.y = bot_cy;
          image_draw_line_color(img, &p1, &p2, col_green);

          /* Gate centre — vertical white line */
          p1.x = (uint16_t)gate_x; p1.y = top_cy;
          p2.x = (uint16_t)gate_x; p2.y = bot_cy;
          image_draw_line_color(img, &p1, &p2, col_white);
        }
      }
    }
  }

  /* ---- 6. Publish results under mutex --------------------------------- */
  pthread_mutex_lock(&_mutex);
  _det = det;
  _gx  = gate_x;
  _gy  = gate_y;
  _gw  = gate_w;
  pthread_mutex_unlock(&_mutex);

  return img;
}

/* -------------------------------------------------------------------------
 * Module API
 * ---------------------------------------------------------------------- */

void gate_detector_init(void)
{
  gd_gate_detected = false;
  gd_gate_x = 0; gd_gate_y = 0; gd_gate_width = 0;
  _det = false; _gx = 0; _gy = 0; _gw = 0;
  pthread_mutex_init(&_mutex, NULL);

#ifdef GATE_DETECTOR_CAMERA
  cv_add_to_device(&GATE_DETECTOR_CAMERA, gate_detector_cb, GATE_DETECTOR_FPS, 0);
#endif
}

void gate_detector_periodic(void)
{
  pthread_mutex_lock(&_mutex);
  gd_gate_detected = _det;
  gd_gate_x        = _gx;
  gd_gate_y        = _gy;
  gd_gate_width    = _gw;
  pthread_mutex_unlock(&_mutex);

#ifdef GATE_DETECTOR_VERBOSE
  if (gd_gate_detected) {
    fprintf(stderr, "[gate_detector] GATE detected: x=%d y=%d aperture=%u\n",
            gd_gate_x, gd_gate_y, gd_gate_width);
  } else {
    fprintf(stderr, "[gate_detector] no gate\n");
  }
#endif
}
