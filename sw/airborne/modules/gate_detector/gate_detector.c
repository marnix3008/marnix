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
 * The camera is in portrait orientation (width=240, height=520).
 * Gate posts appear as tall, narrow, vertical blue blobs.
 *
 * Key design choices for the Bebop's limited hardware:
 *   - 2x subsampling in both dimensions  →  ~120×260 = 31 200 pixels per frame
 *   - Column histogram only (no mask image stored)
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

/* Maximum sub-sampled image width supported (covers up to 640 px wide images). */
#define GD_MAX_SCOLS 320

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

uint8_t gd_min_ratio_x10  = 18;   /* 1.8 × 10 — same threshold as Python */
uint8_t gd_min_col_fill   = 12;   /* ≥ 12 % of sub-sampled rows must be blue */
uint8_t gd_min_post_sep   = 10;   /* posts must be ≥ 10 sub-sampled cols apart */
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

  /* Sub-sampled dimensions. */
  const uint16_t scols = W / 2;
  const uint16_t srows = H / 2;

  if (scols > GD_MAX_SCOLS) { return img; }   /* safety guard */

  /* ---- 1. Build column histogram ------------------------------------ */
  /* col_count[c] = number of blue pixels in sub-sampled column c.
   * col_ymin/ymax track vertical span (in original image rows).        */
  static uint16_t col_count[GD_MAX_SCOLS];
  static uint16_t col_ymin[GD_MAX_SCOLS];
  static uint16_t col_ymax[GD_MAX_SCOLS];

  memset(col_count, 0, scols * sizeof(uint16_t));
  for (uint16_t c = 0; c < scols; c++) {
    col_ymin[c] = H;
    col_ymax[c] = 0;
  }

  /* Process only even rows and even columns (2× sub-sampling).
   * For an even column x = 2c, the UYVY macro-pixel layout gives:
   *   buf[ row*2W + 4c ]     = U (Cb)
   *   buf[ row*2W + 4c + 1 ] = Y0
   *   buf[ row*2W + 4c + 2 ] = V (Cr)                                 */
  for (uint16_t row = 0; row < H; row += 2) {
    const uint32_t row_base = (uint32_t)row * 2u * W;
    for (uint16_t c = 0; c < scols; c++) {
      const uint32_t base = row_base + (uint32_t)c * 4u;
      const uint8_t  cb   = buf[base];
      const uint8_t  y    = buf[base + 1];
      const uint8_t  cr   = buf[base + 2];

      if (is_blue_yuv(y, cb, cr)) {
        col_count[c]++;
        if (row < col_ymin[c]) { col_ymin[c] = row; }
        if (row > col_ymax[c]) { col_ymax[c] = row; }
      }
    }
  }

  /* ---- 2. Threshold columns ----------------------------------------- */
  /* A column is "active" (part of a post) if it has enough blue pixels. */
  const uint16_t fill_thresh = (uint16_t)((uint32_t)srows * gd_min_col_fill / 100u);
  const uint16_t min_th      = (fill_thresh < 2) ? 2u : fill_thresh;

  /* ---- 3. Find runs of active columns --------------------------------- */
  /* We keep the two runs with the most total blue pixels.               */
  typedef struct {
    uint16_t start;   /* first sub-sampled col in run  */
    uint16_t end;     /* last  sub-sampled col in run  */
    uint32_t score;   /* total blue pixels in run      */
    uint16_t ymin;    /* min original row seen         */
    uint16_t ymax;    /* max original row seen         */
  } GdRun;

  GdRun best[2];
  memset(best, 0, sizeof(best));
  best[0].score = 0; best[1].score = 0;

  bool in_run = false;
  GdRun cur;
  memset(&cur, 0, sizeof(cur));
  cur.ymin = H; cur.ymax = 0;

  for (uint16_t c = 0; c <= scols; c++) {
    bool active = (c < scols) && (col_count[c] >= min_th);

    if (active && !in_run) {
      /* Start new run. */
      in_run = true;
      cur.start = c;
      cur.end   = c;
      cur.score = col_count[c];
      cur.ymin  = col_ymin[c];
      cur.ymax  = col_ymax[c];

    } else if (active && in_run) {
      /* Extend run. */
      cur.end    = c;
      cur.score += col_count[c];
      if (col_ymin[c] < cur.ymin) { cur.ymin = col_ymin[c]; }
      if (col_ymax[c] > cur.ymax) { cur.ymax = col_ymax[c]; }

    } else if (!active && in_run) {
      /* End of run: check if it beats one of our best two. */
      in_run = false;
      if (cur.score > best[0].score) {
        best[1] = best[0];
        best[0] = cur;
      } else if (cur.score > best[1].score) {
        best[1] = cur;
      }
      cur.ymin = H; cur.ymax = 0; cur.score = 0;
    }
  }

  /* ---- 4. Validate both posts (height/width ratio) ------------------- */
  bool det = false;
  int16_t gate_x = 0, gate_y = 0;
  uint16_t gate_w = 0;

  if (best[0].score > 0 && best[1].score > 0) {
    /* Sort so that post0 is left, post1 is right. */
    GdRun *left  = (best[0].start < best[1].start) ? &best[0] : &best[1];
    GdRun *right = (best[0].start < best[1].start) ? &best[1] : &best[0];

    /* Minimum post separation check. */
    if ((right->start - left->end) >= gd_min_post_sep) {

      /* Height/width ratio check for each post (integer, ×10).
       * Run width in original pixels = (end - start + 1) * 2.
       * Span in original pixels      = ymax - ymin.             */
      uint16_t lw  = (uint16_t)((left->end  - left->start  + 1u) * 2u);
      uint16_t lh  = (left->ymax  > left->ymin)  ? (left->ymax  - left->ymin)  : 0u;
      uint16_t rw  = (uint16_t)((right->end - right->start + 1u) * 2u);
      uint16_t rh  = (right->ymax > right->ymin) ? (right->ymax - right->ymin) : 0u;

      /* ratio_x10 = h * 10 / w  — avoids float */
      bool l_ok = (lw > 0) && ((uint32_t)lh * 10u / lw >= gd_min_ratio_x10);
      bool r_ok = (rw > 0) && ((uint32_t)rh * 10u / rw >= gd_min_ratio_x10);

      if (l_ok && r_ok) {
        det = true;

        /* Gate centre x: midpoint between the two post centres (in original px). */
        uint16_t l_cx = (uint16_t)((left->start  + left->end  + 1u) * 1u); /* = (sc+.5)*2, approx */
        uint16_t r_cx = (uint16_t)((right->start + right->end + 1u) * 1u);
        gate_x = (int16_t)((l_cx + r_cx) / 2);
        gate_y = (int16_t)(((left->ymin + left->ymax) / 2u +
                             (right->ymin + right->ymax) / 2u) / 2u);
        gate_w = (uint16_t)(r_cx - l_cx);

        /* ---- 5. Optional drawing ------------------------------------ */
        if (gd_draw) {
          uint8_t *dbuf = (uint8_t *)img->buf;

          /* Draw left post: vertical cyan line at post centre column.
           * Draw right post: vertical green line at post centre column.
           * Draw gate centre: horizontal white line at gate_y.          *
           *
           * YUV colour values {U, V}:
           *   cyan  = { 166, 16 }
           *   green = {  44, 21 }
           *   white = { 127,127 }                                       */
          static const uint8_t col_cyan[4]  = {166, 200,  16, 200};
          static const uint8_t col_green[4] = { 44, 200,  21, 200};
          static const uint8_t col_white[4] = {127, 255, 127, 255};

          /* Vertical line helpers – draw at original column x_px. */
          struct point_t p1, p2;

          /* Left post */
          p1.x = (uint16_t)(l_cx); p1.y = left->ymin;
          p2.x = (uint16_t)(l_cx); p2.y = left->ymax;
          image_draw_line_color(img, &p1, &p2, col_cyan);

          /* Right post */
          p1.x = (uint16_t)(r_cx); p1.y = right->ymin;
          p2.x = (uint16_t)(r_cx); p2.y = right->ymax;
          image_draw_line_color(img, &p1, &p2, col_green);

          /* Horizontal gate centre line */
          p1.x = (uint16_t)(l_cx); p1.y = (uint16_t)gate_y;
          p2.x = (uint16_t)(r_cx); p2.y = (uint16_t)gate_y;
          image_draw_line_color(img, &p1, &p2, col_white);

          (void)dbuf; /* suppress unused warning if draw helpers cover everything */
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
    fprintf(stderr, "[gate_detector] GATE detected: x=%d y=%d w=%u\n",
            gd_gate_x, gd_gate_y, gd_gate_width);
  } else {
    fprintf(stderr, "[gate_detector] no gate\n");
  }
#endif
}
