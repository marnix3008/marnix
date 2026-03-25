/*
 * Horizon boundary validity detector.
 *
 * Based on a right-to-left green-run boundary search with local slope checks.
 */

#include "modules/danger_detector/horizon_detector.h"

#include "modules/computer_vision/cv.h"
#include "modules/computer_vision/lib/vision/image.h"
#include "modules/core/abi.h"
#include "std.h"

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>

#ifndef HORIZON_DETECTOR_CAMERA
#define HORIZON_DETECTOR_CAMERA front_camera
#endif

#ifndef HORIZON_DETECTOR_FPS
#define HORIZON_DETECTOR_FPS 0
#endif

#ifndef HORIZON_DETECTOR_DRAW
#define HORIZON_DETECTOR_DRAW 1
#endif

/* Tunables matching the provided Python logic defaults. */
uint8_t hh_row_step                  = 3;
uint8_t hh_min_green_run             = 15;
uint8_t hh_green_min                 = 55;
uint8_t hh_green_max                 = 200;
uint8_t hh_red_min                   = 40;
int8_t  hh_green_minus_blue_min      = 25;
int8_t  hh_green_minus_red_min       = -25;
uint8_t hh_luma_min                  = 40;
uint8_t hh_luma_max                  = 200;
uint8_t hh_cb_max                    = 110;
uint8_t hh_cr_max                    = 130;
float   hh_floor_strip_frac          = 0.25f;
float   hh_invalid_percent_threshold = 15.f;
float   hh_max_slope                 = 2.0f;
float   hh_max_slope_change          = 3.0f;
uint8_t hh_slope_window              = 2;
bool    hh_draw                      = (HORIZON_DETECTOR_DRAW != 0);

uint8_t horizon_invalid_flags[HORIZON_DETECTOR_NUM_ZONES];

static uint8_t _flags_buf[HORIZON_DETECTOR_NUM_ZONES];
static bool _flags_updated = false;
static pthread_mutex_t _mutex;

static inline uint8_t zone_from_row(uint16_t row, uint16_t height)
{
  uint32_t zone = (uint32_t)row * HORIZON_DETECTOR_NUM_ZONES / height;
  if (zone >= HORIZON_DETECTOR_NUM_ZONES) {
    zone = HORIZON_DETECTOR_NUM_ZONES - 1u;
  }
  return (uint8_t)zone;
}

static inline void get_yuv422_pixel(const uint8_t *buf, uint16_t width,
                                    uint16_t row, uint16_t x,
                                    uint8_t *y, uint8_t *cb, uint8_t *cr)
{
  uint32_t base = (uint32_t)row * 2u * width + 2u * x;
  if ((x & 1u) == 0u) {
    *cb = buf[base];
    *y  = buf[base + 1u];
    *cr = buf[base + 2u];
  } else {
    *cb = buf[base - 2u];
    *y  = buf[base + 1u];
    *cr = buf[base];
  }
}

static inline void set_yuv422_pixel(uint8_t *buf, uint16_t width, uint16_t height,
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

static void draw_dot_yuv422(uint8_t *buf, uint16_t width, uint16_t height,
                            uint16_t cx, uint16_t cy, uint8_t radius,
                            uint8_t y, uint8_t cb, uint8_t cr)
{
  int32_t r = (int32_t)radius;
  int32_t r2 = r * r;
  for (int32_t dy = -r; dy <= r; dy++) {
    for (int32_t dx = -r; dx <= r; dx++) {
      if ((dx * dx + dy * dy) <= r2) {
        set_yuv422_pixel(buf, width, height, (int32_t)cy + dy, (int32_t)cx + dx,
                         y, cb, cr);
      }
    }
  }
}

static inline bool is_green_like_pixel(const uint8_t *buf, uint16_t width,
                                       uint16_t row, uint16_t x)
{
  uint8_t y, cb, cr;
  get_yuv422_pixel(buf, width, row, x, &y, &cb, &cr);

  /* Direct YUV thresholds: keep pixels in a green-floor-like region. */
  return (y >= hh_luma_min) &&
         (y <= hh_luma_max) &&
         (cb <= hh_cb_max) &&
         (cr <= hh_cr_max);
}

static struct image_t *horizon_detector_cb(struct image_t *img,
                                           uint8_t camera_id __attribute__((unused)))
{
  if (!img || !img->buf || img->w < 2u || img->h < 2u) {
    return img;
  }

  uint16_t width = img->w;
  uint16_t height = img->h;
  uint8_t *buf = (uint8_t *)img->buf;

  uint8_t row_step = (hh_row_step == 0u) ? 1u : hh_row_step;
  uint16_t sample_count = (uint16_t)((height + row_step - 1u) / row_step);

  int16_t boundary_x[sample_count];
  uint16_t boundary_y[sample_count];
  bool valid_flags[sample_count];
  float slopes[sample_count];
  bool has_slope[sample_count];

  memset(boundary_x, -1, sizeof(boundary_x));
  memset(valid_flags, 0, sizeof(valid_flags));
  memset(slopes, 0, sizeof(slopes));
  memset(has_slope, 0, sizeof(has_slope));

  uint8_t min_run = (hh_min_green_run == 0u) ? 1u : hh_min_green_run;
  if (min_run > width) {
    min_run = (uint8_t)width;
  }

  /* Step 1: right-to-left boundary search on sampled rows. */
  uint16_t si = 0u;
  for (uint16_t y = 0u; y < height; y = (uint16_t)(y + row_step)) {
    boundary_y[si] = y;

    uint8_t run = 0u;
    for (int32_t x = (int32_t)width - 1; x >= 0; x--) {
      bool is_green = is_green_like_pixel(buf, width, y, (uint16_t)x);
      if (is_green) {
        run++;
      } else {
        run = 0u;
      }

      if (run >= min_run) {
        boundary_x[si] = (int16_t)(x + min_run - 1u);
        break;
      }
    }
    si++;
    if (si >= sample_count) {
      break;
    }
  }

  /* Step 2: compute slopes between consecutive valid points. */
  int32_t prev_idx = -1;
  int16_t prev_x = -1;
  uint16_t prev_y = 0u;

  for (uint16_t i = 0u; i < sample_count; i++) {
    if (boundary_x[i] < 0) {
      continue;
    }

    if (prev_idx >= 0) {
      int32_t dy = (int32_t)boundary_y[i] - (int32_t)prev_y;
      if (dy != 0) {
        slopes[i] = ((float)boundary_x[i] - (float)prev_x) / (float)dy;
        has_slope[i] = true;
      }
    }

    prev_idx = (int32_t)i;
    prev_x = boundary_x[i];
    prev_y = boundary_y[i];
  }

  /* Step 3: local slope validity checks. */
  uint8_t valid_point_count = 0u;
  for (uint16_t i = 0u; i < sample_count; i++) {
    if (boundary_x[i] >= 0) {
      valid_point_count++;
    }
  }

  if (valid_point_count >= 3u) {
    uint8_t window = (hh_slope_window == 0u) ? 1u : hh_slope_window;

    for (uint16_t i = 0u; i < sample_count; i++) {
      if (boundary_x[i] < 0) {
        continue;
      }

      int32_t j0 = (int32_t)i - (int32_t)window;
      int32_t j1 = (int32_t)i + (int32_t)window;
      if (j0 < 0) {
        j0 = 0;
      }
      if (j1 >= (int32_t)sample_count) {
        j1 = (int32_t)sample_count - 1;
      }

      float local[32];
      uint8_t n_local = 0u;
      for (int32_t j = j0; j <= j1; j++) {
        if (has_slope[j]) {
          if (n_local < (uint8_t)(sizeof(local) / sizeof(local[0]))) {
            local[n_local++] = slopes[j];
          }
        }
      }

      if (n_local < 2u) {
        continue;
      }

      bool too_steep = false;
      for (uint8_t k = 0u; k < n_local; k++) {
        if (fabsf(local[k]) > hh_max_slope) {
          too_steep = true;
          break;
        }
      }
      if (too_steep) {
        continue;
      }

      bool too_unsmooth = false;
      for (uint8_t k = 1u; k < n_local; k++) {
        float delta = local[k] - local[k - 1u];
        if (fabsf(delta) > hh_max_slope_change) {
          too_unsmooth = true;
          break;
        }
      }
      if (too_unsmooth) {
        continue;
      }

      valid_flags[i] = true;
    }
  }

  /* Step 4: per-zone invalid percentage and flags. */
  uint16_t min_valid_x = (uint16_t)((float)width * hh_floor_strip_frac);
  uint16_t invalid_counts[HORIZON_DETECTOR_NUM_ZONES];
  uint16_t total_counts[HORIZON_DETECTOR_NUM_ZONES];
  uint8_t zone_invalid[HORIZON_DETECTOR_NUM_ZONES];
  memset(invalid_counts, 0, sizeof(invalid_counts));
  memset(total_counts, 0, sizeof(total_counts));
  memset(zone_invalid, 0, sizeof(zone_invalid));

  for (uint16_t i = 0u; i < sample_count; i++) {
    uint8_t zone = zone_from_row(boundary_y[i], height);
    total_counts[zone]++;

    if ((boundary_x[i] < 0) || ((uint16_t)boundary_x[i] < min_valid_x) || (!valid_flags[i])) {
      invalid_counts[zone]++;
    }
  }

  pthread_mutex_lock(&_mutex);
  for (uint8_t z = 0u; z < HORIZON_DETECTOR_NUM_ZONES; z++) {
    float invalid_percent = (total_counts[z] > 0u)
                            ? (100.f * (float)invalid_counts[z] / (float)total_counts[z])
                            : 0.f;
    zone_invalid[z] = (invalid_percent > hh_invalid_percent_threshold) ? 1u : 0u;
    _flags_buf[z] = zone_invalid[z];
  }
  _flags_updated = true;
  pthread_mutex_unlock(&_mutex);

  if (hh_draw) {
    static const uint8_t white[4] = {127, 255, 127, 255};
    static const uint8_t red[4]   = {85,  76, 255, 76};
    static const uint8_t green[4] = {44, 150,  21, 150};

    for (uint8_t z = 1u; z < HORIZON_DETECTOR_NUM_ZONES; z++) {
      uint16_t y_line = (uint16_t)((uint32_t)height * z / HORIZON_DETECTOR_NUM_ZONES);
      struct point_t from = {0u, y_line, 0u, 0u, 0u};
      struct point_t to   = {(uint16_t)(width - 1u), y_line, 0u, 0u, 0u};
      image_draw_line_color(img, &from, &to, white);
    }

    /* Zone-level status dots in the grid: green = safe, red = dangerous. */
    uint16_t dot_x = (width > 20u) ? 10u : (uint16_t)(width / 2u);
    for (uint8_t z = 0u; z < HORIZON_DETECTOR_NUM_ZONES; z++) {
      uint16_t y0 = (uint16_t)((uint32_t)height * z / HORIZON_DETECTOR_NUM_ZONES);
      uint16_t y1 = (uint16_t)((uint32_t)height * (z + 1u) / HORIZON_DETECTOR_NUM_ZONES);
      uint16_t yc = (uint16_t)((y0 + y1) / 2u);

      const uint8_t *c = zone_invalid[z] ? red : green;
      draw_dot_yuv422((uint8_t *)img->buf, width, height, dot_x, yc, 3u,
                      200u, c[0], c[2]);
    }

    for (uint16_t i = 0u; i < sample_count; i++) {
      if (boundary_x[i] < 0) {
        continue;
      }

      uint16_t x = (uint16_t)boundary_x[i];
      uint16_t y = boundary_y[i];
      uint8_t *pix = (uint8_t *)img->buf;
      uint16_t x_pair = (uint16_t)(x & ~1u);
      uint32_t base = (uint32_t)y * 2u * width + 2u * x_pair;
      const uint8_t *c = valid_flags[i] ? green : red;

      pix[base + 0u] = c[0];
      pix[base + 2u] = c[2];
    }
  }

  return img;
}

void horizon_detector_init(void)
{
  memset(horizon_invalid_flags, 0, sizeof(horizon_invalid_flags));
  memset(_flags_buf, 0, sizeof(_flags_buf));
  _flags_updated = false;
  pthread_mutex_init(&_mutex, NULL);

#ifdef HORIZON_DETECTOR_CAMERA
  cv_add_to_device(&HORIZON_DETECTOR_CAMERA, horizon_detector_cb,
                   HORIZON_DETECTOR_FPS, 0);
#endif
}

void horizon_detector_periodic(void)
{
  pthread_mutex_lock(&_mutex);
  if (_flags_updated) {
    memcpy(horizon_invalid_flags, _flags_buf, sizeof(horizon_invalid_flags));
    _flags_updated = false;
  }
  pthread_mutex_unlock(&_mutex);

  AbiSendMsgPAYLOAD_DATA(HORIZON_DETECTOR_SENDER_ID, 0, HORIZON_DETECTOR_DATA_TYPE,
                         HORIZON_DETECTOR_NUM_ZONES, horizon_invalid_flags);

#ifdef HORIZON_DETECTOR_VERBOSE
  fprintf(stderr, "[horizon_detector] invalid flags:");
  for (uint8_t z = 0u; z < HORIZON_DETECTOR_NUM_ZONES; z++) {
    fprintf(stderr, " %u", horizon_invalid_flags[z]);
  }
  fprintf(stderr, "\n");
#endif
}

int horizon_detector_get_flags(uint8_t *out)
{
  if (!out) {
    return -1;
  }

  memcpy(out, horizon_invalid_flags,
         HORIZON_DETECTOR_NUM_ZONES * sizeof(uint8_t));
  return 0;
}
