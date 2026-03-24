/*
 * Reusable helpers for detector zone-grid computations and overlays.
 */

#ifndef DD_GRID_UTILS_H
#define DD_GRID_UTILS_H

#include "modules/computer_vision/lib/vision/image.h"

#include <stdint.h>

static inline uint8_t cv_grid_zone_from_row(uint16_t row, uint16_t height, uint8_t num_zones)
{
  if (height == 0u || num_zones == 0u) {
    return 0u;
  }

  uint32_t zone = (uint32_t)row * (uint32_t)num_zones / (uint32_t)height;
  if (zone >= (uint32_t)num_zones) {
    zone = (uint32_t)num_zones - 1u;
  }
  return (uint8_t)zone;
}

static inline uint16_t cv_grid_zone_boundary_y(uint16_t height, uint8_t zone_idx, uint8_t num_zones)
{
  if (num_zones == 0u) {
    return 0u;
  }
  return (uint16_t)((uint32_t)height * (uint32_t)zone_idx / (uint32_t)num_zones);
}

static inline void cv_grid_draw_horizontal_boundaries(struct image_t *img,
                                                      uint8_t num_zones,
                                                      const uint8_t *color)
{
  if (!img || !color || num_zones < 2u) {
    return;
  }

  for (uint8_t z = 1u; z < num_zones; z++) {
    uint16_t y_line = cv_grid_zone_boundary_y(img->h, z, num_zones);
    struct point_t from = {0u, y_line, 0u, 0u, 0u};
    struct point_t to   = {(uint16_t)(img->w - 1u), y_line, 0u, 0u, 0u};
    image_draw_line_color(img, &from, &to, color);
  }
}

#endif /* DD_GRID_UTILS_H */