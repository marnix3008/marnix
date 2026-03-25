/*
 * Copyright (C) Roland Meertens
 *
 * This file is part of paparazzi
 *
 */
/**
 * @file "modules/orange_avoider/orange_avoider.h"
 * @author Roland Meertens
 * Example on how to use the colours detected to avoid orange pole in the cyberzoo
 */

#ifndef ORANGE_AVOIDER_H
#define ORANGE_AVOIDER_H

// settings
extern float oa_free_threshold;         // danger score below which a slice is free (0-100)
extern float oa_min_gap_width;          // minimum adjacent free slices to form a valid gap
extern float oa_max_heading_increment;  // max heading change per tick [deg]
extern float maxDistance;               // max waypoint displacement [m]

// gate approach settings
extern float gate_y_gain;       // heading correction gain (pixel-normalized, scales oa_max_heading_increment)
extern int   gate_close_px;     // gate_width threshold [px] to trigger GATE_TRAVERSE
extern int   gate_stale_ticks;  // ticks before aborting lost-gate approach

// functions
extern void orange_avoider_init(void);
extern void orange_avoider_periodic(void);

#endif

