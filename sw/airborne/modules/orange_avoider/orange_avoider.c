/*
 * Copyright (C) Roland Meertens
 *
 * This file is part of paparazzi
 *
 */
/**
 * @file "modules/orange_avoider/orange_avoider.c"
 * @author Roland Meertens
 * Modified for AE4317 course: 12-slice obstacle map navigation.
 *
 * Receives per-zone blocked/free arrays (0/1) from danger_detector and
 * horizon_detector via PAYLOAD_DATA ABI.
 * A slice is marked blocked only if both detectors report danger in that slice.
 * Each tick, finds the widest contiguous gap of free slices and steers toward
 * its center. Falls back to weighted-centroid turning when no valid gap exists.
 */

#include "modules/orange_avoider/orange_avoider.h"
#include "modules/danger_detector/danger_detector.h"
#include "modules/danger_detector/horizon_detector.h"
#include "firmwares/rotorcraft/navigation.h"
#include "generated/airframe.h"
#include "generated/flight_plan.h"
#include "state.h"
#include "modules/core/abi.h"
#include <time.h>
#include <stdio.h>

#ifndef ORANGE_AVOIDER_VERBOSE
#define ORANGE_AVOIDER_VERBOSE TRUE   // set FALSE in airframe XML to disable on the real drone
#endif

#define PRINT(string,...) fprintf(stderr, "[orange_avoider->%s()] " string,__FUNCTION__ , ##__VA_ARGS__)
#if ORANGE_AVOIDER_VERBOSE
#define VERBOSE_PRINT PRINT
#else
#define VERBOSE_PRINT(...)
#endif

#define NUM_SLICES   12 // Number of vertical slices, must match the vision module
#define CENTER_SLICE 5.5f   // (NUM_SLICES - 1) / 2.0, ADJUST MANUALLY IF NUM_SLICES CHANGES

static uint8_t moveWaypointForward(uint8_t waypoint, float distanceMeters);
static uint8_t calculateForwards(struct EnuCoor_i *new_coor, float distanceMeters);
static uint8_t moveWaypoint(uint8_t waypoint, struct EnuCoor_i *new_coor);
static uint8_t increase_nav_heading(float incrementDegrees);
static float   find_best_gap(void);
static float   fallback_heading_increment(void);

enum navigation_state_t {
  SAFE,
  OBSTACLE_FOUND,
  SEARCH_FOR_SAFE_HEADING
};

// Settings (tunable via datalink)
float oa_min_gap_width         = 6.f;   // minimum number of adjacent free slices to form a valid gap
float oa_max_heading_increment = 5.f;   // maximum heading change per tick [deg]
float maxDistance              = 0.5f;  // max waypoint displacement [m]

// State
enum navigation_state_t navigation_state = SEARCH_FOR_SAFE_HEADING;
int16_t obstacle_free_confidence = 0;
float   committed_heading_increment = 0.f; // heading increment committed on entering SEARCH state
static int16_t search_ticks = 0;           // ticks spent turning without finding a gap
#define SEARCH_STUCK_TICKS    3            // after this many failed ticks, do a big jump turn
#define SEARCH_STUCK_JUMP_DEG 40.f         // degrees for the jump turn

const int16_t max_trajectory_confidence = 5;

// Slice occupancy data: NUM_SLICES values (0/1), index 0 = leftmost column
static uint8_t slice_blocked[NUM_SLICES] = {0};
static uint8_t danger_blocked[NUM_SLICES] = {0};
static uint8_t horizon_blocked[NUM_SLICES] = {0};

static abi_event danger_detection_ev;
static abi_event horizon_detection_ev;

static void update_blocked_fused(void)
{
  for (int i = 0; i < NUM_SLICES; i++) {
    slice_blocked[i] = (danger_blocked[i] && horizon_blocked[i]) ? 1u : 0u;
  }
}

static void copy_flags_to_slices(uint8_t *dst, uint8_t *src, uint32_t src_size)
{
  if (!dst || !src || src_size == 0u) {
    return;
  }

  for (int i = 0; i < NUM_SLICES; i++) {
    uint32_t idx = ((uint32_t)i * src_size) / (uint32_t)NUM_SLICES;
    if (idx >= src_size) {
      idx = src_size - 1u;
    }
    dst[i] = (src[idx] != 0u) ? 1u : 0u;
  }
}

static void slice_detection_cb(uint8_t sender_id,
                               uint32_t __attribute__((unused)) stamp,
                               int32_t data_type,
                               uint32_t size,
                               uint8_t *data)
{
  if (!data || size == 0u) {
    return;
  }

  if ((sender_id == DANGER_DETECTOR_SENDER_ID) && (data_type == 1)) {
    copy_flags_to_slices(danger_blocked, data, size);
    update_blocked_fused();
  } else if ((sender_id == HORIZON_DETECTOR_SENDER_ID) &&
             (data_type == HORIZON_DETECTOR_DATA_TYPE)) {
    copy_flags_to_slices(horizon_blocked, data, size);
    update_blocked_fused();
  }
}

/*
 * Initialisation function
 */
void orange_avoider_init(void)
{
  srand(time(NULL));
  AbiBindMsgPAYLOAD_DATA(DANGER_DETECTOR_SENDER_ID, &danger_detection_ev, slice_detection_cb);
  AbiBindMsgPAYLOAD_DATA(HORIZON_DETECTOR_SENDER_ID, &horizon_detection_ev, slice_detection_cb);
}

/*
 * Finds the center of the widest contiguous run of free slices.
 * Returns fractional slice index (0 to NUM_SLICES-1),
 * or -1.0 if no valid gap of at least oa_min_gap_width exists.
 */
static float find_best_gap(void)
{
  int best_start = -1, best_width = 0;
  int run_start  = -1, run_width  = 0;

  for (int i = 0; i < NUM_SLICES; i++) {
    if (!slice_blocked[i]) {
      if (run_start == -1) { run_start = i; }
      run_width++;
    } else {
      if (run_width > best_width) {
        best_width = run_width;
        best_start = run_start;
      }
      run_start = -1;
      run_width = 0;
    }
  }
  // check the last run
  if (run_width > best_width) {
    best_width = run_width;
    best_start = run_start;
  }

  if (best_start == -1 || best_width < (int)oa_min_gap_width) {
    return -1.f;
  }

  return best_start + (best_width - 1) / 2.f;
}

/*
 * Fallback when no gap is found: turn away from the side with more blocked slices.
 * Uses the weighted centroid of blocked/free occupancy across all slices.
 * If occupancy is symmetric, defaults to a small right turn.
 */
static float fallback_heading_increment(void)
{
  float weighted_sum = 0.f, total = 0.f;
  for (int i = 0; i < NUM_SLICES; i++) {
    weighted_sum += (float)slice_blocked[i] * (i - CENTER_SLICE); // positive = right side
    total        += (float)slice_blocked[i];
  }
  if (total < 1.f) {
    return oa_max_heading_increment; // default: small right turn
  }
  float obstacle_offset = weighted_sum / total; // negative=left, positive=right

  // If wall is symmetric (centroid within 1 slice of center), the signal is too
  // weak to pick a direction — default to a full right turn to break the deadlock.
  if (obstacle_offset > -1.f && obstacle_offset < 1.f) {
    return oa_max_heading_increment;
  }

  // turn away: obstacle right → turn left (negative increment), and vice versa
  return -(obstacle_offset / CENTER_SLICE) * oa_max_heading_increment;
}

/*
 * Main periodic function (4 Hz).
 * Steers toward the widest free gap each tick in SAFE state.
 * Commits a turn direction and searches when fully blocked.
 */
void orange_avoider_periodic(void)
{
  if (!autopilot_in_flight()) { return; }

  float gap_center = find_best_gap();
  int   gap_found  = (gap_center >= 0.f);

  // Update confidence: gap available = one step safer, no gap = less certain
  if (gap_found) {
    obstacle_free_confidence++;
  } else {
    obstacle_free_confidence -= 2;
  }
  Bound(obstacle_free_confidence, 0, max_trajectory_confidence);

  float moveDistance = fminf(maxDistance, 0.2f * obstacle_free_confidence);

  VERBOSE_PRINT("gap_center: %.1f  confidence: %d  state: %d\n",
                gap_center, obstacle_free_confidence, navigation_state);

  switch (navigation_state) {

    case SAFE:
      moveWaypointForward(WP_TRAJECTORY, 1.5f * moveDistance);

      if (!gap_found) {
        VERBOSE_PRINT("No safe gap found, stopping to find new heading\n");
        navigation_state = OBSTACLE_FOUND;

      } else {
        // Only correct heading when the forward path (middle 6 slices) is becoming dangerous.
        // This prevents jitter from peripheral noise when flying through open space.
        bool forward_blocked = false;
        int forward_start = (NUM_SLICES - 6) / 2;
        int forward_end = forward_start + 6;
        for (int i = forward_start; i < forward_end; i++) {
          if (slice_blocked[i]) {
            forward_blocked = true;
            break;
          }
        }
        if (forward_blocked) {
          // Path ahead is dangerous: turn in place only, do NOT move forward
          float offset = gap_center - CENTER_SLICE;
          float heading_correction = (offset / CENTER_SLICE) * oa_max_heading_increment;
          VERBOSE_PRINT("Forward path blocked, turning toward gap at slice %.1f (correction %.1f deg) — holding position\n",
                        gap_center, heading_correction);
          increase_nav_heading(heading_correction);
        } else {
          // Path clear: fly straight
          VERBOSE_PRINT("Flying straight, forward path clear\n");
          moveWaypointForward(WP_GOAL, moveDistance);
        }
      }
      break;

    case OBSTACLE_FOUND:
      // Stop
      waypoint_move_here_2d(WP_GOAL);
      waypoint_move_here_2d(WP_TRAJECTORY);

      // Commit turn direction: toward gap if one exists, otherwise away from danger
      if (gap_found) {
        float offset = gap_center - CENTER_SLICE;
        committed_heading_increment = (offset / CENTER_SLICE) * oa_max_heading_increment;
      } else {
        committed_heading_increment = fallback_heading_increment();
      }

      VERBOSE_PRINT("Obstacle found. Committed turn: %.1f deg/tick\n",
                    committed_heading_increment);

      search_ticks = 0;
      navigation_state = SEARCH_FOR_SAFE_HEADING;
      break;

    case SEARCH_FOR_SAFE_HEADING:
      if (gap_found && obstacle_free_confidence >= 2) {
        VERBOSE_PRINT("Safe heading found (gap at slice %.1f), resuming flight\n", gap_center);
        search_ticks = 0;
        navigation_state = SAFE;
      } else {
        search_ticks++;
        if (search_ticks >= SEARCH_STUCK_TICKS) {
          float jump = (committed_heading_increment >= 0.f)
                       ?  SEARCH_STUCK_JUMP_DEG : -SEARCH_STUCK_JUMP_DEG;
          VERBOSE_PRINT("Stuck after %d ticks, jumping %.0f deg\n", search_ticks, jump);
          increase_nav_heading(jump);
          search_ticks = 0;
        } else {
          VERBOSE_PRINT("Searching: tick %d/%d, turning %.1f deg\n",
                        search_ticks, SEARCH_STUCK_TICKS, committed_heading_increment);
          increase_nav_heading(committed_heading_increment);
        }
      }
      break;

    default:
      break;
  }
}

/*
 * Increases the NAV heading. Assumes heading is an INT32_ANGLE. It is bound in this function.
 */
static uint8_t increase_nav_heading(float incrementDegrees)
{
  float new_heading = stateGetNedToBodyEulers_f()->psi + RadOfDeg(incrementDegrees);

  // normalize heading to [-pi, pi]
  FLOAT_ANGLE_NORMALIZE(new_heading);

  // set heading, declared in firmwares/rotorcraft/navigation.h
  nav.heading = new_heading;

  VERBOSE_PRINT("Increasing heading to %f\n", DegOfRad(new_heading));
  return false;
}

/*
 * Calculates coordinates of distance forward and sets waypoint 'waypoint' to those coordinates
 */
static uint8_t moveWaypointForward(uint8_t waypoint, float distanceMeters)
{
  struct EnuCoor_i new_coor;
  calculateForwards(&new_coor, distanceMeters);
  moveWaypoint(waypoint, &new_coor);
  return false;
}

/*
 * Calculates coordinates of a distance of 'distanceMeters' forward w.r.t. current position and heading
 */
static uint8_t calculateForwards(struct EnuCoor_i *new_coor, float distanceMeters)
{
  float heading  = stateGetNedToBodyEulers_f()->psi;

  // Now determine where to place the waypoint you want to go to
  new_coor->x = stateGetPositionEnu_i()->x + POS_BFP_OF_REAL(sinf(heading) * (distanceMeters));
  new_coor->y = stateGetPositionEnu_i()->y + POS_BFP_OF_REAL(cosf(heading) * (distanceMeters));
  VERBOSE_PRINT("Calculated %f m forward position. x: %f  y: %f based on pos(%f, %f) and heading(%f)\n", distanceMeters,
                POS_FLOAT_OF_BFP(new_coor->x), POS_FLOAT_OF_BFP(new_coor->y),
                stateGetPositionEnu_f()->x, stateGetPositionEnu_f()->y, DegOfRad(heading));
  return false;
}

/*
 * Sets waypoint 'waypoint' to the coordinates of 'new_coor'
 */
static uint8_t moveWaypoint(uint8_t waypoint, struct EnuCoor_i *new_coor)
{
  VERBOSE_PRINT("Moving waypoint %d to x:%f y:%f\n", waypoint, POS_FLOAT_OF_BFP(new_coor->x),
                POS_FLOAT_OF_BFP(new_coor->y));
  waypoint_move_xy_i(waypoint, new_coor->x, new_coor->y);
  return false;
}
