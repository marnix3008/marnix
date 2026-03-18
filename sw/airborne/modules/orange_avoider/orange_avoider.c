/*
 * Copyright (C) Roland Meertens
 *
 * This file is part of paparazzi
 *
 */
/**
 * @file "modules/orange_avoider/orange_avoider.c"
 * @author Roland Meertens
 * Modified for AE4317 course: 12-slice obstacle danger map navigation.
 *
 * Receives a 12-element danger array (0-100 per slice) via PAYLOAD_DATA ABI.
 * Each tick, finds the widest contiguous gap of free slices and steers toward
 * its center. Falls back to weighted-centroid turning when no valid gap exists.
 *
 * The OBSTACLE_DETECTION_ID define must match the sender ID used by the vision
 * module. Set it in your airframe XML:
 *   <define name="OBSTACLE_DETECTION_ID" value="1"/>
 */

#include "modules/orange_avoider/orange_avoider.h"
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
  SEARCH_FOR_SAFE_HEADING,
  OUT_OF_BOUNDS
};

// Settings (tunable via datalink)
float oa_free_threshold        = 30.f;  // danger score below which a slice is considered free (0-100)
float oa_min_gap_width         = 2.f;   // minimum number of adjacent free slices to form a valid gap
float oa_max_heading_increment = 5.f;   // maximum heading change per tick [deg]
float maxDistance              = 0.5f;  // max waypoint displacement [m]
float flight_altitude          = 0.8f;  // target flight altitude above ground [m]

// State
enum navigation_state_t navigation_state = SEARCH_FOR_SAFE_HEADING;
int16_t obstacle_free_confidence = 0;
float   committed_heading_increment = 0.f; // heading increment committed on entering SEARCH state
static int16_t search_ticks = 0;           // ticks spent turning without finding a gap
#define SEARCH_STUCK_TICKS    3            // after this many failed ticks, do a big jump turn
#define SEARCH_STUCK_JUMP_DEG 40.f         // degrees for the jump turn

const int16_t max_trajectory_confidence = 5;

// Slice danger data: NUM_SLICES values (0-100), index 0 = leftmost column
static uint8_t slice_danger[NUM_SLICES] = {0};

#ifndef OBSTACLE_DETECTION_ID
#define OBSTACLE_DETECTION_ID ABI_BROADCAST
#endif
static abi_event slice_detection_ev;
static void slice_detection_cb(uint8_t __attribute__((unused)) sender_id,
                                uint32_t __attribute__((unused)) stamp,
                                int32_t data_type,
                                uint32_t size,
                                uint8_t *data)
{
  if (data_type == 1 && size == NUM_SLICES) {
    for (int i = 0; i < NUM_SLICES; i++) {
      slice_danger[i] = data[i];
    }
  }
}

/*
 * Initialisation function
 */
void orange_avoider_init(void)
{
  srand(time(NULL));
  AbiBindMsgPAYLOAD_DATA(OBSTACLE_DETECTION_ID, &slice_detection_ev, slice_detection_cb);
  nav.fp_altitude = flight_altitude;  // apply default altitude on startup
}

/*
 * Called by the GCS slider to update flight altitude immediately (no periodic overhead).
 */
void orange_avoider_set_altitude(float alt)
{
  flight_altitude  = alt;
  nav.fp_altitude  = alt;
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
    if (slice_danger[i] < (uint8_t)oa_free_threshold) {
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
 * Fallback when no gap is found: turn away from the side with more danger.
 * Uses the weighted centroid of danger across all slices.
 * If all slices are equally dangerous, defaults to a small right turn.
 */
static float fallback_heading_increment(void)
{
  float weighted_sum = 0.f, total = 0.f;
  for (int i = 0; i < NUM_SLICES; i++) {
    weighted_sum += slice_danger[i] * (i - CENTER_SLICE); // positive = right side
    total        += slice_danger[i];
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

      if (!InsideObstacleZone(WaypointX(WP_TRAJECTORY), WaypointY(WP_TRAJECTORY))) {
        VERBOSE_PRINT("Trajectory out of arena bounds, turning back\n");
        navigation_state = OUT_OF_BOUNDS;

      } else if (!gap_found) {
        VERBOSE_PRINT("No safe gap found, stopping to find new heading\n");
        navigation_state = OBSTACLE_FOUND;

      } else {
        // Only correct heading when the forward path (center third of slices) is becoming dangerous.
        // This prevents jitter from peripheral noise when flying through open space.
        uint8_t forward_danger = 0;
        for (int i = NUM_SLICES / 3; i < 2 * NUM_SLICES / 3; i++) {
          if (slice_danger[i] > forward_danger) { forward_danger = slice_danger[i]; }
        }
        if (forward_danger >= (uint8_t)oa_free_threshold) {
          // Path ahead is dangerous: turn in place only, do NOT move forward
          float offset = gap_center - CENTER_SLICE;
          float heading_correction = (offset / CENTER_SLICE) * oa_max_heading_increment;
          VERBOSE_PRINT("Forward danger %d, turning toward gap at slice %.1f (correction %.1f deg) — holding position\n",
                        forward_danger, gap_center, heading_correction);
          increase_nav_heading(heading_correction);
        } else {
          // Path clear: fly straight
          VERBOSE_PRINT("Flying straight, forward path clear (danger %d)\n", forward_danger);
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

    case OUT_OF_BOUNDS:
      increase_nav_heading(oa_max_heading_increment);
      moveWaypointForward(WP_TRAJECTORY, 1.5f);

      if (InsideObstacleZone(WaypointX(WP_TRAJECTORY), WaypointY(WP_TRAJECTORY))) {
        VERBOSE_PRINT("Back inside arena, resuming search for safe heading\n");
        increase_nav_heading(oa_max_heading_increment);
        obstacle_free_confidence = 0;
        navigation_state = SEARCH_FOR_SAFE_HEADING;
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
