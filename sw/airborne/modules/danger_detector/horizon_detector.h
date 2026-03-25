/*
 * Horizon boundary validity detector.
 */

#ifndef HORIZON_DETECTOR_H
#define HORIZON_DETECTOR_H

#include <stdbool.h>
#include <stdint.h>

#ifndef HORIZON_DETECTOR_NUM_ZONES
#define HORIZON_DETECTOR_NUM_ZONES 12
#endif

#ifndef HORIZON_DETECTOR_SENDER_ID
#define HORIZON_DETECTOR_SENDER_ID 2
#endif

#ifndef HORIZON_DETECTOR_DATA_TYPE
#define HORIZON_DETECTOR_DATA_TYPE 2
#endif

/* Public output: per-zone horizon-invalid flags (0 = valid/ok, 1 = invalid). */
extern uint8_t horizon_invalid_flags[HORIZON_DETECTOR_NUM_ZONES];

/* Tunables (GCS). */
extern uint8_t hh_row_step;                  /* sampled row stride, default 3 */
extern uint8_t hh_min_green_run;             /* consecutive green pixels, default 6 */
extern uint8_t hh_green_min;                 /* green channel lower bound */
extern uint8_t hh_green_max;                 /* green channel upper bound */
extern uint8_t hh_red_min;                   /* red channel lower bound */
extern int8_t  hh_green_minus_blue_min;      /* (G - B) lower bound */
extern int8_t  hh_green_minus_red_min;       /* (G - R) lower bound */
extern uint8_t hh_luma_min;                  /* Y lower bound for green floor */
extern uint8_t hh_luma_max;                  /* Y upper bound for green floor */
extern uint8_t hh_cb_max;                    /* Cb upper bound for green floor */
extern uint8_t hh_cr_max;                    /* Cr upper bound for green floor */
extern float   hh_floor_strip_frac;          /* min valid x = width * fraction */
extern float   hh_invalid_percent_threshold; /* per-zone invalid percentage threshold */
extern float   hh_max_slope;                 /* |dx/dy| threshold */
extern float   hh_max_slope_change;          /* |d(dx/dy)| threshold */
extern uint8_t hh_slope_window;              /* local slope window around each point */
extern bool    hh_draw;                      /* draw validity points and zone lines */

extern void horizon_detector_init(void);
extern void horizon_detector_periodic(void);

/* Copies latest per-zone invalid flags to out[]. Returns 0 on success. */
extern int horizon_detector_get_flags(uint8_t *out);

#endif /* HORIZON_DETECTOR_H */
