#ifndef ANN_kd_haus_H
#define ANN_kd_haus_H

#include "kd_tree.h"
#include "kd_util.h"
#include "pr_queue_k.h"

#include <ANNperf.h>

extern int           ANNkdDim;
extern ANNpoint      ANNkdQ;
extern double        ANNkdMaxErr;
extern ANNpointArray ANNkdPts;
extern ANNmin_k      *ANNkdPointMK;
extern int           ANNptsVisited;

#endif
