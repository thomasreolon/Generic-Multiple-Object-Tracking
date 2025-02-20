#!/usr/bin/env bash
# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------


set -x

set -o pipefail

OUTPUT_DIR=$1

# clean up *.pyc files
rmpyc() {
  rm -rf $(find -name __pycache__)
  rm -rf $(find -name "*.pyc")
}


cp submit_dance.py $OUTPUT_DIR

pushd $OUTPUT_DIR

args=$(cat tmp.args)
# rlaunch --cpu 8 --gpu 1 --memory 24000 --positive-tags 2080ti -P 13 -- python3 submit_dance.py ${args} --resume checkpoint.pth --exp_name tracker
python submit_dance.py ${args} --resume /home/intern/Desktop/Generic-Multiple-Object-Tracking/checkpoint.pth --exp_name exp/fscd_v1

popd

# python3 ../TrackEval/scripts/run_mot_challenge.py \
#     --SPLIT_TO_EVAL val  \
#     --METRICS HOTA CLEAR Identity  \
#     --GT_FOLDER /data/datasets/DanceTrack/val \
#     --SEQMAP_FILE seqmap \
#     --SKIP_SPLIT_FOL True \
#     --TRACKER_SUB_FOLDER tracker \
#     --TRACKERS_TO_EVAL $OUTPUT_DIR \
#     --USE_PARALLEL True \
#     --NUM_PARALLEL_CORES 8 \
#     --PLOT_CURVES False \
#     --TRACKERS_FOLDER '' | tee -a $OUTPUT_DIR/eval.log
