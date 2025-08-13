#!/usr/bin/env bash
set -Eeo pipefail
echo "-- Starting pediatric leg length module..."
python run.py $MERCURE_IN_DIR $MERCURE_OUT_DIR
echo "-- Done."