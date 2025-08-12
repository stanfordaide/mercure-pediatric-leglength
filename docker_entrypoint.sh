#!/usr/bin/env bash
set -Eeo pipefail
echo "-- Starting testmodule..."
python run.py $MERCURE_IN_DIR $MERCURE_OUT_DIR
echo "-- Done."