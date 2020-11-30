#!/bin/bash
SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
SCRIPTPATH=${SCRIPTPATH%/mlflow}
export PYTHONPATH=$SCRIPTPATH:$PYTHONPATH

