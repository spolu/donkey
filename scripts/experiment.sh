#!/bin/bash

PYTHON="${PYTHON:-python}"
CONFIG=$1

if [ "$1" == "" ]; then
  echo "[ERROR] Usage: experiment.sh {config_path} [{experiment_id}]"
  exit
else
  CONFIG=$1
fi

if [ "$2" == "" ]; then
  EXPERIMENT=`date +'%Y%m%d_%H%M'`
else
  EXPERIMENT=$2
fi

TEMP="$(dirname `mktemp`)/exp_$EXPERIMENT"

echo "[Start] experiment=$EXPERIMENT config=$CONFIG tempdir=$TEMP"

mkdir -p $TEMP

# Dump config.json for traceability
cp $CONFIG $TEMP/config.json

./scripts/update_experiment.sh $$ $EXPERIMENT $TEMP &

# Start the fuzzer on the config in the background
$PYTHON trainer.py $CONFIG > $TEMP/out.log 2>&1
