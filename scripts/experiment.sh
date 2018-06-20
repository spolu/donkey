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

TEMP="$(dirname `mktemp`)/reinforce_$EXPERIMENT"

ARGS="--save_dir $TEMP"

if [ "$2" != "" ]; then
  ARGS="--save_dir $TEMP --load_dir=$TEMP"
fi

echo "[Start] experiment=reinforce_$EXPERIMENT config=$CONFIG tempdir=$TEMP"

mkdir -p $TEMP
cp $CONFIG $TEMP/config.json
./scripts/update_experiment.sh $$ $EXPERIMENT $TEMP &

touch $TEMP/out.log
$PYTHON reinforce_trainer.py $CONFIG $ARGS >> $TEMP/out.log 2>&1
