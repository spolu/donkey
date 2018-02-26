#!/bin/bash

MAKEPID=$1
EXPERIMENT=$2
TEMP=$3

START=`date +%s`
DUMP_PERIOD=60

while [ true ]; do
  sleep $DUMP_PERIOD

  # If make process is dead, exit.
  kill -0 $MAKEPID 2>/dev/null || exit

  RUNTIME=$((`date +%s`-$START))
  echo "[Update] experiment=$EXPERIMENT tempdir=$TEMP runtime=$RUNTIME"
  grep STAT $TEMP/out.log | cut -d' ' -f3 | gnuplot -p -e 'set terminal png; set output "'$TEMP'/reward.png"; plot "/dev/stdin";'
done
