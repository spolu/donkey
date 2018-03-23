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
  grep step $temp/out.log | cut -d' ' -f6 | gnuplot -p -e 'set terminal png; set output "'$temp'/fps.png"; plot "/dev/stdin";'
  grep step $temp/out.log | cut -d' ' -f9 | gnuplot -p -e 'set terminal png; set output "'$temp'/mean.png"; plot "/dev/stdin";'
  grep step $temp/out.log | cut -d' ' -f10 | gnuplot -p -e 'set terminal png; set output "'$temp'/median.png"; plot "/dev/stdin";'
  grep step $temp/out.log | cut -d' ' -f13 | gnuplot -p -e 'set terminal png; set output "'$temp'/min.png"; plot "/dev/stdin";'
  grep step $temp/out.log | cut -d' ' -f14 | gnuplot -p -e 'set terminal png; set output "'$temp'/max.png"; plot "/dev/stdin";'
  grep step $temp/out.log | cut -d' ' -f16 | gnuplot -p -e 'set terminal png; set output "'$temp'/loss_entropy.png"; plot "/dev/stdin";'
  grep step $temp/out.log | cut -d' ' -f18 | gnuplot -p -e 'set terminal png; set output "'$temp'/loss_value.png"; plot "/dev/stdin";'
  grep step $temp/out.log | cut -d' ' -f20 | gnuplot -p -e 'set terminal png; set output "'$temp'/loss_action.png"; plot "/dev/stdin";'
  grep STAT $TEMP/out.log | cut -d' ' -f3 | gnuplot -p -e 'set terminal png; set output "'$TEMP'/reward.png"; plot "/dev/stdin";'
done
