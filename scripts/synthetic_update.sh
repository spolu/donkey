#!/bin/bash

TEMP=/tmp/$1

echo "[Update] experiment=$1 tempdir=$TEMP"
grep TRAIN $TEMP/out.log | tail -n 20000 | cut -d' ' -f 6 | gnuplot -p -e 'set terminal png; set output "'$TEMP'/bce_loss.png"; plot "/dev/stdin";'
grep TRAIN $TEMP/out.log | tail -n 20000 | cut -d' ' -f 8 | gnuplot -p -e 'set terminal png; set output "'$TEMP'/mse_loss.png"; plot "/dev/stdin";'
grep TRAIN $TEMP/out.log | tail -n 20000 | cut -d' ' -f 10 | gnuplot -p -e 'set terminal png; set output "'$TEMP'/kld_loss.png"; plot "/dev/stdin";'
