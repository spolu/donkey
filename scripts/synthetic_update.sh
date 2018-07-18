#!/bin/bash

TEMP=/tmp/$1

echo "[Update] experiment=$1 tempdir=$TEMP"
grep TRAIN $TEMP/out.log | cut -d' ' -f 6 | gnuplot -p -e 'set terminal png; set output "'$TEMP'/fake_loss.png"; plot "/dev/stdin";'
grep TRAIN $TEMP/out.log | cut -d' ' -f 8 | gnuplot -p -e 'set terminal png; set output "'$TEMP'/real_loss.png"; plot "/dev/stdin";'
grep TRAIN $TEMP/out.log | cut -d' ' -f 10 | gnuplot -p -e 'set terminal png; set output "'$TEMP'/vae_l1_loss.png"; plot "/dev/stdin";'
grep TRAIN $TEMP/out.log | cut -d' ' -f 12 | gnuplot -p -e 'set terminal png; set output "'$TEMP'/vae_mse_loss.png"; plot "/dev/stdin";'
grep TRAIN $TEMP/out.log | cut -d' ' -f 14 | gnuplot -p -e 'set terminal png; set output "'$TEMP'/vae_bce_loss.png"; plot "/dev/stdin";'
grep TRAIN $TEMP/out.log | cut -d' ' -f 16 | gnuplot -p -e 'set terminal png; set output "'$TEMP'/vae_kld_loss.png"; plot "/dev/stdin";'
grep TRAIN $TEMP/out.log | cut -d' ' -f 18 | gnuplot -p -e 'set terminal png; set output "'$TEMP'/vae_gan_loss.png"; plot "/dev/stdin";'
grep TRAIN $TEMP/out.log | cut -d' ' -f 20 | gnuplot -p -e 'set terminal png; set output "'$TEMP'/stl_mse_loss.png"; plot "/dev/stdin";'
grep TRAIN $TEMP/out.log | cut -d' ' -f 22 | gnuplot -p -e 'set terminal png; set output "'$TEMP'/stl_kld_loss.png"; plot "/dev/stdin";'
grep TRAIN $TEMP/out.log | cut -d' ' -f 24 | gnuplot -p -e 'set terminal png; set output "'$TEMP'/e2e_mse_loss.png"; plot "/dev/stdin";'
