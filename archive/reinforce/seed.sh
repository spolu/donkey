#!/bin/bash

SEED=`date +'%Y%m%d_%H%M'`

mkdir -p $1/../seed_$SEED/

for p in "$1"/*
do
  cp $p $1/../seed_$SEED/$(basename $p)
done
