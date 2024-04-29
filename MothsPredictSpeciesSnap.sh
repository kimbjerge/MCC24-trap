#!/bin/bash
filepath="/mnt/Dfs/Tech_TTH-KBE/MAVJF/data/2022/$1"
csv=".csv"
npy=".npy"
echo "$filepath$line"
python detectClassifySpecies.py --weights insectMoths-bestF1-1280m6.pt --result $1 --img 1280 --conf 0.20 --nosave --source $filepath
mv $1.csv "M2022S/$1$csv"
mv $1.npy "M2022S/$1$npy"
