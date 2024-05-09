#!/bin/bash
input="$1.txt"
filepath="/mnt/Dfs/Tech_TTH-KBE/MAVJF/data/2022/$1/"
csv=".csv"
npy=".npy"
trapdir="M2022S/$1"
while IFS= read -r line
do
	echo "$filepath$line"
	python detectClassifySpecies.py --weights insectMoths-bestF1-1280m6.pt --result $1 --device $2 --img 1280 --conf 0.20 --nosave --source $filepath$line
	if [ -d $trapdir ]; then
            echo "$trapdir save results"
        else
            mkdir $trapdir
            echo "$trapdir created for result files"
        fi
	mv $1.csv "$trapdir/$line$csv"
	mv $1.npy "$trapdir/$line$npy"
done < "$input"
