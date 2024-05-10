#!/bin/bash
#Usage: trap s6|m6
source /home/pi/venv/bin/activate
input="$1.txt"
filepath="/media/pi/T7/$1/"
csv=".csv"
npy=".npy"
trapdir="M2022S/$1/$2/"
while IFS= read -r line
do
	echo "$filepath$line"
	python detectClassifySpecies.py --weights insectMoths-bestF1-1280$2.pt --result $1$2 --device cpu --img 1280 --conf 0.20 --nosave --source $filepath$line
	if [ -d $trapdir ]; then
            echo "$trapdir save results"
        else
            mkdir $trapdir
            echo "$trapdir created for result files"
        fi
	mv $1$2.csv "$trapdir/$line$csv"
	mv $1$2.npy "$trapdir/$line$npy"
done < "$input"
