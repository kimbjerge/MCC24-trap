#!/bin/bash
input="$1.txt"
filepath="/mnt/Dfs/Tech_TTH-KBE/MAVJF/data/2022/$1/"
csv=".csv"
npy=".npy"
while IFS= read -r line
do
	echo "$filepath$line"
	python detectClassifyInsects.py --weights insectMoths-bestF1-1280m6.pt --result $1 --img 1280 --conf 0.20 --nosave --source $filepath$line
	mv $1.csv "M2022/$1/$line$csv"
	mv $1.npy "M2022/$1/$line$npy"
done < "$input"
