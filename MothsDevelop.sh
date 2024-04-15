#!/bin/bash
input="MothsTest.txt"
filepath="/mnt/Dfs/Tech_TTH-KBE/MAVJF/Annotations/2022/"
csv=".csv"
while IFS= read -r line
do
	echo "$filepath$line"
	python detectClassifyInsects.py --weights insectMoths-bestF1-1280m6.pt --img 1280 --conf 0.25 --save-txt --nosave --source $filepath$line
	mv resultMoths.csv "Moths/$line$csv"
done < "$input"
