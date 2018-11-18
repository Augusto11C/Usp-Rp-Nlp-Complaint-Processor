#!/bin/sh
for i in 6 7 8 9 10; do
    filename="REG_LDA_1000_FEATURES_RUN_STDOUT_"$i".txt";
    python SMELLY_LDA.py $i 0 | iconv -f ISO8859-1 -t utf-8 | tee "$filename"
done;