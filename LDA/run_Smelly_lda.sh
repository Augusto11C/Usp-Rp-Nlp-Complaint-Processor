#!/bin/sh
for i in 7 8 9 10; do
    filename="SMELLY_LDA_500FEATURES_RUN_RESULTS_"$i".txt";    
    python SMELLY_LDA.py $i | iconv -f ISO8859-1 -t utf-8 > "$filename";
done