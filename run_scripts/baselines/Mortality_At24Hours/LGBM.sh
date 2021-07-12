source activate icu-benchmark
python -m icu_benchmarks.run train \
                             -c configs/hirid/Classification/LGBM.gin \
                             -l logs/benchmark_exp/LGBM/ \
                             -t Mortality_At24Hours \
                             --maxlen 288 \
                             -o True \
                             --depth 7 \
                             --subsample-feat 1.00 \
                             --subsample-data 0.33 \
                             -sd 1111 2222 3333 4444 5555 6666 7777 8888 9999 0000
