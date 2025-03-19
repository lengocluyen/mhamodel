#!/bin/bash

# Loop from 1 to 16 and run main.py with each argument
for i in {1..16}
do
    echo "Running python ./main.py $i ..."
    python ./main.py $i >> ./outputs/output_$i.txt 2>&1
    echo "Finished running python ./main.py $i, output saved to ./outputs/output_$i.txt"
done

echo "All models have been executed."
