
#!/bin/bash

seed=1
predictions_dir="..."
output_dir="..."
python="python"
submit="bsub -queue x86_1h -cores 2+1 -require v100 -mem 100g"
model=t5
size=base
train="\
    $python QCPG/evaluate.py \
"
for training_type in "bleurt"
do
    
    for task_name in "mscoco" "wikians" "parabk2"
    do
        for lr in "1e-3" "5e-3" "1e-4" "5e-4"
        do
            
            name=$model-$size-cond-$task_name-$training_type-lr$lr-v$seed

            output_file="$output_dir/$name-metrics.csv"

            if [ ! -f "$output_file" ]; then
                job="$submit $train \
                                --train_file $predictions_dir/$name/generated_predictions.csv \
                                --dataset_split train \
                                --predictions_column prediction \
                                --references_column source \
                                --metric metrics/para_metric \
                                --output_path $output_file \
                    "
                # echo $job
                eval $job
            fi            
        done
    done
done