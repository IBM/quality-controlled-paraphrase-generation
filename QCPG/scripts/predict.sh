
#!/bin/bash

seed=1
model_dir="..."
output_dir="..."
datasets_dir="..."
python=python
submit="bsub -queue x86_1h -cores 2+1 -require a100 -mem 100g"
model=t5
size=base
predict="\
    $python QCPG/predict.py \
    --per_device_eval_batch_size 256 \
    --per_device_train_batch_size 256 \
    --source_column reference \
    --target_column prediction \
    --dataset_map 'semantic_sim = 5 * round(bleurt_score * 100 / 5); lexical_div = 5 * round(set_diversity * 100 / 5); syntactic_div = 5 * round(syn_diversity * 100 / 5)' \
    --conditions_columns '[\"semantic_sim\", \"lexical_div\", \"syntactic_div\"]' \
"

for training_type in "bleurt"
do
    
    for task_name in "mscoco" "wikians" "parabk2"
    do
        for lr in "1e-3" "5e-3" "1e-4" "5e-4"
        do
            name=$model-$size-cond-$task_name-$training_type-lr$lr-v$seed

            output_file="$output_dir/validation/$name"

            if [ ! -f "$output_file/generated_predictions.csv" ]; then
                job="$submit $predict \
                                --train_file $datasets_dir/$task_name/validation.csv.gz \
                                --dataset_split train \
                                --model_name_or_path $model_dir/$name \
                                --output_dir $output_file \
                    "
                set +f
                GLOBIGNORE=*
                # echo $job
                eval $job
            fi
            
        done
    done
done