
#!/bin/bash
seed=1
datasets_dir="..."
output_dir="..."
python="python"
submit="bsub -queue x86_24h -cores 2+2 -require a100 -mem 60g"
model=t5
size=base
train="\
    $python QCPG/train.py \
    --model_name_or_path $model-$size \
    --do_train \
    --do_eval \
    --source_column reference \
    --target_column prediction \
    --per_device_eval_batch_size 16 \
    --per_device_train_batch_size 16 \
    --predict_with_generate \
    --evaluation_strategy epoch \
    --num_train_epochs 6 \
    --lr_scheduler_type constant \
    --save_total_limit 1 \
    --dataset_generate_mode force_redownload \
    --dataset_keep_in_memory \
    --conditions_columns '[\"semantic_sim\", \"lexical_div\", \"syntactic_div\"]' \
    --overwrite_output_dir
"

for training_type in "bleurt"
do
    score="_score"
    map="--dataset_map 'semantic_sim = 5 * round(bleurt_score * 100 / 5); lexical_div = 5 * round(set_diversity * 100 / 5); syntactic_div = 5 * round(syn_diversity * 100 / 5)'"
    for task_name in "mscoco" "wikians" "parabk2"
    do
        for lr in "1e-3" "5e-3" "1e-4" "5e-4"
        do

            tags="$model,$size,$task_name,lr:$lr,v:$seed,metric:$training_type,type:conditional"
            
            task_config=${tasks["$task_name"]}
            data_version=""
            name=$model-$size-cond-$task_name-$training_type-lr$lr-v$seed
            job="$submit $train $task_config $map \
                            --train_file $datasets_dir/$task_name$data_version/train.csv.gz \
                            --validation_file $datasets_dir/$task_name$data_version/validation.csv.gz \
                            --learning_rate $lr \
                            --output_dir $output_dir/$name \
                            --dataset_generate_mode force_redownload \
                            --run_name $name \
                            --run_tags '$tags'
                "
            set +f
            GLOBIGNORE=*
            eval $job
        done
    done
done