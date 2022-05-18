seed=1
datasets_dir="..."
dir="..."
output_dir="..."
python="python"
submit="bsub -queue x86_24h -cores 2+2 -require a100 -mem 100g"

model=google/electra-base-discriminator
model_name=electra
size=base
train="\
    $python QP/train.py \
    --model_name_or_path $model \
    --do_train \
    --do_eval \
    --do_predict \
    --per_device_eval_batch_size 16 \
    --per_device_train_batch_size 16 \
    --evaluation_strategy epoch \
    --save_total_limit 1 \
    --overwrite_output_dir \
    --num_train_epochs 6 \
    --dataset_generate_mode force_redownload \
    --dataset_keep_in_memory \
    --label_columns \"[\\\"bleurt_score\\\",\\\"set_diversity\\\",\\\"syn_diversity\\\"]\" \
    --input_columns \"[\\\"reference\\\"]\"
"

for training_type in "bleurt"
do
    score="_score"
    labels="--label_columns '[\"bleurt_score\",\"set_diversity\",\"syn_diversity\"]' "
    for task_name in "mscoco" "wikians" "parabk2"
    do
        for lr in "3e-5" "5e-5" "1e-4" "1.5e-4"
        do
            data_version=""

            tags="$model_name,$size,$task_name,lr:$lr,v:$seed,metric:$training_type,type:cond_reg"
                        
            name=$model_name-$size-reg-cond-$task_name-$training_type-lr$lr-v$seed

            job="$submit $train $task_config \
                            --train_file $datasets_dir/$task_name$data_version/train.csv.gz \
                            --validation_file $datasets_dir/$task_name$data_version/validation.csv.gz \
                            --learning_rate $lr \
                            --output_dir $output_dir/$name \
                            --run_name $name \
                            --run_tags '$tags'
                "
            set +f
            GLOBIGNORE=*
           
            echo $job
            # eval $job
             sleep 300 # we need to understand why some jobs has not been submitted                   

        done
    done
done
