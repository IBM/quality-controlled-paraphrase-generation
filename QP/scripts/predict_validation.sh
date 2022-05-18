seed=1
datasets_dir="..."
dir="..."
model_dir="..."
output_dir="..."
python=".python"
submit="bsub -queue x86_1h -cores 2+1 -require a100 -mem 30g"
model="google/electra-base-discriminator"
model_name="electra"
size="base"

training_type=cond_cls
declare -A tasks; tasks["mscoco"]="5e-5"; tasks["wikians"]="3e-5"; tasks["parabk2"]="3e-5";

for dataset in ${!tasks[@]}
    do  
        lr=${tasks["$dataset"]}
        tags="$model_name,$size,$dataset,lr:$lr,v:$seed,type:$training_type,eval"                   
        name=$model_name-$size-reg-cond-$dataset-bleurt-lr$lr-v$seed
        $python QP/train.py \
                --do_predict \
                --per_device_eval_batch_size 256 \
                --input_columns '["reference"]' \
                --label_columns '["bleurt_score", "set_diversity", "syn_diversity"]' \
                --model_name_or_path $model_dir/$name \
                --validation_file $datasets_dir/$dataset/validation.csv.gz \
                --dataset_split '{"validation":"validation[10000:]"}' \
                --output_dir $output_dir/$dataset \
                --run_name $name-test \
                --run_tags '$tags' 
    done




