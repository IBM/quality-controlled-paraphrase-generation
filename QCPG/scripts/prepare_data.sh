

python="python"

raw_datasets_dir="data"
processed_datasets_dir="datasets"

for dataset in "parabk2" "wikians" "mscoco"
do
    for split in "train" "validation" "test"
    do
        output_file=$processed_datasets_dir/$dataset/$split.csv.gz
        command="$python QCPG/evaluate.py \
                --train_file $raw_datasets_dir/$dataset/$source.csv.gz \
                --dataset_split train \
                --predictions_column source \
                --references_column target \
                --metric metrics/para_metric \
                --output_path $output_file"
        
        if [ ! -f "$output_file" ]; then
            echo $command
        fi

    done
done
