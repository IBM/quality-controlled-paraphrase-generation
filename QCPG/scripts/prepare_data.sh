
python="python"
datasets_dir="..."
launch="bsub -queue x86_6h -mem 60g -cores 2+1 -require v100"

declare -A splits; splits["train1"]=":100000"; splits["train2"]="100000:200000"; splits["train3"]="200000:300000"; splits["train4"]="300000:400000"; splits["train5"]="400000:500000"; splits["train6"]="500000:600000"; splits["train7"]="600000:700000"; splits["train8"]="700000:800000"; splits["train9"]="800000:900000"

for dataset in "parabk2" "wikians" "mscoco"
do
    for split in ${!splits[@]}
    do
        output_file=$datasets_dir/$dataset/$split.csv.gz
        command="$python QCPG/evaluate.py \
                --train_file .../${dataset}.csv \
                --dataset_split train[${splits["$split"]}] \
                --predictions_column sentence1 \
                --references_column sentence0 \
                --metric metrics/para_metric \
                --output_path $output_file"
        
        if [ ! -f "$output_file" ]; then
           echo "done"
        fi
            
    done
done