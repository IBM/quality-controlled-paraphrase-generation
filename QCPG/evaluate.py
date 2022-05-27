import sys
import os

from dataclasses import dataclass, field
from typing import Optional
from datasets import GenerateMode
from data import DatasetArguments, prepare_dataset
from transformers import HfArgumentParser
from datasets import load_metric
import pandas as pd

@dataclass
class EvalArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    predictions_column: Optional[str] = field(
        metadata={"help": "the source column"}
    )
    references_column: Optional[str] = field(
        default=None, metadata={"help": "the target column"}
    )
    metric_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "metric you wish to apply"}
    )
    output_path: Optional[str] = field(
        default='pairs_evals.csv', metadata={"help": "metric you wish to apply"}
    )

def main():    

    from datasets import set_caching_enabled
    set_caching_enabled(False)

    parser = HfArgumentParser((DatasetArguments, EvalArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        dataset_args, eval_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        dataset_args, eval_args = parser.parse_args_into_dataclasses()

    os.makedirs(os.path.abspath(os.path.dirname(eval_args.output_path)), exist_ok=True)

    dataset = prepare_dataset(dataset_args)
    column_names = dataset.column_names

    predictions = dataset[column_names[1]] if eval_args.predictions_column is None \
                  else dataset[eval_args.predictions_column]
    references = dataset[column_names[0]] if eval_args.references_column is None \
                  else dataset[eval_args.references_column]
    
    metric = load_metric(eval_args.metric_name_or_path, experiment_id=os.getpid())

    print('Computing metric...')
    result = metric.compute(predictions=predictions, references=references)
    
    result['prediction'] = predictions
    result['reference'] = references

    try:
        df = pd.DataFrame(result)
    except:
        print(result)
        raise NotImplementedError
        
    df.to_csv(eval_args.output_path, index=False)
    


if __name__ == "__main__":
    main()
