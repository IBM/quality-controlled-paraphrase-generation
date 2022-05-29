import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm.auto import tqdm
import torch
from math import ceil

def to_batches(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

# TODO: Add BibTeX citation
_CITATION = """\
@InProceedings{huggingface:metric,
title = {A great new metric},
authors={huggingface, Inc.},
year={2020}
}
"""

# TODO: Add description of the metric here
_DESCRIPTION = """\
This new metric is designed to solve this great NLP task and is crafted with a lot of care.
"""


# TODO: Add description of the arguments of the metric here
_KWARGS_DESCRIPTION = """
Calculates how good are predictions given some references, using certain scores
Args:
    predictions: list of predictions to score. Each predictions
        should be a string with tokens separated by spaces.
    references: list of reference for each prediction. Each
        reference should be a string with tokens separated by spaces.
Returns:
    accuracy: description of the first score,
    another_score: description of the second score,
Examples:
    Examples should be written in doctest format, and should illustrate how
    to use the function.
    >>> my_new_metric = datasets.load_metric("my_new_metric")
    >>> results = my_new_metric.compute(references=[0, 1], predictions=[0, 1])
    >>> print(results)
    {'accuracy': 1.0}
"""

# TODO: Define external resources urls if needed
BAD_WORDS_URL = "http://url/to/external/resource/bad_words.txt"


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class BLEURT(datasets.Metric):
    """TODO: Short description of my metric."""

    def _info(self):
        # TODO: Specifies the datasets.MetricInfo object
        return datasets.MetricInfo(
            # This is the description that will appear on the metrics page.
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            # This defines the format of each prediction and reference
            features=datasets.Features({
                'predictions': datasets.Value('string'),
                'references': datasets.Value('string'),
            }),
            # Homepage of the metric for documentation
            homepage="http://metric.homepage",
            # Additional links to the codebase or references
            codebase_urls=["http://github.com/path/to/codebase/of/new_metric"],
            reference_urls=["http://path.to.reference.url/new_metric"]
        )

    def _download_and_prepare(self, dl_manager):
        """Optional: download external resources useful to compute the scores"""
        if self.config_name == 'default':
            model_name = 'Elron/bleurt-large-512'
        else:
            model_name = self.config_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.set_device('cuda' if torch.cuda.is_available() else 'cpu')

    def set_device(self, device):
        self.model.to(torch.device(device))
        self.device = device
    
    scores = []
    def _compute(self, predictions, references, batch_size=64, device=None, adjusted=True):
        """Returns the scores"""
        if device != self.device and device is not None:
            self.set_device(device)

        scores = []
        for preds, refs in tqdm(zip(to_batches(predictions, batch_size), to_batches(references, batch_size)), total=int(ceil(min(len(predictions),len(references)) / batch_size)), desc="bleurt", disable=True):

            inputs = self.tokenizer(preds, refs, return_tensors='pt', padding='longest')
            inputs = {
                name: tensor.to(torch.device(self.device)) if isinstance(tensor, torch.Tensor) else tensor
                for name, tensor in inputs.items()
            }   
            
            with torch.no_grad():
                outputs = self.model(**inputs)['logits'].squeeze(-1)

            if adjusted:
                outputs = 1 / (1 + 2 ** (-4 * outputs)) # scaled sigmoid
            
            scores += outputs.tolist()
        
        return {
            "scores": scores,
        }
