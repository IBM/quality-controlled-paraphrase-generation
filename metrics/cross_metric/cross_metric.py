import datasets
from transformers import AutoTokenizer, AutoModel
import json
from tqdm.auto import tqdm
from torch import cuda
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
class CrossMetric(datasets.Metric):
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
            model_name = 'sentence-transformers/paraphrase-mpnet-base-v2'
        else:
            model_name = self.config_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.set_device('cuda' if torch.cuda.is_available() else 'cpu')

    def set_device(self, device):
        self.model.to(torch.device(device))
        self.device = device
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def embed(self, sentences):

        inputs = self.tokenizer(list(sentences), padding=True, truncation=True, return_tensors='pt')
        inputs = {
            name: tensor.to(torch.device(self.device)) if isinstance(tensor, torch.Tensor) else tensor
            for name, tensor in inputs.items()
        }
        with torch.no_grad():
            model_output = self.model(**inputs)
        sentence_embeddings = self.mean_pooling(model_output, inputs['attention_mask'])
        return sentence_embeddings.cpu().detach()

    def _compute(self, predictions, references, batch_size=64, device=None):
        """Returns the scores"""
        if device != self.device and device is not None:
            self.set_device(device)

        preds_embeds = []
        refs_embeds = []
        for preds, refs in tqdm(zip(to_batches(predictions, batch_size), to_batches(references, batch_size)), total=int(ceil(min(len(predictions),len(references)) / batch_size)), desc="cross_metric"):
            
            preds_embeds.append(self.embed(preds))
            refs_embeds.append(self.embed(refs)) 
        
        preds = torch.cat(preds_embeds)
        refs = torch.cat(refs_embeds)

        scores = torch.nn.functional.cosine_similarity(preds, refs, dim=1)

        scores = (scores + 1) / 2

        return {
            "scores": scores.tolist(),
        }