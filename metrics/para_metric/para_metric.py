import datasets
from datasets import load_metric, GenerateMode
from Levenshtein import setratio, seqratio
from tqdm.auto import tqdm
import os

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
class ModelBasedMetric(datasets.Metric):
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
        self.bertscore = load_metric("bertscore", experiment_id=self.experiment_id)
        self.bleu = load_metric("sacrebleu", experiment_id=self.experiment_id)
        self.sbert = load_metric("metrics/cross_metric", experiment_id=self.experiment_id)
        self.syntdiv = load_metric("metrics/syntdiv_metric", experiment_id=self.experiment_id)
        self.bleurt = load_metric("metrics/bleurt", experiment_id=self.experiment_id)
    
    def _compute(self, predictions, references, batch_size=100):
        """Returns the scores"""
        set_diversities = []
        seq_diversities = []
        diversity_scores = []
        total=len(predictions)
        for pred, ref in tqdm(zip(predictions, references), total=total, desc="edit_distance", disable=True):
            set_diversity = 1 - setratio(pred, ref)
            seq_diversity = 1 - seqratio(pred, ref)
            diversity_score = (set_diversity + seq_diversity) / 2 
            set_diversities.append(set_diversity)
            seq_diversities.append(seq_diversity)
            diversity_scores.append(diversity_score)
        bleu_scores = [self.bleu.compute(predictions=[pred], references=[[ref]])['score'] for pred, ref in tqdm(zip(predictions, references), total=total, desc="bleu")]
        bleurt_score = self.bleurt.compute(predictions=predictions, references=references)['scores']
        # bertscore_score = self.bertscore.compute(predictions=predictions, references=references, lang='en')["f1"]
        # semantic_scores = self.sbert.compute(predictions=predictions, references=references)["scores"]
        syntactic_diversity = self.syntdiv.compute(predictions=predictions, references=references)["scores"]
        

        diversity_scores = [(lex + syn) / 2 for lex, syn in zip(syntactic_diversity, set_diversities)]

        return {
            "set_diversity": set_diversities,
            "seq_diversity": seq_diversities,
            "syn_diversity": syntactic_diversity,
            "diversity_score": diversity_scores,
            "bleu_score": bleu_scores,
            "bleurt_score": bleurt_score,
            # "bertscore_score": bertscore_score,
            # "semantic_score": semantic_scores
        }