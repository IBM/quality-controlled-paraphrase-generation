import datasets
import spacy
import benepar
from apted import APTED
from apted.helpers import Tree
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

def to_batches(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def normalize_tree(tree_string, max_depth=3):
    res = []
    depth = -1
    leaf = False
    for c in tree_string:
        if c in ['{', '}']:
            continue
        if c == '(':
            leaf=False
            depth += 1

        elif c == ')':
            leaf=False
            depth -= 1
            if depth < max_depth:
                res.append('}')
                continue
                
        elif c == ' ':
            leaf=True
            continue

        if depth <= max_depth and not leaf and c != ')':
            res.append(c if c != '(' else '{')
        
    return ''.join(res)

def tree_edit_distance(lintree1, lintree2):
    
    tree1 = Tree.from_text(lintree1)
    tree2 = Tree.from_text(lintree2)
    n_nodes_t1 = lintree1.count('{')
    n_nodes_t2 = lintree2.count('{')

    apted = APTED(tree1, tree2)
    ted = apted.compute_edit_distance()
    return ted / (n_nodes_t1 + n_nodes_t2)

def get_tree_string(doc):
    return next(iter(doc.sents))._.parse_string

def dist(pair):

    p_tree_n = normalize_tree(pair[0], max_depth=3)
    r_tree_n = normalize_tree(pair[1], max_depth=3)
    
    ted = tree_edit_distance(p_tree_n, r_tree_n)

    return ted


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
        benepar.download('benepar_en3')
        spacy.prefer_gpu()
        self.nlp = spacy.load('en_core_web_sm')
        if spacy.__version__.startswith('2'):
            self.nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))
        else:
            self.nlp.add_pipe("benepar", config={"model": "benepar_en3"})

    

      # or thread_map

    def _compute(self, predictions, references, batch_size=64, workers=16):
        """Returns the scores"""
        with self.nlp.select_pipes(enable=["parser", "benepar"]):
            preds = list(tqdm(self.nlp.pipe(predictions, batch_size=batch_size), total=len(predictions), desc="syntdiv:parse_preds", disable=True))
            preds = list(map(get_tree_string, preds))
            refs = list(tqdm(self.nlp.pipe(references, batch_size=batch_size), total=len(references), desc="syntdiv:parse_refs", disable=True))
            refs =  list(map(get_tree_string, refs))
        
        scores = list(tqdm(map(dist, zip(preds, refs)), total=len(preds), desc="syntdiv:calc_dist"))

        return {
            "scores": scores,
        }