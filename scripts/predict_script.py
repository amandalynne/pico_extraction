import argparse
import sys
import json

from allennlp.common.checks import check_for_gpu, ConfigurationError
from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor

from pico_extraction.models.baseline_crf_tagger import CrfTagger
from pico_extraction.dataset_readers.ebm_nlp import EBMNLPDatasetReader
from pico_extraction.predictors.pico_span_predictor import PicoSpanPredictor

if __name__ == "__main__":
    """
    This script is meant to help with error analysis. It prints out, for each 
    instance, the input tokens, true labels, and predicted labels for easier
    side-by-side comparison.

    It prints directly to the command line, so if you have a lot of instances
    that you want to predict, redirect the output with '>'.

    Usage: python predict_script.py <model_archive_file> <sentences_file>

    <model_archive_file> should be a model.tar.gz file.
    
    Each line of <sentences_file> should be JSON that looks as follows:
        {"sentence": "...", "labels": "..."}
    """
    model_file = sys.argv[1]
    sentences_file = sys.argv[2]

    archive = load_archive(model_file)
    predictor = Predictor.from_archive(archive, 'pico-predictor')    
    
    print('{:<20s}{:<8s}{:<8s}'.format("token", "true", "pred\n"))

    with open(sentences_file) as inf:
        for line in inf.readlines():
            instance = json.loads(line)
            result = predictor.predict_json(instance)
            tokens = instance['sentence'].split()
            labels = instance['labels'].split()
            preds = result.get('labels')
            for token, true, pred in zip(tokens, labels, preds):
                print('{:<20s} {:<8s} {:<8s}'.format(token, true, pred))
