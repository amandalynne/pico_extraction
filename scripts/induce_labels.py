import sys
import json

import numpy as np
import torch

from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor

from pico_extraction.models.sentence_attn import PicoSentenceLabeler
from pico_extraction.dataset_readers.pico_sentences import PicoSentsDatasetReader 
from pico_extraction.predictors.pico_span_predictor import PicoSpanPredictor

LABELS_TO_CONLL = { 'P': 'I-PAR', 'I': 'I-INT', 'O': 'I-OUT', 'N': 'O' }

CONLL_TO_LABELS = { 'I-PAR': 'P', 'I-INT': 'I', 'I-OUT': 'O', 'O': 'N' }

if __name__ == "__main__":
    """
    This script uses a pretrained self-attention model to apply soft labels
    to tokens based on the attention weights. 
    
    Usage: python viterbi_labeler.py <model_archive_file> <sentences_file> <probabilities> <output_file>

    <model_archive_file> should be a *.tar.gz file.
    
    <sentences_file> is a .txt file of PICO sentences.

    <probabilities> is a JSON file with tag init & transition probabilities pre-calculated from EBM-NLP.
    """
    model_file = sys.argv[1]
    sentences_file = sys.argv[2]
    output_file = sys.argv[3]

    # Load predictor from model
    archive = load_archive(model_file)
    predictor = Predictor.from_archive(archive, 'pico-predictor')    
    
    # Vocab from model -- convenient for mapping tags to indices
    vocab = predictor._model.vocab

    with open(output_file, 'w+') as outf:
        for instance in predictor._dataset_reader._read(sentences_file):
            tokens = instance.fields['tokens'].tokens
            result = predictor.predict_instance(instance)
            attn_weights = result.get('attn_weights')

            # Argmax on attn weights per token
            label_indices = np.argmax(np.array(attn_weights), axis=0) 

            soft_labels = [vocab.get_token_from_index(i, namespace='labels') for i in label_indices]

            for token, soft_label in zip(tokens, soft_labels):
                # Translate to CoNLL format
                if soft_label in ['P', 'I', 'O']:
                    soft_label = LABELS_TO_CONLL[soft_label]
                if soft_label == 'N':
                    soft_label = 'O'
                outf.write('{0} {1}\n'.format(token.text, soft_label))
