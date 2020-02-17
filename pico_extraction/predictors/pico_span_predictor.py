from typing import Tuple

from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor

@Predictor.register('pico-predictor')
class PicoSpanPredictor(Predictor):
    """Predictor wrapper for PICO span labeler"""
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        # Data is already tokenized so just split on spaces
        self._tokenizer = JustSpacesWordSplitter() 

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like: ``{"sentence": "..."}``.
        Tokenizes the sentence and returns an Instance
        """
        sentence = json_dict['sentence']
        tokens = self._tokenizer.split_words(sentence)
        instance = self._dataset_reader.text_to_instance(tokens)
        return instance
