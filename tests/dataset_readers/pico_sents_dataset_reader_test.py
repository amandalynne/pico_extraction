import pytest

from allennlp.common.util import ensure_list
from allennlp.data.tokenizers import PretrainedTransformerTokenizer

from pico_extraction.dataset_readers.pico_sentences import PicoSentsDatasetReader
from tests import FIXTURES_ROOT

class TestPicoSentsDatasetReader:


    def test_read_from_file(self):
        reader = PicoSentsDatasetReader() 
        instances = reader.read(FIXTURES_ROOT / "PICO_test.txt") 
        instances = ensure_list(instances)

        assert len(instances) == 11 
  
        # Test a positive instance
        line = "The patients attended @ supervised visits over a @ period .".split()
        instance = {"tokens": line,
                     "label": "I"} 
        fields = instances[6].fields
        assert [t.text for t in fields["tokens"].tokens] == instance["tokens"]
        assert fields["label"].label == instance["label"]
    
        # Test a negative instance
        line = "Cross-sectional study .".split()
        instance = {"tokens": line,
                     "label": "N"} 
        fields = instances[3].fields
        assert [t.text for t in fields["tokens"].tokens] == instance["tokens"]
        assert fields["label"].label == instance["label"]
