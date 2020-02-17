import os

from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list

from pico_extraction.dataset_readers.pico_sentences import PicoSentsDatasetReader

class TestPicoSentsDatasetReader(AllenNlpTestCase):

    def setUp(self):
        super(TestPicoSentsDatasetReader, self).setUp()
        self.reader = PicoSentsDatasetReader()
        self.test_file = 'tests/fixtures/PICO_test.txt'
        self.instances = ensure_list(self.reader.read(self.test_file))

    def test_smoke(self):
        assert self.reader is not None

    def test_read_from_file(self):
        assert len(self.instances) == 11 
  
        # Test a positive instance
        line = "The patients attended @ supervised visits over a @ period .".split()
        instance = {"tokens": line,
                     "label": "I"} 
        fields = self.instances[6].fields
        assert [t.text for t in fields["tokens"].tokens] == instance["tokens"]
        assert fields["label"].label == instance["label"]
    
        # Test a negative instance
        line = "Cross-sectional study .".split()
        instance = {"tokens": line,
                     "label": "N"} 
        fields = self.instances[3].fields
        assert [t.text for t in fields["tokens"].tokens] == instance["tokens"]
        assert fields["label"].label == instance["label"]
