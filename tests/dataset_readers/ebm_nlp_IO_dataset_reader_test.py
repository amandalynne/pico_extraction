import os

from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list

from pico_extraction.dataset_readers.ebm_nlp import EBMNLPDatasetReader

class TestEBMNLPDatasetReader(AllenNlpTestCase):

    def setUp(self):
        super(TestEBMNLPDatasetReader, self).setUp()
        self.reader = EBMNLPDatasetReader('all')
        self.test_dir = 'tests/fixtures/ebm_nlp_IO_sample'
        self.instances = ensure_list(self.reader.read(self.test_dir))
   
    def test_smoke(self):
        assert self.reader is not None

    def test_read_from_directory(self):
        assert len(self.instances) == 2
  
    def test_read_from_file(self):
        # A single instance's tokens comprise an entire abstract.
        # Here we just test tokens and labels for first sentence of one abstract.
        line1 = "[ Effect of branch chain amino acid enriched formula on\
                 postoperative fatigue and nutritional status after\
                 digestive surgery ] .".split()
        instance1 = {"tokens": line1,
                     "labels": ['O', 'O', 'O', 'O', 'O', 'I-INT', 'I-INT',
                                'I-INT', 'I-INT', 'O', 'I-OUT', 'I-OUT', 'O',
                                'I-OUT', 'I-OUT', 'I-PAR', 'I-PAR', 'I-PAR',
                                'O', 'O']}
        fields = self.instances[0].fields
        assert [t.text for t in fields["tokens"].tokens[:20]] == instance1["tokens"]
        assert fields["labels"].labels[:20] == instance1["labels"]

    def test_instance_fields(self):
        instance_1 = self.instances[0]
        assert "tokens" in instance_1.fields
        assert "labels" in instance_1.fields
