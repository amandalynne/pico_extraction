from typing import Dict, List

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer

@DatasetReader.register("pico_sent_reader")
class PicoSentsDatasetReader(DatasetReader):
    """
    Reads instances from PICO text files structured as follows:
    
    ``
    ###ABSTRACT_ID
    HEADING|X| token1 token2 ... tokenN
    HEADING|X| token1 token2 ... tokenN

    ###ABSTRACT_ID
    HEADING|X| token1 token2 ... tokenN   
    ``
    
    where each line corresponds to a pretokenized sentence from a PubMed abstract.
    Each |X| is a label corresponding to a HEADING. 
    For example, the heading PARTICIPANTS maps to the label P.

    Since we are only interested in PICO elements, only labels P, I, and O 
    are considered; sentences with other labels are considered negative examples
    and relabeled with the label N.
    
    Each ``Instance`` has a ``"tokens"`` ``TextField`` containing the tokenized
    words and a ``"label"`` ``LabelField`` corresponding to the sentence label.
   
    Parameters
    ----------
    binarize : ``bool``, optional (default=False)
        If True, the dataset will have binary labels { PICO / NOT_PICO }
        rather than multiclass { P, I, O, N }.
    tokenizer: ``Tokenizer``, optional (default = ``WhitespaceTokenizer``)
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        Default to single word id token indexer.
    """
   

    def __init__(self,
                 binarize: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.binarize = binarize
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        # The relevant labels; others will be considered negative examples
        self.pio_labels = set(['P', 'I', 'O'])

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), 'r') as data_file:
            # Read in all the lines that are not Abstract IDs
            lines = [line.strip().split('|') for line in data_file.readlines()
                     if not line.startswith('#')]
            for line in lines:
                # Skip empty lines
                if line[0]:
                    label = line[1]
                    # If the label is not PIO, it is a negative example
                    if not label in self.pio_labels:
                        label = 'N'
                    # Rename 'O' to avoid label clash with IO tagged EBM-NLP
                    if label == 'O':
                        label = 'O-sent'
                    else:
                        if self.binarize:
                            label = 'PICO' 
                    text = line[2]
                    yield self.text_to_instance(text, label)


    @overrides
    def text_to_instance(self, text: str, label: str = None) -> Instance:
        tokens = self.tokenizer.tokenize(text)
        text_field = TextField(tokens, self._token_indexers)
        instance_fields: Dict[str, Field] = {'tokens': text_field}
       
        if label is not None:
            # Don't use default label namespace 
            instance_fields['label'] = LabelField(label, "pico_labels")
       
        return Instance(instance_fields)
