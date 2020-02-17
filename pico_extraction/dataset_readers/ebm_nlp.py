from typing import Dict, List, Sequence
import os

from overrides import overrides

from glob import glob

from allennlp.common.checks import ConfigurationError
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import MetadataField, TextField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

@DatasetReader.register("ebm_nlp")
class EBMNLPDatasetReader(DatasetReader):
    """
    Reads instances from a directory of tokenized PubMed abstract files, where 
    each line is in one of the following formats:

    1) WORD P-TAG I-TAG O-TAG
    
    where P, I, and O stand for 'participants', 'interventions', and 'outcomes',
    respectively. These are in BIO format.

    2) WORD PIO-TAG

    Disallows span overlaps and is in IO format.

    Each ``Instance`` has a ``"tokens"`` ``TextField`` containing the tokenized 
    words.
    
    As of Oct 08: only train for one PIO element at a time.
    As such, the ``"tags"`` ``SequenceLabelField`` contains the specified 
    ``element``.

    Parameters
    ----------
    element : ``str``
        Specify which PIO element (`participants`, `interventions`, `outcomes`, `all`)
        to load for which span labels to learn.
    single_sent : ``bool``, optional (default=False)
        Specify whether to base Instances around single sentences; if False,
        then an entire abstract constitutes an Instance.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        Default to single word id token indexer.
    """
    
    _PIO_ELEMENTS = ['participants', 'interventions', 'outcomes']

    def __init__(self,
                 element: str,
                 single_sent: bool = False,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__()
        if element not in self._PIO_ELEMENTS and element != "all":
            raise ConfigurationError("unknown PIO label type: {0}".format(element))
        self.element = element
        self.single_sent = single_sent
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        """
        For EBM-NLP, we expect file_path to refer to a directory of tokenized 
        abstracts.
        """
        # Iterate over documents in the directory.
        for doc in sorted(os.listdir(file_path)): 
            with open(os.path.join(file_path, doc), 'r') as data_file:
                # Collect the tokens and tags
                lines = [line.strip().split() for line in data_file.readlines()]
                if not self.single_sent:
                    # If we want a full abstract to be an instance
                    fields = [list(line) for line in zip(*lines)]
                    if self.element == 'all':
                        tokens_, labels_ = fields
                        labels = { 'all': labels_ } 
                    else:
                        tokens_, p_tags, i_tags, o_tags = fields
                        labels = { 'participants' : p_tags,
                                   'interventions': i_tags,
                                   'outcomes': o_tags }
                    tokens = [Token(token) for token in tokens_]
                    yield self.text_to_instance(tokens, labels)
                else:
                    # Yield individual sentences
                    end_of_sent = False
                    tokens_ = []
                    labels_ = []
                    for token, label in lines:
                        if not end_of_sent:
                            tokens_.append(token)
                            labels_.append(label)
                            if token == '.':
                                end_of_sent = True
                                tokens = [Token(token) for token in tokens_]
                                # Only doing the 'all' option here. 
                                # No time for that other sh*t.
                                labels = { 'all': labels_ } 
                                yield self.text_to_instance(tokens, labels)
                                tokens_ = []
                                labels_ = []
                        else:
                            end_of_sent = False
                            tokens_.append(token)
                            labels_.append(label)

    @overrides
    def text_to_instance(self,
                         tokens: List[Token],
                         labels: Dict[str, List[int]] = None):
        sequence = TextField(tokens, self._token_indexers)
        instance_fields: Dict[str, Field] = {'tokens': sequence}
        
        # Set the field 'labels' according to the specified PIO element
        if labels is not None:
            instance_fields['labels'] = SequenceLabelField(labels[self.element], sequence, 'ebm_labels')
        
        # Also make fields for all the PIO element tags.
        # This should come in handy later.
        if self.element != 'all':
            for element, label_seq in labels.items():
                tag_name = "{0}_tags".format(element[0])
                instance_fields[tag_name] = SequenceLabelField(label_seq, sequence, tag_name)
        return Instance(instance_fields)
