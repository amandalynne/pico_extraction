from typing import Dict, Optional, List, Any
import warnings

from overrides import overrides
import torch
from torch.nn.modules.linear import Linear
import torch.nn.functional as F

from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
import allennlp.nn.util as util
from allennlp.training.metrics import CategoricalAccuracy, F1Measure

@Model.register("pico_sentence_labeler")
class PicoSentenceLabeler(Model):
    """
    The ``PicoSentenceLabeler`` encodes a sequence of text with a ``Seq2SeqEncoder``,
    and then this encoding is passed through a Seq2Vec encoding.
    then predicts a PICO label for the sentence. 

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the tokens ``TextField`` we get as input to the model.
    encoder : ``Seq2SeqEncoder``
        The encoder that we will use in between embedding tokens and predicting output labels.
    feedforward : ``FeedForward``
    dropout:  ``float``, optional (detault=``None``)
    verbose_metrics : ``bool``, optional (default = False)
        If true, metrics will be returned per label class in addition
        to the overall statistics.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 feedforward: FeedForward,
                 dropout: Optional[float] = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.encoder = encoder 
        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None
        self._feedforward = feedforward

        self.metrics = {
            "accuracy": CategoricalAccuracy()
        }
        # Add F1 score for individual labels to metrics                         
        for index, label in self.vocab.get_index_to_token_vocabulary("labels").items():
            self.metrics[label] = F1Measure(positive_label=index)

        self.loss = torch.nn.CrossEntropyLoss()
    
        check_dimensions_match(text_field_embedder.get_output_dim(), self.encoder.get_input_dim(),
                               "text field embedding dim", "encoder input dim")
        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None,
                # pylint: disable=unused-argument
                **kwargs) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens : ``Dict[str, torch.LongTensor]``, required
            The output of ``TextField.as_array()``, which should typically be passed directly to a
            ``TextFieldEmbedder``. This output is a dictionary mapping keys to ``TokenIndexer``
            tensors.  At its most basic, using a ``SingleIdTokenIndexer`` this is: ``{"tokens":
            Tensor(batch_size, num_tokens)}``. This dictionary will have the same keys as were used
            for the ``TokenIndexers`` when you created the ``TextField`` representing your
            sequence.  The dictionary is designed to be passed directly to a ``TextFieldEmbedder``,
            which knows how to combine different word representations into a single vector per
            token in your input.
        label : ``torch.LongTensor``, optional (default = ``None``)
            A torch tensor representing the gold label of shape
            ``(batch_size, )``.
        metadata : ``List[Dict[str, Any]]``, optional, (default = None)
            metadata containg the original words in the sentence to be tagged under a 'words' key.

        Returns
        -------
        An output dictionary consisting of:
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_classes)`` representing a distribution over the
            label classes for each instance.
        loss : ``torch.FloatTensor``, optional
            A scalar loss to be optimised. 
        """
        embedded_text_input = self.text_field_embedder(tokens)
        mask = util.get_text_field_mask(tokens)

        if self.dropout:
            embedded_text_input = self.dropout(embedded_text_input)
        
        # Seq2Seq
        encoded_text = self.encoder(embedded_text_input, mask)

        if self.dropout:
            encoded_text = self.dropout(encoded_text)
        
        # Grab the final state of the seq2seq output
        final_state = util.get_final_encoder_states(encoded_text, mask, bidirectional=True)

        logits = self._feedforward(final_state)
        class_probabilities = F.softmax(logits, dim=1)
        output_dict = {"class_probabilities": class_probabilities}

        if label is not None:
            loss = self.loss(logits, label)
            for metric in self.metrics.values():
                metric(logits, label)
            output_dict["loss"] = loss

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the class probabilities, converts indices to string labels, and
        adds a ``"label"`` key to the dictionary with the result.
        """
        class_probabilities = output_dict['class_probabilities']
        predictions = class_probabilities.cpu().data.numpy()
        argmax_indices = numpy.argmax(predictions, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace="labels")
                  for x in argmax_indices]
        output_dict['label'] = labels
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_return = {"accuracy": self.metrics["accuracy"].get_metric(reset)}

        avg_f1 = 0.                                                             
        label_count = 0.                                                        
        # Get label-specific scores for each entity                             
        for index, label in self.vocab.get_index_to_token_vocabulary("labels").items():
            p, r, f1 = self.metrics[label].get_metric(reset)                    
            p_name = "{0}-p".format(label)                                      
            r_name = "{0}-r".format(label)                                      
            f1_name = "{0}-f1".format(label)                                    
            # Inelegant but it's short and it works.                            
            label_count += 1                                                    
            for score, metric in zip([p, r, f1], [p_name, r_name, f1_name]):    
                metrics_to_return[metric] = score                               
                if metric.endswith('f1'):                                       
                    avg_f1 += score                                             
                                                                                
        # Average across labels                                                 
        metrics_to_return["avg_f1"] = avg_f1 / label_count
        return metrics_to_return
