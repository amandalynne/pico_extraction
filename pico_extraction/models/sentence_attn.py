from typing import Dict, Optional, List, Any
import warnings

import numpy
from overrides import overrides
import torch
import torch.nn.functional as F
from torch.nn.modules.linear import Linear

import allennlp.nn.util as util

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2SeqEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy, F1Measure


@Model.register("pico_sentence_attn")
class PicoSentenceLabeler(Model):
    """
    This ``PicoSentenceLabeler`` encodes a sequence of text with a ``Seq2SeqEncoder``,
    and multiheaded attention weights are computed from the hidden states of this
    encoding.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the tokens ``TextField`` we get as input to the model.
    encoder : ``Seq2SeqEncoder``
        The encoder that we will use in between embedding tokens and predicting output labels.
    feedforward : ``FeedForward``
        Multi-layer perceptron on logits.
    heads: ``int``, required
        The number of attention heads per token.
    dropout:  ``float``, optional (default=``None``)
    do_token_norm: ``bool``, optional (default=True)
        Normalize attention weights across tokens (or not).
    gamma: ``float``, optional (default=0.01)
        Controls importance of auxiliary objectives in loss.
    verbose_metrics : ``bool``, optional (default = False)
        If true, metrics will be returned per label class in addition
        to the overall statistics.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 feedforward: FeedForward,
                 heads: int,
                 dropout: Optional[float] = None,
                 do_token_norm: Optional[bool] = True,
                 gamma: Optional[float] = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.encoder = encoder

        self.num_heads = heads
        self.do_token_norm = do_token_norm

        # Following architecture from Rei et al (2018) paper
        # These come in handy in forward() as we compute attention weights.
        self.token_rep_layer = Linear(self.encoder.get_output_dim(), self.encoder.get_output_dim())
        self.pre_attn_layer = Linear(self.encoder.get_output_dim(), self.num_heads)
        self.attn_layer = Linear(self.num_heads, self.num_heads)
   
        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None

        self._feedforward = feedforward

        if gamma:
            self.gamma = float(gamma)
        else:
            self.gamma = 0.01

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
       
        # Seq2Seq on input text
        encoded_text = self.encoder(embedded_text_input, mask)
       
        # ReLU on encoded text
        # (batch_size, sequence_length, hidden_dim)
        token_rep = F.relu(self.token_rep_layer(encoded_text))
        if self.dropout:
            token_rep = self.dropout(token_rep)

        # do feedforward and relu on token rep to produce unnormalized weights
        # borrowing variable names from Rei paper equations
        # dims: (batch_size, sequence_length, num_heads)
        e = F.relu(self.pre_attn_layer(token_rep))
        a_tilde_unnorm = torch.sigmoid(self.attn_layer(e))

        # Normalization across tokens first:
        # (should sum to 1 across all tokens)
        if self.do_token_norm:
            a_tilde = a_tilde_unnorm / torch.sum(a_tilde_unnorm, dim=1).unsqueeze(1)

        # Linear normalization of attention weights across heads.
        # (so that they sum to 1 over one token)
        # (batch_size, sequence_length, num_heads)
        if self.num_heads > 1:
            attn_weights = a_tilde / torch.sum(a_tilde, dim=2).unsqueeze(2)
            # reshape for matrix multiplication to work later
            # (batch_size, num_heads, sequence_length)
            attn_weights = torch.transpose(attn_weights, 1, 2)
        else:       
            attn_weights = a_tilde.squeeze() / torch.sum(a_tilde, dim=1)

        # Apply attention to sentence
        # (batch_size, [num_heads], hidden_dim)
        sentence_rep = util.weighted_sum(token_rep, attn_weights)

        # MLP and softmax for sentence label prediction
        # (batch_size, num_classes)
        logits = self._feedforward(sentence_rep)
        if len(logits.shape) == 3:
            logits = logits.squeeze(2)
        class_probabilities = F.softmax(logits, dim=1)
        # Also add attention weights to output_dict
        output_dict = {"class_probabilities": class_probabilities,
                       "attn_weights": attn_weights}

        if label is not None:
            loss = self.loss(logits, label)
            # Convert label to one-hot encoding
            one_hot = torch.zeros(logits.shape, device=label.device).scatter_(1, label.unsqueeze(-1), 1)
            # Max. unnormalized attn. weight across tokens
            max_weights = a_tilde_unnorm.transpose(1,2).max(dim=2)[0]
            # Adapted from loss L3 from Rei paper
            l3_loss = torch.sum(torch.sum(((max_weights * one_hot) - one_hot), dim=1)**2)
            loss += (self.gamma * l3_loss)
            
            for metric in self.metrics.values():
                metric(logits, label)
            output_dict["loss"] = loss

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the class probabilities, converts indices to
        string labels, and adds a ``"label"`` key to the dictionary with the result.
        
        Additionally, does argmax over attention weights per token and does a 
        lookup for the class label for that token ("soft labeling").
        """
        class_probabilities = output_dict['class_probabilities']
        predictions = class_probabilities.cpu().data.numpy()
        argmax_index = numpy.argmax(predictions)
        label = self.vocab.get_token_from_index(argmax_index, namespace="labels")
        output_dict['label'] = label
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_return = {"accuracy": self.metrics["accuracy"].get_metric(reset)}

        avg_f1 = 0.                                                            
        label_count = 0.                                                       
        # Get label-specific scores for each element                            
        for index, label in self.vocab.get_index_to_token_vocabulary("labels").items():
            p, r, f1 = self.metrics[label].get_metric(reset)                   
            p_name = "{0}-p".format(label)                                     
            r_name = "{0}-r".format(label)                                     
            f1_name = "{0}-f1".format(label)                                   
            label_count += 1                                                   
            for score, metric in zip([p, r, f1], [p_name, r_name, f1_name]):   
                metrics_to_return[metric] = score                              
                if metric.endswith('f1'):                                      
                    avg_f1 += score                                            
                                                                               
        # Average across labels                                                
        metrics_to_return["avg_f1"] = avg_f1 / label_count
        return metrics_to_return
