from typing import Dict, Optional, List, Any

from overrides import overrides
import torch
from torch.nn.modules.linear import Linear
from torch.nn import Softmax
from torch.nn import CrossEntropyLoss

from allennlp.modules.token_embedders.bag_of_word_counts_token_embedder import BagOfWordCountsTokenEmbedder
from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, Seq2VecEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules import ConditionalRandomField, FeedForward
from allennlp.modules.conditional_random_field import allowed_transitions
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
import allennlp.nn.util as util
from allennlp.training.metrics import CategoricalAccuracy, SpanBasedF1Measure


@Model.register("sem_parser")
class SemParser(Model):
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 label_namespace: str = "logical_form",
                 feedforward: Optional[FeedForward] = None,
                 dropout: Optional[float] = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self.num_tags = max(self.vocab.get_vocab_size("logical_form_1"), 
                            self.vocab.get_vocab_size("logical_form_2"),
                            self.vocab.get_vocab_size("logical_form_3"))
        self.encoder = encoder
        
        self.text_field_embedder = text_field_embedder
        self.BOW_embedder_question = BagOfWordCountsTokenEmbedder(
            vocab, "tokens", projection_dim=self.encoder.get_output_dim())
        self.BOW_embedder_description = BagOfWordCountsTokenEmbedder(
            vocab, "tokens", projection_dim=self.encoder.get_output_dim())
        self.BOW_embedder_detail = BagOfWordCountsTokenEmbedder(
            vocab, "tokens", projection_dim=self.encoder.get_output_dim())
        
        
        # using crf as the estimator for sequential tags
        self.crf = ConditionalRandomField(
            self.num_tags, 
            include_start_end_transitions=False
        )
        
        self.softmax_layer = Softmax()
        self.ce_loss = CrossEntropyLoss()
        
        self.matched = 0
        self.all_pred = 0

        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None
        self._feedforward = feedforward

        if feedforward is not None:
            output_dim = feedforward.get_output_dim()
        else:
            output_dim = self.encoder.get_output_dim()
            
        self.question_pred_layer = Linear(4*output_dim, 3*self.num_tags)
        # if  constrain_crf_decoding and calculate_span_f1 are not
        # provided, (i.e., they're None), set them to True
        # if label_encoding is provided and False if it isn't.

        self.metrics = {}
        check_dimensions_match(text_field_embedder.get_output_dim(), encoder.get_input_dim(),
                               "text field embedding dim", "encoder input dim")
        if feedforward is not None:
            check_dimensions_match(encoder.get_output_dim(), feedforward.get_input_dim(),
                                   "encoder output dim", "feedforward input dim")
        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                entity_surface: Dict[str, torch.LongTensor],
                entity_description: Dict[str, torch.LongTensor],
                entity_detail: Dict[str, torch.LongTensor],
                entity_type: torch.LongTensor = None,
                logical_form_1: torch.LongTensor = None,
                logical_form_2: torch.LongTensor = None,
                logical_form_3: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        embedded_text_input = self.text_field_embedder(question)
        batch_size = int(embedded_text_input.size(0))
        mask = util.get_text_field_mask(question)
        q_vec = self.encoder(embedded_text_input, mask)
        
        bow_question_vec = self.BOW_embedder_question(question['tokens'].unsqueeze(1))
        bow_description_vec = self.BOW_embedder_description(entity_description['tokens'].unsqueeze(1))
        bow_detail_vec = self.BOW_embedder_detail(entity_detail['tokens'].unsqueeze(1))
        
        
        fin_repr = torch.cat([bow_question_vec, bow_description_vec, bow_detail_vec,q_vec],1)
        
        
        pred_logits = self.question_pred_layer(fin_repr).view(batch_size, 3, -1)
        
        
        device_num = pred_logits.get_device()
        if device_num < 0:
            device_num="cpu"
            
        mask = torch.ones((batch_size,3), dtype=torch.long,device=device_num)
        vi_path = self.crf.viterbi_tags(pred_logits, mask)
        pred_result = torch.stack([torch.tensor(x[0],device=device_num) for x in vi_path])
        
        output = {"pred_result": pred_result}
        if logical_form_1 is not None:
            target = torch.stack([logical_form_1, 
                                  logical_form_2,
                                  logical_form_3],dim=1)
            self.matched += int(((pred_result == target).int().sum(dim=-1) == 3).int().sum())
            self.all_pred += int(pred_result.size(0))
            output["loss"] = -self.crf(pred_logits, target, mask)
            
        
        
        # testing
        if metadata is not None:
            output["question_words"] = [x["question_words"] for x in metadata]
            output["entity_description"] = [x["entity_description"] for x in metadata]
        return output

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Converts the tag ids to the actual tags.
        ``output_dict["tags"]`` is a list of lists of tag_ids,
        so we use an ugly nested list comprehension.
        """
        output_dict["tags"] = [
                [self.vocab.get_token_from_index(int(instance_tags[0]), namespace="logical_form_1"),
                 self.vocab.get_token_from_index(int(instance_tags[1]), namespace="logical_form_2"),
                 self.vocab.get_token_from_index(int(instance_tags[2]), namespace="logical_form_3"),
                 ]
                for instance_tags in output_dict["pred_result"]
        ]

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        
        metrics_to_return = {"fullmatch": self.matched/self.all_pred}
        if reset:
            self.matched = 0
            self.all_pred = 0

        return metrics_to_return
