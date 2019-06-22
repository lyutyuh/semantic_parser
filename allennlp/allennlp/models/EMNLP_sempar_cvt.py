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


@Model.register("sem_parser_cvt")
class SemParserCVT(Model):
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 entity_embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 label_namespace: str = "logical_form",
                 feedforward: Optional[FeedForward] = None,
                 dropout: Optional[float] = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self.num_tags = self.vocab.get_vocab_size("logical_form")
        self.encoder = encoder
        
        self.text_field_embedder = text_field_embedder
        self.entity_embedder = entity_embedder

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
        
        self.crf_for_both = ConditionalRandomField(
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

        output_dim = self.encoder.get_output_dim()
            
        self.pred_layer = Linear(4*output_dim, 3*self.num_tags)
        self.load_pretrained_weights()
        
        self.pred_layer_both = Linear(8*output_dim, 3*self.num_tags)
        # if  constrain_crf_decoding and calculate_span_f1 are not
        # provided, (i.e., they're None), set them to True
        # if label_encoding is provided and False if it isn't.

        self.metrics = {}
        check_dimensions_match(text_field_embedder.get_output_dim(), encoder.get_input_dim(),
                               "text field embedding dim", "encoder input dim")
        initializer(self)

    def load_pretrained_weights(self):
        weight_dir = "/home3/chenhongyin/LIU/WORK/semantic_parsing/DUMP/Model_single_relation/best.th"
        state_dict = torch.load(weight_dir)
        self.encoder.load_state_dict({ky[1+len("encoder"):]:val for ky,val in state_dict.items() if ky.startswith("encoder")})
        self.text_field_embedder.load_state_dict({ky[1+len("text_field_embedder"):]:val for ky,val in state_dict.items() if "text_field_embedder" in ky})
        self.BOW_embedder_question.load_state_dict({ky[1+len("BOW_embedder_question"):]:val for ky,val in state_dict.items() if "BOW_embedder_question" in ky})
        self.BOW_embedder_description.load_state_dict({ky[1+len("BOW_embedder_description"):]:val for ky,val in state_dict.items() if "BOW_embedder_description" in ky})
        self.BOW_embedder_detail.load_state_dict({ky[1+len("BOW_embedder_detail"):]:val for ky,val in state_dict.items() if "BOW_embedder_detail" in ky})
        # using crf as the estimator for sequential tags
        self.crf.load_state_dict({ky[4:]:val for ky,val in state_dict.items() if "crf" in ky})
        self.pred_layer.load_state_dict({ky[1+len("question_pred_layer"):]:val for ky,val in state_dict.items() if "question_pred_layer" in ky})
        return
        
    @overrides
    def forward(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                entity_surface: Dict[str, torch.LongTensor],
                entity_description: Dict[str, torch.LongTensor],
                entity_detail: Dict[str, torch.LongTensor],
                entity_type: Dict[str, torch.LongTensor] = None,
                logical_form_1: torch.LongTensor = None,
                logical_form_2: torch.LongTensor = None,
                logical_form_both: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        
        
        embedded_text_input = self.text_field_embedder(question)
        
        split_entity_surface = [{}, {}]
        for x in entity_surface:
            split_entity_surface[0][x] = entity_surface[x][:,0]
            split_entity_surface[1][x] = entity_surface[x][:,1]
            
        embedded_entity_surface_1 = self.entity_embedder(split_entity_surface[0])
        embedded_entity_surface_2 = self.entity_embedder(split_entity_surface[1])
        
        batch_size = int(embedded_text_input.size(0))
        mask = util.get_text_field_mask(question)
        q_vec = self.encoder(embedded_text_input, mask)
        
        bow_question_vec = self.BOW_embedder_question(question['tokens'].unsqueeze(1))
        
        bow_description_1_vec = self.BOW_embedder_description(entity_description['tokens'][:,0,:].unsqueeze(1))
        bow_detail_1_vec = self.BOW_embedder_detail(entity_detail['tokens'][:,0,:].unsqueeze(1))        
        bow_description_2_vec = self.BOW_embedder_description(entity_description['tokens'][:,1,:].unsqueeze(1))
        bow_detail_2_vec = self.BOW_embedder_detail(entity_detail['tokens'][:,1,:].unsqueeze(1))
        
        fin_repr_1 = torch.cat([bow_question_vec, bow_description_1_vec, bow_detail_1_vec,q_vec],1)
        fin_repr_2 = torch.cat([bow_question_vec, bow_description_2_vec, bow_detail_2_vec,q_vec],1)
        fin_repr_both = torch.cat([fin_repr_1, fin_repr_2],1)
        
        pred_logits_1 = self.pred_layer(fin_repr_1).view(batch_size, 3, -1)
        pred_logits_2 = self.pred_layer(fin_repr_2).view(batch_size, 3, -1)
        pred_logits_both = self.pred_layer_both(fin_repr_both).view(batch_size, 3, -1)
        
        device_num = pred_logits_1.get_device()
        if device_num < 0:
            device_num="cpu"
            
        mask = torch.ones((batch_size,3), dtype=torch.long,device=device_num)
        
        vi_path_1 = self.crf.viterbi_tags(pred_logits_1, mask)
        vi_path_2 = self.crf.viterbi_tags(pred_logits_2, mask)
        vi_path_both = self.crf_for_both.viterbi_tags(pred_logits_both, mask)
        
        
        pred_result_1 = torch.stack([torch.tensor(x[0],device=device_num) for x in vi_path_1])
        pred_result_2 = torch.stack([torch.tensor(x[0],device=device_num) for x in vi_path_2])
        pred_result_both = torch.stack([torch.tensor(x[0],device=device_num) for x in vi_path_both])
        
        output = {"pred_result_1": pred_result_1,
                  "pred_result_2": pred_result_2,
                  "pred_result_both": pred_result_both}
        if logical_form_1 is not None:
            self.matched += int((((pred_result_1 == logical_form_1).int() + (pred_result_2 == logical_form_2).int() + (pred_result_both == logical_form_both).int()).sum(dim=-1) == 9).int().sum())
            self.all_pred += int(pred_result_1.size(0))
            
            output["loss"] = -(self.crf(pred_logits_1, logical_form_1, mask) + self.crf(pred_logits_2, logical_form_2, mask) + self.crf_for_both(pred_logits_both, logical_form_both, mask))
            
        
        
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
                [self.vocab.get_token_from_index(int(instance_tags[0]), namespace="logical_form"),
                 self.vocab.get_token_from_index(int(instance_tags[1]), namespace="logical_form"),
                 self.vocab.get_token_from_index(int(instance_tags[2]), namespace="logical_form_both"),
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
