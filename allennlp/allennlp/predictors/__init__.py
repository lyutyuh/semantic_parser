"""
A :class:`~allennlp.predictors.predictor.Predictor` is
a wrapper for an AllenNLP ``Model``
that makes JSON predictions using JSON inputs. If you
want to serve up a model through the web service
(or using ``allennlp.commands.predict``), you'll need
a ``Predictor`` that wraps it.
"""
from allennlp.predictors.predictor import Predictor
from allennlp.predictors.atis_parser import AtisParserPredictor
from allennlp.predictors.biaffine_dependency_parser import BiaffineDependencyParserPredictor
from allennlp.predictors.bidaf import BidafPredictor
from allennlp.predictors.constituency_parser import ConstituencyParserPredictor
from allennlp.predictors.coref import CorefPredictor
from allennlp.predictors.decomposable_attention import DecomposableAttentionPredictor
from allennlp.predictors.dialog_qa import DialogQAPredictor
from allennlp.predictors.event2mind import Event2MindPredictor
from allennlp.predictors.nlvr_parser import NlvrParserPredictor
from allennlp.predictors.open_information_extraction import OpenIePredictor
from allennlp.predictors.quarel_parser import QuarelParserPredictor
from allennlp.predictors.semantic_role_labeler import SemanticRoleLabelerPredictor
from allennlp.predictors.sentence_tagger import SentenceTaggerPredictor
from allennlp.predictors.seq2seq import Seq2SeqPredictor
from allennlp.predictors.simple_seq2seq import SimpleSeq2SeqPredictor
from allennlp.predictors.text_classifier import TextClassifierPredictor
from allennlp.predictors.wikitables_parser import WikiTablesParserPredictor

from allennlp.predictors.sempar_predictor import SemanticParserPredictor