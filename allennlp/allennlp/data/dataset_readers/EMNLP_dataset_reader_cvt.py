from typing import Dict, List, Sequence, Iterable
import itertools
import logging
import pickle
import re

from overrides import overrides

from nltk.tokenize import word_tokenize
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.dataset_utils import to_bioul
from allennlp.data.fields import TextField, SequenceLabelField, Field, MetadataField, MultiLabelField, ListField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def instance2dict(instance, question_type):
    tmp_dict = {}
    tmp_dict['question_type'] = instance[3].split('\t')[1].strip()
    if tmp_dict['question_type'] != question_type:
        return tmp_dict 
	# convert the instance into a dict
    question_surface = instance[0].split('\t')[1].strip()
    tmp_dict['question'] = [Token(token) for token in word_tokenize(" ".join(question_surface.split()))]
    
    logical_form_rexpr_pattern = r"[\(] [r\-]*mso:([\w\.]*) (([\?][xy])|[\S]+) (([\?][xy])|[\S]+) [\)]"
    entity_rexpr_pattern = r"([\S]+) (?:\(entity\)|\(value\)|\(type\)) \[([\d]+,[\d]+)\]"
    
    logical_form = instance[1].split('\t')[1].strip()
        
    # 2 parameters in cvt
    param_matches = re.findall(entity_rexpr_pattern, instance[2].split('\t')[1].strip())
    # entity 0, entity 1
    tmp_dict['parameters'] = [param_matches[0][0], param_matches[1][0]]
    
    ent_span = (eval("("+param_matches[0][1]+")"),eval("("+param_matches[1][1]+")"))
    tmp_dict['parameters_surface'] = [question_surface.split()[slice(ent_span[0][0], ent_span[0][1]+1)],
                                      question_surface.split()[slice(ent_span[1][0], ent_span[1][1]+1)]]
    tmp_dict['parameters_surface'] = [[Token(token) for token in word_tokenize(" ".join(tmp_dict['parameters_surface'][0]))],
                                      [Token(token) for token in word_tokenize(" ".join(tmp_dict['parameters_surface'][1]))]]
    
    try:
        # 3 logical forms in cvt
        log_form_match = re.findall(logical_form_rexpr_pattern, logical_form)
        if tmp_dict['parameters'][0] in log_form_match[1]:
            # matching logical form with parameter
            log_form_match[0], log_form_match[1] = log_form_match[1], log_form_match[0]
        tmp_dict['logical_form'] = [x[0].split(".") for x in log_form_match]
    except:
        tmp_dict['logical_form'] = []
    
    
    return tmp_dict


@DatasetReader.register("EMNLPDatasetReaderCVT")
class EMNLPDatasetReaderCVT(DatasetReader):

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False,
                 KB_path: str = "",
                 label_namespace: str = "labels",
                 question_type: str = "cvt") -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.KB_path = KB_path
        self.label_namespace = label_namespace
        assert question_type == "cvt" # ...
        self.question_type = question_type

    @overrides
    def _read(self, 
              file_path: str) -> Iterable[Instance]:
        # KB_path should be a pickle file of a dictionary
        # if `file_path` is a URL, redirect to the cache
        KB_path = self.KB_path
        file_path = cached_path(file_path)
        dict_entity_lookup = pickle.load(open(KB_path,"rb"))

        with open(file_path, "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            
            tmp_instance = []
            for line in data_file.readlines():
                if line.strip()=="==================================================":
                    instance_dict = instance2dict(tmp_instance, self.question_type)
                    if instance_dict["question_type"] != self.question_type:
                        tmp_instance = []
                        continue
                    
                    question = instance_dict['question']
                    entity_surface = instance_dict['parameters_surface']
                    
                    entity = instance_dict['parameters']
                    e_type = []
                    e_descr = []
                    e_detail = []
                    
                    for x in entity:
                        try:
                            KB_gloss = dict_entity_lookup.get(x,{}).get("itemListElement")[0]
                        except:
                            KB_gloss = {}
                        # useful fields: @type -> list, description, detailedDescription
                        e_type.append([Token(token) for token in KB_gloss.get("result",{}).get("@type","")])
                        e_descr.append([Token(token) for token in word_tokenize(KB_gloss.get("result",{}).get("description",""))])
                        e_detail.append([Token(token) for token in word_tokenize(KB_gloss.get("result",{}).get("detailedDescription",{}).get("articleBody",""))])
                    
                    logical_form = instance_dict['logical_form']
                    assert len(entity) == 2
                    assert len(e_type)==2 and len(e_descr) == 2 and len(e_detail)==2
                    yield self.text_to_instance(question, entity, entity_surface, e_type, e_descr, e_detail, logical_form)
                    tmp_instance = []
                    e_type = []
                    e_descr = []
                    e_detail = []
                else:
                    tmp_instance.append(line)
                    
                    
    def text_to_instance(self, # type: ignore
                         question: List[Token],
                         entity: List[str],
                         entity_surface: List[List[Token]],
                         e_type: List[List[Token]] = None,
                         e_descr: List[List[Token]] = None,
                         e_detail: List[List[Token]] = None, 
                         logical_form: List[List[str]] = None) -> Instance:
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.
        """
        # pylint: disable=arguments-differ
        sequence = TextField(question, self._token_indexers)
        entity_sequence = ListField([TextField(x, self._token_indexers) for x in entity_surface])
        description = ListField([TextField(x, self._token_indexers) for x in e_descr])
        detail = ListField([TextField(x, self._token_indexers) for x in e_detail])
        
        instance_fields: Dict[str, Field] = {'question': sequence, 
                                             'entity_surface':entity_sequence, 
                                             "entity_description":description, 
                                             "entity_detail":detail}
        
        instance_fields["metadata"] = MetadataField({"question_words": [x.text for x in question],
                                                     "entity_surface": [x.text for y in entity_sequence for x in y],
                                                     "entity_description": [x.text for y in description for x in y],
                                                     "entity_detail": [x.text for y in detail for x in y]})

        instance_fields['entity_type'] = ListField([TextField(x, self._token_indexers) for x in e_type])
        instance_fields['entity'] = ListField([LabelField(x, "entity") for x in entity])
        if len(logical_form) > 0:
            instance_fields['logical_form_1'] = ListField([LabelField(x, "logical_form") for x in logical_form[0]])
            instance_fields['logical_form_2'] = ListField([LabelField(x, "logical_form") for x in logical_form[1]])
            instance_fields['logical_form_both'] = ListField([LabelField(x, "logical_form") for x in logical_form[2]])
        else:
            pass

        

        return Instance(instance_fields)
