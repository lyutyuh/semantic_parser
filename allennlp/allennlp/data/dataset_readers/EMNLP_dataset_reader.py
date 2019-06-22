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


def instance2dict(instance):
	# convert the instance into a dict
    tmp_dict = {}
    tmp_dict['question'] = [Token(token) for token in word_tokenize(" ".join(instance[0].split('\t')[1].strip().split()))]
    
    logical_form_rexpr_pattern = r"[\(] [r\-]*mso:([\w\.]*) (([\?][xy])|[\S]+) (([\?][xy])|[\S]+) [\)]"
    entity_rexpr_pattern = r"([\S]+) (?:\(entity\)|\(value\)|\(type\)) \[([\d]+,[\d]+)\]"
    
    logical_form = instance[1].split('\t')[1].strip()
    
    try:
        tmp_dict['logical_form'] = re.findall(logical_form_rexpr_pattern, logical_form)[0][0].split(".")
    except:
        tmp_dict['logical_form'] = None
    
    tmp_dict['parameters'] = re.findall(entity_rexpr_pattern, instance[2].split('\t')[1].strip())[0][0]
    
    ent_span = eval("("+re.findall(entity_rexpr_pattern, instance[2].split('\t')[1].strip())[0][1]+")")
    tmp_dict['parameters_surface'] = instance[0].split('\t')[1].strip().split()[slice(ent_span[0], ent_span[1]+1)]
    tmp_dict['parameters_surface'] = [Token(token) for token in word_tokenize(" ".join(tmp_dict['parameters_surface']))]
    
    tmp_dict['question_type'] = instance[3].split('\t')[1].strip()
    return tmp_dict

def fetch_data(file):
	# fetch data from the original file
	lines = file.readlines()
	instance_list = []
	tmp_instance = []
	for line in file.readlines():
		if line.strip()=="==================================================":
			instance_list.append(instance2dict(tmp_instance))
			tmp_instance = []
		else:
			tmp_instance.append(line)
	return instance_list


@DatasetReader.register("EMNLPDatasetReader")
class EMNLPDatasetReader(DatasetReader):

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False,
                 KB_path: str = "",
                 label_namespace: str = "labels",
                 question_type: str = "single-relation") -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.KB_path = KB_path
        self.label_namespace = label_namespace
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
                    instance_dict = instance2dict(tmp_instance)
                    
                    question = instance_dict['question']
                    entity_surface = instance_dict['parameters_surface']
                    
                    entity = instance_dict['parameters']
                    try:
                        KB_gloss = dict_entity_lookup[entity]["itemListElement"][0]
                        # useful fields: @type -> list, description, detailedDescription
                        e_type = KB_gloss["result"].get("@type",[])
                        e_descr = [Token(token) for token in word_tokenize(KB_gloss["result"].get("description",[]))]
                        e_detail = [Token(token) for token in word_tokenize(KB_gloss["result"].get("detailedDescription",{}).get("articleBody",[]))]
                    except:
                        KB_gloss = None
                        e_type = []
                        e_descr = []
                        e_detail = []
                    
                    logical_form = instance_dict['logical_form']
                    if instance_dict["question_type"] == self.question_type:
                        yield self.text_to_instance(question, entity, entity_surface, e_type, e_descr, e_detail, logical_form)
                    tmp_instance = []
                else:
                    tmp_instance.append(line)
                    
                    
    def text_to_instance(self, # type: ignore
                         question: List[Token],
                         entity: str,
                         entity_surface: List[Token],
                         e_type: List[str] = None,
                         e_descr: List[Token] = None,
                         e_detail: List[Token] = None, 
                         logical_form: List[str] = None) -> Instance:
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.
        """
        # pylint: disable=arguments-differ
        sequence = TextField(question, self._token_indexers)
        entity_sequence = TextField(entity_surface, self._token_indexers)
        description = TextField(e_descr, self._token_indexers)
        detail = TextField(e_detail, self._token_indexers)
        
        instance_fields: Dict[str, Field] = {'question': sequence, 
                                             'entity_surface':entity_sequence, 
                                             "entity_description":description, 
                                             "entity_detail":detail}
        instance_fields["metadata"] = MetadataField({"question_words": [x.text for x in question],
                                                     "entity_surface": [x.text for x in entity_sequence],
                                                     "entity_description": [x.text for x in description],
                                                     "entity_detail": [x.text for x in detail]})

        instance_fields['entity_type'] = MultiLabelField(e_type, "entity_type")
        instance_fields['entity'] = LabelField(entity, "entity")
        if logical_form is not None:
            instance_fields['logical_form_1'] = LabelField(logical_form[0], "logical_form_1")
            instance_fields['logical_form_2'] = LabelField(logical_form[1], "logical_form_2")
            instance_fields['logical_form_3'] = LabelField(logical_form[2], "logical_form_3")
        else:
            pass

        

        return Instance(instance_fields)
