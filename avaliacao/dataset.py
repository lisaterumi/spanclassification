# coding: utf-8
# created by deng on 2019-01-19

import os
import copy
import csv
import torch
import joblib
import numpy as np
import json
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import logging
from torch.utils.data import TensorDataset

import utils.json_util as ju
import utils.utils as ju2

from utils.utils import get_label, load_tokenizer

from utils.path_util import from_project_root, dirname

from importlib import reload 
reload(ju2)


logger = logging.getLogger(__name__)

#COLAB
DATA_DIR='./data'
#LOCAL
#DATA_DIR='./model/data'
LABEL_FILE='label.txt'
TRAIN_FILE = "span.train"
DEV_FILE = "span.test"
TEST_FILE = "span.test"
PREDICT_FILE = "span.predict"

class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        sentence: string. The untokenized sentence withous special tags
        list_text_a: list of strings. All untokenized sentence with special tags.
        list_label: (Optional) list of string. The label of the list_text_a example. This should be
        specified for train and dev examples, but not for test examples.
    """

    def __init__(self, sentence, list_text_e1, list_label):
        self.sentence = sentence
        self.list_text_e1 = list_text_e1
        self.list_label = list_label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class Listatokens(object):
    def __init__(self, list_tokens, list_indices, list_labels):
        self._list_tokens = list_tokens
        self._list_indices = list_indices
        self._list_labels = list_labels
    
    @property
    def list_tokens(self):
         # self.list_tokens
         return self._list_tokens

    @list_tokens.setter
    def list_tokens(self, value):
         # self.list_tokens = value
         self._list_tokens = value

    @property
    def list_indices(self):
         return self._list_indices

    @list_indices.setter
    def list_indices(self, value):
         self._list_indices = value

    @property
    def list_labels(self):
         return self._list_labels

    @list_labels.setter
    def list_labels(self, value):
         self._list_labels = value
class InputFeatures(object):
    """
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        lista_label_id: lista com labels de cada regiao
        lista_indices_e1: lista com indices onde estão as entidades
    """

    #def __init__(self, input_ids, attention_mask, token_type_ids, label_id, e1_mask, e2_mask):
    def __init__(self, model_name_or_path, max_seq_len, device, mode, labels):
        super().__init__()
        #self.input_ids, self.attention_mask, self.token_type_ids, self.lista_label_id, self.lista_indices_e1, self.lista_tokens = load_and_cache_examples(model_name_or_path, max_seq_len, device=device, mode=mode)
        self.input_ids, self.attention_mask, self.token_type_ids, self.lista_tokens = load_and_cache_examples(model_name_or_path, max_seq_len, device=device, mode=mode)
        self.device = device
        self.mode = mode
        #labels.append('NA')
        self.labels = labels
        self.n_tags = len(labels)
        
    def __getitem__(self, index):
        return self.input_ids[index], self.attention_mask[index], self.token_type_ids[index], self.lista_tokens[index]

    def __len__(self):
        return len(self.input_ids)

    def collate_func(self, features):
        all_input_ids = torch.tensor([f[0] for f in features], dtype=torch.long).to(self.device)
        all_attention_mask = torch.tensor([f[1] for f in features], dtype=torch.long).to(self.device)
        all_token_type_ids = torch.tensor([f[2] for f in features], dtype=torch.long).to(self.device)
        all_tokens = [f[3] for f in features]

        # passar as labels das regioes tudo em uma lista só
        '''
        new_list_lista_label_id = []
        for lista_label_id in all_lista_label_id:
            for label in lista_label_id:
                new_list_lista_label_id.append(label)
        
        new_list_lista_label_id = torch.LongTensor(new_list_lista_label_id).to(self.device)
        all_lista_indices_e1 = torch.LongTensor(all_lista_indices_e1).to(self.device)
        '''
        #return all_input_ids, all_attention_mask, all_token_type_ids, new_list_lista_label_id, all_lista_indices_e1, all_tokens 
        return all_input_ids, all_attention_mask, all_token_type_ids, all_tokens 
        
class ChunkProcessor(object):
    """Processor for the Semeval data set """

    def __init__(self, data_dir, label_file, train_file, dev_file, test_file, predict_file):
        self.train_file = train_file
        self.dev_file = dev_file
        self.test_file = test_file
        self.predict_file = predict_file
        self.data_dir = data_dir
        self.labels = get_label(data_dir, label_file)

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
        #with open(input_file, "r", encoding="latin1") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        sentence = ''
        list_label = []
        list_text_e1 = []
        for (i, line) in enumerate(lines):
            if i % 1000 == 0:
                logger.info(line)
            if len(line)==1:
                # nao tem label, entao é uma sentenca
                # salvando os dados da sentenca anterior
                if sentence!='':
                  examples.append(InputExample(sentence=sentence, list_text_e1=list_text_e1, list_label=list_label))              
                list_label = []
                list_text_e1 = []
                sentence = line[0]
            else:
                label = self.labels.index(line[0])
                list_label.append(label)
                text_e1 = line[1]
                list_text_e1.append(text_e1)
        examples.append(InputExample(sentence=sentence, list_text_e1=list_text_e1, list_label=list_label))              

        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test, predict
        """
        file_to_read = None
        if mode == "train":
            file_to_read = self.train_file
        elif mode == "dev":
            file_to_read = self.dev_file
        elif mode == "test":
            file_to_read = self.test_file
        elif mode == "predict":
            file_to_read = self.predict_file
            

        logger.info("LOOKING AT {}".format(os.path.join(self.data_dir, file_to_read)))
        print("LOOKING AT {}".format(os.path.join(self.data_dir, file_to_read)))
        return self._create_examples(self._read_tsv(os.path.join(self.data_dir, file_to_read)), mode)

processors = {"chunk": ChunkProcessor}

# examples = lista de InputExample(sentence=sentence, list_text_e1=list_text_e1, list_label=list_label)
def convert_examples_to_features(
    examples,
    max_seq_len,
    model_name_or_path,
):
    label_lst = get_label(DATA_DIR, LABEL_FILE)
    #print('label_lst:', label_lst)
    cls_token="[CLS]"
    sep_token="[SEP]"
    cls_token_segment_id=0
    pad_token=0
    pad_token_segment_id=0
    sequence_a_segment_id=0
    mask_padding_with_zero=True
    #add_sep_token=False # TODO
    add_sep_token=True # TODO

    list_input_ids=list()
    list_attention_mask=list()
    list_token_type_ids=list()
    list_lista_label_id=list()
    list_lista_indices_e1=list()
    list_token_obj=list() # lista do objeto ListaToken
    list_tokens=list()

    #new_list_lista_label_id = list()

    numMaxRegioes=0
    tokenizer = load_tokenizer(model_name_or_path)
    #print('len(examples):', len(examples))
    cont_punlando=0
    for (ex_index, example) in enumerate(examples):
        indices_e1_sentenca = []
        lista_label = []
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        sentence_tok = tokenizer.tokenize(example.sentence)

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        if add_sep_token:
            special_tokens_count = 2
        else:
            special_tokens_count = 1

        maxLen = max_seq_len - special_tokens_count
        #maxLen = max_seq_len - special_tokens_count -1 #-1 pelo token de PAD no final
        if len(sentence_tok) > maxLen:
            sentence_tok = sentence_tok[:maxLen]

        tokens = sentence_tok
        if add_sep_token:
            tokens += [sep_token]

        # add PAD token no final
        #tokens += [pad_token]

        token_type_ids = [sequence_a_segment_id] * len(tokens)

        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(
            len(attention_mask), max_seq_len
        )
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(
            len(token_type_ids), max_seq_len
        )

        list_text_a=[]
        list_text_a_tokenizado=[]
        # para cada candidato (regiao) da sentenca
        for text_a, label in zip(example.list_text_e1, example.list_label):
            list_text_a.append(text_a)
            list_tokens.append(text_a)
            
            
            tokens_a = tokenizer.tokenize(text_a)
            list_text_a_tokenizado.append(tokens_a)
            e11_p = tokens_a.index("<e1>")  # the start position of entity1
            e12_p = tokens_a.index("</e1>")  # the end position of entity1

            # Add 1 because of the [CLS] token
            e11_p += 1

            # Sub 1 - pq o proprio indice tbm conta
            e12_p -= 1

            # se a posticao da entidade é maior que o tamanho maximo, entao ficara de fora
            # tirando entao
            if e12_p > maxLen:
                #print('e12_p > maxLen:', e12_p > maxLen)
                cont_punlando = cont_punlando+1
                continue

            indices_e1 = [e11_p, e12_p]
            indices_e1_sentenca.append(indices_e1)
            #new_list_lista_label_id.append(label)
            #new_list_lista_indices_e1.append(indices_e1)

            lista_label.append(label)
        list_token_obj.append(Listatokens(list_tokens, indices_e1_sentenca, lista_label))
        list_tokens=list()
        indices_e1_sentenca=list()
        lista_label=list()

        if ex_index < 2:
            print("*** Example ***")
            #print("tokens: %s" % " ".join([str(x) for x in tokens]))
            print("tokens:", tokens)
            #print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            #print("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            #print("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            #print("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            print("lista_label:", lista_label)
            print("indices_e1_sentenca:", indices_e1_sentenca)
            #print("list_text_a:", list_text_a)
            #print("list_text_a_tokenizado:", list_text_a_tokenizado)
                        
        if numMaxRegioes<len(lista_label):
            numMaxRegioes=len(lista_label)
        
        list_input_ids.append(input_ids)
        list_attention_mask.append(attention_mask)
        list_token_type_ids.append(token_type_ids)
        list_lista_label_id.append(lista_label)
        list_lista_indices_e1.append(indices_e1_sentenca)

    if cont_punlando>0:
        print('cont_pulando:', cont_punlando)
    #print('numMaxRegioes:', numMaxRegioes)

    # aqui - colocar padding nos labels e indices..
    # labels - usar indice = len(num_labels)
    # regiao - o q usar?? -1? ai se o indice for negativo, coloca o token de <PAD>... seria bom fazer agora entao? ai depois 
    # nao preciso me preocupar..

    n_labels = len(label_lst) # label inexistente
    #for lista_label_regiao, lista_indices_regiao in zip(list_lista_label_id, list_lista_indices_e1):
    #    new_list_lista_label_id.append(lista_label_regiao)
    #    new_list_lista_indices_e1.append(lista_indices_regiao)
    '''
    for lista_label_regiao, lista_indices_regiao in zip(list_lista_label_id, list_lista_indices_e1):
        num_regioes_frase = len(lista_label_regiao)
        lista_label_regiao_cp = lista_label_regiao.copy()
        lista_indices_regiao_cp = lista_indices_regiao.copy()
        if num_regioes_frase < numMaxRegioes:
            #complementar com PAD
            for i in range(num_regioes_frase, numMaxRegioes):
                regiao_pad = n_labels
                lista_label_regiao_cp.append(regiao_pad)
                id_pad = [-1, -1]
                lista_indices_regiao_cp.append(id_pad)
            
        new_list_lista_label_id.append(lista_label_regiao_cp)
        new_list_lista_indices_e1.append(lista_indices_regiao_cp)
    '''
    new_list_lista_indices_e1 = list()
    for lista_indices_regiao in list_lista_indices_e1:
        num_regioes_frase = len(lista_indices_regiao)
        lista_indices_regiao_cp = lista_indices_regiao.copy()
        if num_regioes_frase < numMaxRegioes:
            #complementar com PAD
            for i in range(num_regioes_frase, numMaxRegioes):
                id_pad = [-1, -1]
                lista_indices_regiao_cp.append(id_pad)
            
        new_list_lista_indices_e1.append(lista_indices_regiao_cp)

    
    return list_input_ids, list_attention_mask, list_token_type_ids, list_token_obj


def load_and_cache_examples(model_name_or_path, max_seq_len, device, mode):
    processor = processors['chunk'](DATA_DIR, LABEL_FILE, TRAIN_FILE, DEV_FILE, TEST_FILE, PREDICT_FILE)

    # Load data features from cache or dataset file
    '''
    cached_features_file = os.path.join(
        DATA_DIR,
        "cached_{}_{}_{}_{}".format(
            mode,
            list(filter(None, model_name_or_path.split("/"))).pop(),
            max_seq_len
        ),
    )
    

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
    '''

    logger.info("Creating features from dataset file at %s", DATA_DIR)
    if mode == "train":
        examples = processor.get_examples("train")
    elif mode == "dev":
        examples = processor.get_examples("dev")
    elif mode == "test":
        examples = processor.get_examples("test")
    elif mode == "predict":
        examples = processor.get_examples("predict")
    else:
        raise Exception("For mode, Only train, dev, test is available")

    features = convert_examples_to_features(examples, max_seq_len, model_name_or_path)
    return features


def main():
    #format_time(time.time())
    #start_time = time.time()
    print('os.getcwd():', os.getcwd())
    label_lst = get_label(DATA_DIR, LABEL_FILE)
    test_set = InputFeatures('pucpr/biobertpt-all', 256, device='cpu', mode='test', labels=label_lst)

if __name__ == '__main__':
    main()

