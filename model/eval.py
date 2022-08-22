# coding: utf-8
# created by deng on 2019-03-22

import torch
import io
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForTokenClassification, BertConfig
#from transformers.models.bert.modeling_bert import BertModel,BertForMaskedLM

import torch.nn.functional as F
from model import BertForChunkClassification
from dataset import InputFeatures, load_and_cache_examples
from utils.torch_util import calc_f1, get_device
from utils.utils import get_label
from utils.path_util import from_project_root

# COLAB:
BERT_MODEL ='pucpr/biobertpt-clin'
#BERT_MODEL ='pucpr/biobertpt-all'
#BERT_MODEL ='bert-base-uncased'
#BERT_MODEL_NER ='lisaterumi/genia-biobert-ent'
#BERT_MODEL_NER ='./model2'
#BERT_MODEL ='bert-base-cased'
MAX_SEQ_LEN = 256
#MAX_SEQ_LEN = 300
# COLAB:
#DATA_DIR='./data/'
# LOCAL
DATA_DIR='./model/data/'
LABEL_FILE='label.txt'
HIDDEN_SIZE = 768
#COLAB
BATCH_SIZE=200
#LOCAL
#BATCH_SIZE=10
#BERT_MODEL ='./data/model/'

#bert_model -> apenas para tokenizador
def evaluate(model, bert_model, mode="dev", batch_size=BATCH_SIZE):
    """ evaluating end2end model on dataurl

    Args:
        model: trained end2end model
        data_url: url to test dataset for evaluating
        bsl_model: trained binary sequence labeling model
        batch_size: batch_size when predicting

    Returns:
        ret: dict of precision, recall, and f1

    """
    print("\nevaluating model on:", mode, "\n")
    device = get_device('auto')
    label_lst = get_label(DATA_DIR, LABEL_FILE)
    eval_set = InputFeatures(bert_model, MAX_SEQ_LEN, device=device, mode=mode, labels=label_lst)
    loader = DataLoader(eval_set, batch_size=batch_size, collate_fn=eval_set.collate_func)
    ret = {'precision': 0, 'recall': 0, 'f1': 0}

    region_true_list, region_pred_list = list(), list()
    region_true_count, region_pred_count = 0, 0

    # switch to eval mode
    # data = input_ids, attention_mask, token_type_ids, lista_e1_mask
    model.eval()
    model.to(device)
    criterion = F.cross_entropy

    total_loss = 0
    with torch.no_grad():
        for all_input_ids, all_attention_mask, all_token_type_ids, all_tokens in loader:
            try:
                all_indices = list()
                labels = list()
                for listatoken in all_tokens:
                    indices = list()
                    for indice in listatoken.list_indices:
                        indices.append(indice)
                    all_indices.append(indices)
                    for label in listatoken.list_labels:
                        labels.append(label)
                pred_region_output = model.forward(all_input_ids, all_attention_mask, all_token_type_ids, lista_indices_e1=all_indices)
            except RuntimeError:
                print("all 0 tags, no evaluating this epoch")
                return ret, 0

            labels_tensor = torch.LongTensor(labels)
            loss = criterion(pred_region_output, labels_tensor)    
            total_loss += loss.item()

            pred_region_labels = []  # for all tokens are not in-entity
            if len(pred_region_output) > 0:
                pred_region_labels = torch.argmax(pred_region_output, dim=1).to(device)

            for label_true_individual, label_pred_individual in zip(labels, pred_region_labels):
                #  label_true e label_pred -> n_regiao labels
                if label_true_individual==len(label_lst): # se é igual n_tags é pad
                    continue
                if label_pred_individual==len(label_lst): # se caiu na label do padding, considera 0
                    label_pred_individual = 0
                if label_true_individual==0 and label_pred_individual==0: # se os dois são zero, nao contabilza
                    continue
                if label_true_individual>0:
                    region_true_count += 1
                if label_pred_individual>0:
                    region_pred_count += 1

                # se chegou até aqui, é pq precisamos adicionar nos arrays.. pq pelo menos um deles tem valor...
                region_true_list.append(eval_set.labels[int(label_true_individual)])
                region_pred_list.append(eval_set.labels[int(label_pred_individual)])

        #print('len(region_true_list:)', len(region_true_list))
        #print('len(region_pred_list:)', len(region_pred_list))

        print(classification_report(region_true_list, region_pred_list,labels=eval_set.labels,
                                    target_names=eval_set.labels, digits=6))        
        ret = dict()
        tp = 0
        for pv, tv in zip(region_pred_list, region_true_list):
            if pv == tv:
                tp += 1
        fp = region_pred_count - tp
        fn = region_true_count - tp
        ret['precision'], ret['recall'], ret['f1'] = calc_f1(tp, fp, fn)
        ret['accuracy'] = accuracy_score(region_true_list, region_pred_list)

        print('ret:', ret)

        # Calculate the average loss over the training data.
        avg_val_loss = total_loss / len(loader)  

    return ret, avg_val_loss

def evaluateMatriz(model, bert_model, mode, batch_size=BATCH_SIZE):
    """ avalia sem excluir entidade O

    Returns:
        ret: dict of precision, recall, and f1

    """
    print("\nevaluating model on:", mode, "\n")
    device = get_device('auto')
    label_lst = get_label(DATA_DIR, LABEL_FILE)
    eval_set = InputFeatures(bert_model, MAX_SEQ_LEN, device=device, mode=mode, labels=label_lst)
    loader = DataLoader(eval_set, batch_size=batch_size, collate_fn=eval_set.collate_func)
    ret = {'precision': 0, 'recall': 0, 'f1': 0}

    region_true_list, region_pred_list = list(), list()

    # switch to eval mode
    # data = input_ids, attention_mask, token_type_ids, lista_e1_mask
    model.eval()
    model.to(device)
    save_url = 'predictions_labels2.txt'

    with torch.no_grad():
        #for all_input_ids, all_attention_mask, all_token_type_ids, region_labels, indices, tokens in loader:
        for all_input_ids, all_attention_mask, all_token_type_ids, all_tokens in loader:
            print('começando um novo batch')
            try:
                all_indices = list()
                labels = list()
                tokens=list()
                for listatoken in all_tokens:
                    indices = list()
                    for indice in listatoken.list_indices:
                        indices.append(indice)
                    all_indices.append(indices)
                    for label in listatoken.list_labels:
                        labels.append(label)
                    for token in listatoken.list_tokens:
                        tokens.append(token)
                pred_region_output = model.forward(all_input_ids, all_attention_mask, all_token_type_ids, lista_indices_e1=all_indices)
            except RuntimeError:
                print("all 0 tags, no evaluating this epoch")
                return ret, 0

            pred_region_labels = []  # for all tokens are not in-entity
            if len(pred_region_output) > 0:
                pred_region_labels = torch.argmax(pred_region_output, dim=1).to(device)

            for label_true_individual, label_pred_individual in zip(labels, pred_region_labels):
                region_true_list.append(eval_set.labels[int(label_true_individual)])
                region_pred_list.append(eval_set.labels[int(label_pred_individual)])

            with open(save_url, 'a', encoding='utf-8', newline='\n') as save_file:
              for texto, value, label in zip(tokens, pred_region_labels, labels):
                  save_file.write("{} - ".format(value))
                  save_file.write("{} - ".format(label))
                  save_file.write("{}\n".format(texto))
                  #save_file.write(value+'\t'+label+'\n')

        ret = confusion_matrix(region_true_list, region_pred_list, labels=eval_set.labels)
        print(ret)
        print(classification_report(region_true_list, region_pred_list,labels=eval_set.labels,
                                    target_names=eval_set.labels, digits=6))        


    return ret

def predict(model, bert_model, mode, batch_size=BATCH_SIZE):
    """ predict NER result for sentence list
    Args:
        model: trained end2end model
        data_url: url to test dataset for evaluating
        bsl_model: trained binary sequence labeling model
        batch_size: batch_size when predicting
    """

    print("\npediction model on:", mode, "\n")
    device = get_device('auto')
    label_lst = get_label(DATA_DIR, LABEL_FILE)
    test_set = InputFeatures(bert_model, MAX_SEQ_LEN, device=device, mode="test", labels=label_lst)
    loader = DataLoader(test_set, batch_size=batch_size, collate_fn=test_set.collate_func)

    # data = input_ids, attention_mask, token_type_ids, lista_e1_mask
    model.eval()
    save_url = 'predictions.txt'
    with torch.no_grad():
        for all_input_ids, all_attention_mask, all_token_type_ids, all_tokens in loader:
            try:
                all_indices = list()
                labels = list()
                for listatoken in all_tokens:
                    indices = list()
                    for indice in listatoken.list_indices:
                        indices.append(indice)
                    all_indices.append(indices)
                    for label in listatoken.list_labels:
                        labels.append(label)
                
                pred_region_output = model.forward(all_input_ids, all_attention_mask, all_token_type_ids, lista_indices_e1=all_indices)
            except RuntimeError:
                print("all 0 tags, no evaluating this epoch")

            pred_region_labels = []  # for all tokens are not in-entity
            if len(pred_region_output) > 0:
                # pred_region_output (n_regions, n_tags)
                pred_region_labels = torch.argmax(pred_region_output, dim=1).view(-1).to(device)
                # (n_regions)
            with open(save_url, 'w', encoding='utf-8', newline='\n') as save_file:
                for value in pred_region_labels:
                    save_file.write("{}\n".format(value))
  

def main():
    #COLAB
    #model_url = from_project_root("data/model/best_model.pt")
    #model_url = from_project_root("data/model/")
    model_url = from_project_root(r"C:\Users\lisat\OneDrive\jupyter notebook\span-model\model-exp3")
    #LOCAL
    #model_url = from_project_root("./model/")
    print("loading model from", model_url)
    label_lst = get_label(DATA_DIR, LABEL_FILE)
    num_labels = len(label_lst)
    print('num_labels:', num_labels)

    config = BertConfig.from_pretrained(
        model_url,
        num_labels=num_labels
    )
    #model = torch.load(model_url, encoding='latin1')
    #model = torch.load(model_url)
    #with open(model_url, 'rb') as f:
    #  buffer = io.BytesIO(f.read())
    #model = torch.load(buffer)
    #model = AutoModelForTokenClassification.from_pretrained(model_url)
    model = BertForChunkClassification.from_pretrained(model_url, config=config, hidden_size=HIDDEN_SIZE, num_labels=num_labels)
    #model = BertForChunkClassification.from_pretrained(model_url, hidden_size=HIDDEN_SIZE)
    print('--Eval sem NER---')
    #evaluate(model, BERT_MODEL, "test")
    evaluateMatriz(model,BERT_MODEL,"predict")
    #predict(model, predict_url)
    #predict(model, BERT_MODEL, "test")
    pass


if __name__ == '__main__':
    main()
