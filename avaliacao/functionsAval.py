from transformers import AutoConfig
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import os
from pathlib import Path
import re
import pickle
from transformers import AutoTokenizer, AutoModelForTokenClassification
import nltk    
from nltk import tokenize 
import torch
from transformers import BertTokenizer,BertForTokenClassification
import numpy as np
import json   
import random
from model import BertForChunkClassification
from transformers import AdamW, BertConfig, get_linear_schedule_with_warmup
import xml.dom.minidom
from torch.utils.data import DataLoader
from importlib import reload 
#from eval import predict
import eval
reload(eval)
from dataset import InputFeatures, load_and_cache_examples
import dataset
reload(dataset)


def save_obj(name, obj):
    existeDir = os.path.exists('../obj')
    if not existeDir:
        os.makedirs('../obj')
    with open('../obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)

def load_obj(name):
    existeDir = os.path.exists('../obj')
    if not existeDir:
        os.makedirs('../obj')
    try:
        with open('../obj/' + name + '.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        print('erro ao pegar obj')
        return None
    
def replaceWhiteSpaces(str):
    return re.sub('\s{2,}',' ',str)
    
def calc_f1(tp, fp, fn, print_result=True):
    """ calculating f1
    Args:
        tp: true positive
        fp: false positive
        fn: false negative
        print_result: whether to print result
    Returns:
        precision, recall, f1
    """
    precision = 0 if tp + fp == 0 else tp / (tp + fp)
    recall = 0 if tp + fn == 0 else tp / (tp + fn)
    f1 = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    if print_result:
        print(" precision = %f, recall = %f, micro_f1 = %f\n" % (precision, recall, f1))
    return precision, recall, f1


def loadSentencesTest():
    # gabarito = dicSentences_new.pkl
    #dicSentences_train = load_obj('dic_sentencesTrain.pkl')
    #dicSentences_train[32]
    # gabarito = dicSentences_new.pkl
    print('Pegando sentencas de teste gabarito: dic_sentencesTest.pkl')
    dicSentences_new_test = load_obj('dic_sentencesTest')
    return dicSentences_new_test

    
def predictBERTNER_IO(sentencas, tipo_entidade):

    model=''
    if tipo_entidade == 'Tratamento':
        model = 'lisaterumi/portuguese-ner-biobertptclin-tratamento'
    elif tipo_entidade == 'Teste':
        model = 'lisaterumi/portuguese-ner-biobertptclin-teste'
    elif tipo_entidade == 'Anatomia':
        model = 'lisaterumi/portuguese-ner-biobertptclin-anatomia'
    elif tipo_entidade == 'Problema':
        model = 'lisaterumi/portuguese-ner-biobertptclin-problema'

    tokenizer = AutoTokenizer.from_pretrained(model)
    config = AutoConfig.from_pretrained(model)
    idx2tag = config.id2label
    #idx2tag = config.label2id 
    print('idx2tag:', idx2tag)
    model = AutoModelForTokenClassification.from_pretrained(model)

    predictedModel=[]
    all_tokens=[]
    
    for test_sentence in sentencas:
        #print('test_sentence:', test_sentence)
        tokenized_sentence = tokenizer.encode(test_sentence)
        input_ids = torch.tensor([tokenized_sentence])#.cuda()
        
        with torch.no_grad():
            output = model(input_ids)
        label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)
        
        # join bpe split tokens
        tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
        new_tokens, new_labels = [], []
        for token, label_idx in zip(tokens, label_indices[0]):
            if token.startswith("##"):
                new_tokens[-1] = new_tokens[-1] + token[2:]
            else:
                new_labels.append(label_idx)
                new_tokens.append(token)
            
        FinalLabelSentence = []
        FinalToken = []
        #print(len(new_tokens))
        #print(len(new_labels))
        for token, label in zip(new_tokens, new_labels):
            label = idx2tag[label]
            #label = str(label)
            if label == "O" or label == "X":
                FinalLabelSentence.append("O")
                FinalToken.append(token)
            else:
                FinalLabelSentence.append(label)
                FinalToken.append(token)
                
        predictedModel.append(FinalLabelSentence[1:-1]) # delete [SEP] and [CLS]
        all_tokens.append(FinalToken[1:-1])
                    
    return predictedModel, all_tokens


def getDicPredictions(tags, tokens):
    dic_predictions = {}
    num=-1
    #[tokens], [indices], tag
    for tags_sentence, tokens_sentence in zip(tags, tokens):
        num=num+1
        entidades = []
        tokens_frase = []
        entidade_anterior=0
        #print('---inicio sentenca:--')
        num_token=0
        indices_entidade = list()
        tokens_entidade = list()
        for tag, token in zip(tags_sentence, tokens_sentence):
            #print(token + ' ' + tag)
            tokens_frase.append([token, num_token])
            if tag!='O' and (entidade_anterior==tag or num==0):
                #entidades.append([num_token, tag])
                indices_entidade.append(num_token)
                #print('token:', token)
                tokens_entidade.append(token)
                entidade_anterior=tag
            #elif tag!='O': # nova entidade
            else:
                if entidade_anterior!='O' and len(tokens_entidade)>0: 
                    entidades.append([tokens_entidade, indices_entidade, entidade_anterior])
                indices_entidade=list()
                tokens_entidade = list()
                if tag!='O': 
                    indices_entidade.append(num_token)
                    tokens_entidade.append(token)
                entidade_anterior=tag
            num_token=num_token+1
        #dic_predictions[num]=entidades
        dic_predictions[num]=[tokens_frase, entidades]
        entidades=list()
    return dic_predictions

def getListaRegionsTruePred(BATCH, dicSentences_new_test, dic_predictions):
    region_true_list, region_pred_list = list(), list() # labels
    region_true_count, region_pred_count = 0, 0 # contagem

    for i in range(BATCH):
        entidades_gabarito = dicSentences_new_test[i][1]
        entidades_preditas = dic_predictions[i][1]
        #print('---entidades_gabarito--:', entidades_gabarito)
        #print('entidades_preditas:', entidades_preditas)
        for entidade_gabarito in entidades_gabarito:
            indices_gabarito = entidade_gabarito[1]
            tag_gabarito = entidade_gabarito[2]
            region_true_count=region_true_count+1
            region_true_list.append(tag_gabarito)
            # ver se previu essa entidade
            previu=0
            for entidade_predita in entidades_preditas:
                indices_predita = entidade_predita[1]
                tag_predita = entidade_predita[2]
                if indices_predita == indices_gabarito:
                    region_pred_list.append(tag_predita)
                    previu=1
                    if tag_predita !='O':
                        region_pred_count=region_pred_count+1
                    break
            if previu==0:
                region_pred_list.append('O')

        # agora o contrario, ver o q previu mas nao era

        for entidade_predita in entidades_preditas:
            indices_predita = entidade_predita[1]
            tag_predita = entidade_predita[2]
            # ver se a entidade prevista existe ou é FP
            existe=0
            for entidade_gabarito in entidades_gabarito:
                indices_gabarito = entidade_gabarito[1]
                if indices_predita == indices_gabarito:
                    existe=1
                    break
            if existe==0:
                region_true_list.append('O')
                region_pred_list.append(tag_predita)

    return region_true_list, region_pred_list


def get_label(data_dir, label_file):
    return [label.strip() for label in open(os.path.join(data_dir, label_file), "r", encoding="utf-8")]


def getModel():
    model_url = r"C:\Users\lisat\OneDrive\jupyter notebook\ChunkClassification\model"
    print("loading model from", model_url)
    label_lst = get_label(r"C:\Users\lisat\OneDrive\jupyter notebook\ChunkClassification\data", "label.txt")
    num_labels = len(label_lst)
    print('num_labels:', num_labels)

    config = BertConfig.from_pretrained(
        model_url,
        num_labels=num_labels
    )
    model = BertForChunkClassification.from_pretrained(model_url, config=config, hidden_size=768, num_labels=num_labels)
    return model

def areConsecutive(arr):
    # Sort the array
    arr.sort()
    n = len(arr)
    # checking the adjacent elements
    for i in range (1,n):
        if(arr[i]!=arr[i-1]+1):
            return False;
             
    return True;   

# 1o. encontrar os postaggers das entidades...
def getDicPosTagger():
    pathCorpus=r'../PreProcessamento/GENIAcorpus3.02.merged_corrigido.xml'
    dicPostagger = {}
    doc = xml.dom.minidom.parse(pathCorpus);
    sentences = doc.getElementsByTagName("sentence")
    for sentence in sentences:
        # ver se tem entidade do interesse
        cons = sentence.getElementsByTagName("cons")
        list_w = sentence.getElementsByTagName("w")
        for i, w in enumerate(list_w):
            texto = w.firstChild.data.lower()
            postagger = w.getAttribute("c")
            if (postagger == '*'):
                postagger = 'NN'
            dicPostagger[texto] = postagger
            for palavra in texto.split('-'):
                if not palavra in dicPostagger.keys():
                    dicPostagger[palavra] = postagger
            for palavra in texto.split('/'):
                if not palavra in dicPostagger.keys():
                    dicPostagger[palavra] = postagger
            if not texto.replace('(','').replace(')','') in dicPostagger.keys():
                dicPostagger[texto] = postagger
    #save_obj('dicPostagger', dicPostagger)
    return dicPostagger

def tipoPostaggerTokens(entidade_token, dicPostagger):
    postagger = ''
    for p in entidade_token:
        if p.lower() in dicPostagger.keys():
            postagger = postagger + '-' + dicPostagger.get(p.lower())
        elif p.lower().replace(',','') in dicPostagger.keys():
            postagger = postagger + '-' + dicPostagger.get(p.lower().replace(',',''))
        elif p.lower().replace('(','').replace(')','') in dicPostagger.keys():
            postagger = postagger + '-' + dicPostagger.get(p.lower().replace('(','').replace(')',''))
        else:
            for palavra in p.lower().split('/'):
                if palavra in dicPostagger.keys():
                    postagger = postagger + '-' + dicPostagger.get(palavra)
            for palavra in p.lower().split('-'):
                if palavra in dicPostagger.keys():
                    postagger = postagger + '-' + dicPostagger.get(palavra)
    return postagger


def getListaPostaggerEntidades(dic_predictions, dicPosTagger):
    lista_postaggers_entidades = []
    for key, value in dic_predictions.items():
        entidades = value[1]
        #print(entidades)
        for entidade in entidades:
            pos_tagger=tipoPostaggerTokens(entidade[0], dicPosTagger)
            #if pos_tagger=='-JJ':
                #print(entidade)
                #print(value[0])
            if pos_tagger not in lista_postaggers_entidades:
                lista_postaggers_entidades.append(pos_tagger)
    return lista_postaggers_entidades

def getCombinacaoEntidadesAll(dic_predictions, dicPosTagger, lista_postaggers_entidades, filtroPostagger=True):
    palavrasDescontinua_underline=['AND_NOT', 'AND/OR', 'AS_WELL_AS', 'AND', 'OR', 'BUT_NOT', 'NEITHER_NOR', 'THAN']
    num=0
    numDescontinuas=0
    erro_corpus=0
    num_frases_sem_entidade=0
    lista_erro_corpus=list()
    combinacaoEntidadesAll = list()
    combinacaoEntidades = list()
    pulando_termos_postagger = list()
    for key, value in dic_predictions.items():
        num=num+1
        combinacaoEntidades = list()
        #print('key:', key)
        #print(value)
        tokens=value[0].copy()
        so_tokens = [t[0] for t in tokens]
        entidades=value[1].copy()
        #print(so_tokens)
        #print(entidades)
        for entidade in entidades:
            erros_entidade = list()
            #print(entidade[1])
            texto_entidade=entidade[0]
            indices = entidade[1]
            tipo_entidade = entidade[2]
            if areConsecutive(indices): # ver se não é descontinua
                #print(entidade[1])
                #print(frase)
                frase = so_tokens.copy()
                inicio=indices[0]
                fim=indices[-1]
                frase.insert(inicio, '<e1>')
                frase.insert(fim+2, '</e1>')
                if ' '.join(texto_entidade).strip()=='-' or ' '.join(texto_entidade).strip()=='=' or ' '.join(texto_entidade).strip()=='+' or ' '.join(texto_entidade).strip()==':' or ' '.join(texto_entidade).strip()==',' or ' '.join(texto_entidade).strip()=="'" or ' '.join(texto_entidade).strip()=='"' or ' '.join(texto_entidade).strip()=='.' or ' '.join(texto_entidade).strip()==';' or ' '.join(texto_entidade).strip()=='/' or ' '.join(texto_entidade).strip()=='(' or ' '.join(texto_entidade).strip()==')' or ' '.join(texto_entidade).strip()=='[' or ' '.join(texto_entidade).strip()==']':
                    pass
                #print('--texto_entidade--:', texto_entidade)
                #print('frase:', frase)
                #print('frase[inicio+1:fim+2]:', frase[inicio+1:fim+2])
                #print(tokens[indice])
                # o corpus tem alguns problemas, ex tem entidade descontinua sem o CC
                # entao aqui, se não bater, não adicionar no arquivo de treinamento
                # ex frase "The inhibition of c - fos and c - jun expression by IL - 4 in LPS - treated cells was shown to be due to a lower transcription rate of the c - fos and c - jun genes ."
                #print("texto_entidade:", texto_entidade)
                #print("frase[inicio:fim]:", frase[inicio+1:fim+2])
                texto_entidade_comparar=' '.join(texto_entidade).strip().replace('/','').replace(')','').replace('(','').replace(']','').replace('[','').replace(',','').replace('.','').replace(';','').replace('-','').replace('+','').replace("'",'')
                texto_entidade_comparar = replaceWhiteSpaces(texto_entidade_comparar)
                texto_frase_comparar = ' '.join(frase[inicio+1:fim+2]).strip().replace('/','').replace(')','').replace('(','').replace(']','').replace('[','').replace(',','').replace('.','').replace(';','').replace('-','').replace('+','').replace("'",'')
                texto_frase_comparar = replaceWhiteSpaces(texto_frase_comparar)
                #tem = [palavra.lower() in texto_entidade_comparar for palavra in palavrasDescontinua]
                #tem = [' '+palavra.lower()+' ' in ' '+texto_entidade_comparar+' ' for palavra in palavrasDescontinua]
                #texto_frase_comparar2 = texto_frase_comparar.replace('and not', 'and_not').replace('as well as', 'as_well_as').replace('neither nor', 'neither_nor').replace('but not', 'but_not')
                #print(frase)
                #print('texto_frase_comparar2.split():', texto_frase_comparar2.split())
                #print('texto_frase_comparar2.split():', texto_frase_comparar2.split()[0])
                #tem = [texto_frase_comparar2.split()[-1]==palavra.lower() for palavra in palavrasDescontinua_underline]
                #tem = [' '+palavra.lower()+' ' in ' '+texto_frase_comparar2+' ' for palavra in palavrasDescontinua_underline]
                #tem_boolean = True in tem
                #print([' '.join(frase).strip(), tipo_entidade])
                #if not tem_boolean:
                #if 1==2:
                #    combinacaoEntidades.append([' '.join(frase).strip(), tipo_entidade]) # apendando entidades reais
                texto_entidade_comparar = texto_entidade_comparar.lower()
                texto_frase_comparar = texto_frase_comparar.lower()
                if ((texto_entidade_comparar == texto_frase_comparar) or (texto_entidade_comparar+'s' == texto_frase_comparar) or (texto_entidade_comparar+'es' == texto_frase_comparar) or (texto_entidade_comparar+'ies' == texto_frase_comparar) or (texto_entidade_comparar[:-1]+'ies' == texto_frase_comparar) or (texto_entidade_comparar[:-1] == texto_frase_comparar.replace(' /','/'))):
                    combinacaoEntidades.append([' '.join(frase).strip(), tipo_entidade]) # apendando entidades reais
                else:
                    #print('tem:', tem)
                    #print('texto_entidade_comparar:', texto_entidade_comparar)
                    #print('texto_frase_comparar:', texto_frase_comparar)
                    #print('texto_frase_comparar2.split()[-1]:', texto_frase_comparar2.split()[-1])
                    #print("' '.join(texto_entidade).strip()):", ' '.join(texto_entidade).strip())
                    #print("' '.join(frase[inicio:fim].strip():", ' '.join(frase[inicio+1:fim+2]).strip())
                    erro_corpus=erro_corpus+1
                    erros_entidade.append(indices)
                    #print('-----indices erro:----', indices)
                    lista_erro_corpus.append([' '.join(frase).strip(), tipo_entidade, ' '.join(so_tokens), entidade])
            else: # agora, qdo são descontinuas
                numDescontinuas=numDescontinuas+1
                #print(entidade[1])
                #frase = so_tokens.copy()
                #inicio=indices[0]
                #fim=indices[-1]
                #frase.insert(inicio, '<e1>')
                #frase.insert(fim+2, '</e1>')

        for entidade in entidades:
                indices = entidade[1]
                #print('indices:', indices)
                if indices in erros_entidade:
                    continue
                inicio=indices[0]
                fim=indices[-1]
                # agora, fazer a combinacao entre eles.. todas a seguir serão do tipo 'O'           
                for indice in indices:
                    for i in range(indice, fim+1):
                        # ver se nao tem antes
                        frase = so_tokens.copy()
                        termo = frase[indice:i+2]
                        frase.insert(indice, '<e1>')
                        #print(i)

                        frase.insert(i+2, '</e1>')
                        frase_string=' '.join(frase).strip()
                        #print(frase_string)
                        #if frase_string not in combinacaoEntidades:
                        # ver se frase não termina com pontuacao ('.',',',';','-',')','(',']','[','/','"',)
                        devePular = 0
                        if '. </e1>' in frase_string or ', </e1>' in frase_string  or '; </e1>' in frase_string or '- </e1>' in frase_string  or ': </e1>' in frase_string  or '= </e1>' in frase_string  or '/ </e1>' in frase_string  or '( </e1>' in frase_string  or ') </e1>' in frase_string  or '[ </e1>' in frase_string  or '] </e1>' in frase_string  or ': </e1>' in frase_string or 'and </e1>' in frase_string or 'or </e1>' in frase_string:
                            devePular=1
                        if '<e1> .' in frase_string or '<e1> ,' in frase_string  or '<e1> ;' in frase_string or '<e1> -' in frase_string  or '<e1> :' in frase_string  or '<e1> =' in frase_string  or '<e1> /' in frase_string  or '<e1> (' in frase_string  or '<e1> )' in frase_string  or '<e1> [' in frase_string  or '<e1> ]' in frase_string  or '<e1> :' in frase_string  or '<e1> and' in frase_string  or '<e1> or' in frase_string:
                            devePular=1
                        if re.search("<e1> [0-9]* </e1>", frase_string):
                            devePular=1
                        if (filtroPostagger):
                            pos_tagger_termo = tipoPostaggerTokens(termo, dicPosTagger)
                            if pos_tagger_termo not in lista_postaggers_entidades:
                                pulando_termos_postagger.append([termo, pos_tagger_termo])
                                devePular=1

                        tem_frase = 0
                        for frase in combinacaoEntidades:
                            if frase[0] == frase_string:
                                tem_frase=''
                                break
                        if tem_frase==0 and devePular==0:
                            #print('inserindo:', frase_string)
                            #print('indice:', indice)
                            combinacaoEntidades.append([frase_string, 'O'])
        # shuffle no combinacaoEntidades
        if len(combinacaoEntidades)>0:
            random.shuffle(combinacaoEntidades)
            combinacaoEntidadesAll.append([' '.join(so_tokens).strip(), combinacaoEntidades])
        else:
            num_frases_sem_entidade = num_frases_sem_entidade+1
            combinacaoEntidadesAll.append([])
            #print("key sem entidade", key)
            #print('len(combinacaoEntidades):', len(combinacaoEntidades))
            #print(so_tokens)
        combinacaoEntidades = list()
        if (num % 1000) ==0:
        #if (num %3) == 0:
            print('key:', key)
            #break

    print('numDescontinuas:', numDescontinuas)
    print('erro_corpus:', erro_corpus)
    print('num_frases_sem_entidade:', num_frases_sem_entidade)
    print('len(combinacaoEntidadesAll:)',len(combinacaoEntidadesAll))
    
    return combinacaoEntidadesAll

def gravaArquivoPredict(combinacaoEntidadesAll, tipo='io'):
    numTotalEntidades=0
    file=r'data/genia.predict'
    f = open(file, 'w')
    numFrases=0
    for i, combinacaoEntidades in enumerate(combinacaoEntidadesAll):
        #print(dicSentences[i])
        if combinacaoEntidades:
            numFrases=numFrases+1
            frase = combinacaoEntidades[0]
            frases_entidade = combinacaoEntidades[1]
            f.write(frase+'\n')
            for frase_entidade in frases_entidade:
                f.write(frase_entidade[1]+'\t'+frase_entidade[0]+'\n')
                numTotalEntidades=numTotalEntidades+1

    f.close()

    print('numTotalEntidades:', numTotalEntidades)
    print('numFrases:', numFrases)
    print('gravando em:', file)
    
    
def predictSpan(model):
    """ predict NER result for sentence list
    Args:
        model: trained end2end model
        data_url: url to test dataset for evaluating
        bsl_model: trained binary sequence labeling model
        batch_size: batch_size when predicting
    """
    #print(label_lst)
    label_lst = get_label(r"C:\Users\lisat\OneDrive\jupyter notebook\ChunkClassification\data", "label.txt")

    device='cpu'
    test_set = InputFeatures(model, 200, device='cpu', mode="predict", labels=label_lst)
    #print('len(test_set):', len(test_set))
    loader = DataLoader(test_set, batch_size=200, collate_fn=test_set.collate_func)
    #print('len(loader):', len(loader))

    # data = input_ids, attention_mask, token_type_ids, lista_e1_mask
    model.eval()
    save_url = 'predictions.txt'
    numB=0
    with torch.no_grad():
        for all_input_ids, all_attention_mask, all_token_type_ids, _, indices, _ in loader:
            numB=numB+1
            #print('vai fazer predicao, numB:', numB)
            #print('len(all_input_ids):', len(all_input_ids))
            #print('len(indices):', len(indices))
            try:
                pred_region_output = model.forward(all_input_ids, all_attention_mask, all_token_type_ids, lista_indices_e1=indices)
            except RuntimeError:
                print("all 0 tags, no evaluating this epoch")

            #print('len(pred_region_output:)', len(pred_region_output))
            #try:
            #    print('pred_region_output.size()', pred_region_output.size())
            #except:
            #    print('erro ao imprimir shape do pred_region_output')
            pred_region_labels = []  # for all tokens are not in-entity
            if len(pred_region_output) > 0:
                # pred_region_output (n_regions, n_tags)
                pred_region_labels = torch.argmax(pred_region_output, dim=1).view(-1).to(device)
                # (n_regions)
            #print('len(pred_region_labels:)', len(pred_region_labels))
            
            with open(save_url, 'w', encoding='utf-8', newline='\n') as save_file:
                for value in pred_region_labels:
                    save_file.write("{}\n".format(value))
    #print('numB:', numB)
    return pred_region_labels


def gather_duplicate_indices(a):
    _,tags,count = np.unique(a, axis=0, return_inverse=True, return_counts=True)
    sidx = tags.argsort()
    return np.split(sidx, count.cumsum())[:-1]

def getCombinacaoEntidadesAll_pred(combinacaoEntidadesAll, pred_region_labels):
    # TODO - mesclar predicoes com combinacaoEntidadesAll
    #print(pred_region_labels[:15])
    labels = ['O', 'DNA', 'RNA', 'PROTEIN', 'CELL_LINE', 'CELL_TYPE']
    combinacaoEntidadesAll_pred = list()
    num=0
    news = list()
    for combinacao in combinacaoEntidadesAll:
        #print(combinacao)
        if combinacao:
            for ent in combinacao[1]:
                tag = int(pred_region_labels[num])
                num=num+1
                #new = [combinacao[1][0][0], combinacao[1][0][1]]
                if labels[tag] !=ent[0]:
                    pass
                new = [ent[0], labels[tag]]
                news.append(new)
                tag=0
            #combinacao = new
            combinacaoEntidadesAll_pred.append([combinacao[0], news])
            news=list()
        else:
            combinacaoEntidadesAll_pred.append([])

    print('num:', num)    
    return combinacaoEntidadesAll_pred

def getDicPredictionsAll(combinacaoEntidadesAll_pred, dic_predictions):
    num=-1
    dic_predictions_all={}
    entidades=list()
    numAcrescentou=0
    for valor in combinacaoEntidadesAll_pred:
        num=num+1
        if valor:
            #print('valor[1]:', valor[1])
            frase = valor[0]
            frase = frase.split()
            frase_prevista = dic_predictions[num].copy()
            #print('frase_prevista:', frase_prevista[0])
            entidades_ja_previstas = dic_predictions[num].copy()[1]
            #print('entidades_ja_previstas:', entidades_ja_previstas)
            entidades=list()
            for entidade_frase in valor[1]:
                tag = entidade_frase[1]
                palavras=entidade_frase[0]
                if tag!='O':
                    #print('tag:', tag)
                    #print('palavras:', palavras)
                    palavras = palavras.split()
                    indice_inicio = palavras.index('<e1>')+1
                    indice_fim =  palavras.index('</e1>')-1
                    #print('indice_inicio:', indice_inicio)
                    #print('indice_fim:', indice_fim)
                    indices = [i for i in range(indice_inicio-1, indice_fim)]
                    tokens = frase[indice_inicio-1:indice_fim]
                    #print([tokens, indices, tag+'#######'])
                    entidades.append([tokens, indices, tag])
                    #entidades.append([tokens, indices, tag+'#######'])

            if len(entidades)>0:
                tem=0
                for entidade_nova in entidades:
                    indices_nova = ''.join(str(entidade_nova[1]))
                    for entidade_prevista in entidades_ja_previstas:
                        indices_prevista = ''.join(str(entidade_prevista[1]))
                        if indices_prevista==indices_nova:
                            tem=1
                            break
                    if tem==0:
                        numAcrescentou=numAcrescentou+1
                        entidades_ja_previstas.append(entidade_nova) # TODO
            dic_predictions_all[num]=[frase_prevista[0],entidades_ja_previstas]
            entidades=list()
        else:
            dic_predictions_all[num]=[dic_predictions[num][0], []]
            entidades=list()
        #if num>2:
        #    break
    print('numAcrescentou:', numAcrescentou)
    return dic_predictions_all


def AvalFinal(dicSentences_new_test, dic_predictions_all, BATCH):

    region_true_list, region_pred_list = list(), list() # labels
    region_true_count, region_pred_count = 0, 0 # contagem
    numErro1=0
    numErro2=0

    for i in range(0, BATCH, 1):
        #print('\n---Label vs predicao--------')
        #print('frase:', dicSentences_new_test[i][0])
        #print('i:', i)
        #print(dicSentences_new_test[i][1])
        #print(dic_predictions_all[i][1])


        entidades_gabarito = dicSentences_new_test[i][1]
        try:
            entidades_preditas = dic_predictions_all[i][1]
        except:
            print(i)
        #print('---entidades_gabarito--:', entidades_gabarito)
        #print('entidades_preditas:', entidades_preditas)
        for entidade_gabarito in entidades_gabarito:
            indices_gabarito = entidade_gabarito[1]
            tag_gabarito = entidade_gabarito[2]
            region_true_count=region_true_count+1
            region_true_list.append(tag_gabarito)
            # ver se previu essa entidade
            previu=0
            for entidade_predita in entidades_preditas:
                indices_predita = entidade_predita[1]
                tag_predita = entidade_predita[2]
                if indices_predita == indices_gabarito:
                    region_pred_list.append(tag_predita)
                    previu=1
                    if tag_predita !='O':
                        region_pred_count=region_pred_count+1
                    break
            if previu==0:
                numErro1=numErro1+1
                region_pred_list.append('O')

        # agora o contrario, ver o q previu mas nao era

        for entidade_predita in entidades_preditas:
            indices_predita = entidade_predita[1]
            tag_predita = entidade_predita[2]
            # ver se a entidade prevista existe ou é FP
            existe=0
            for entidade_gabarito in entidades_gabarito:
                indices_gabarito = entidade_gabarito[1]
                if indices_predita == indices_gabarito:
                    existe=1
                    break
            if existe==0:
                numErro2 = numErro2+1
                region_true_list.append('O')
                region_pred_list.append(tag_predita)

    #print('===resultados====')
    #print('region_true_list:', region_true_list)
    #print('region_pred_list:', region_pred_list)

    print('numErro1:', numErro1)
    print('numErro2:', numErro2)
    #print(classification_report(region_true_list, region_pred_list, labels=['O', 'PROTEIN', 'DNA', 'RNA', 'CELL_TYPE', 'CELL_LINE'], target_names=['O', 'PROTEIN', 'DNA', 'RNA', 'CELL_TYPE', 'CELL_LINE'], digits=6))
    print(classification_report(region_true_list, region_pred_list, digits=6))

    return region_true_list, region_pred_list


def getDicPredictionsSpan(combinacaoEntidadesAll_pred, dic_predictions):
    num=-1
    dic_predictions_span={}
    entidades=list()
    numAcrescentou=0
    for valor in combinacaoEntidadesAll_pred:
        num=num+1
        if valor:
            #print('valor[1]:', valor[1])
            frase = valor[0]
            frase = frase.split()
            frase_prevista = dic_predictions[num].copy()
            #print('frase_prevista:', frase_prevista[0])
            #print('entidades_ja_previstas:', entidades_ja_previstas)
            entidades=list()
            for entidade_frase in valor[1]:
                tag = entidade_frase[1]
                palavras=entidade_frase[0]
                if tag!='O':
                    #print('tag:', tag)
                    #print('palavras:', palavras)
                    palavras = palavras.split()
                    indice_inicio = palavras.index('<e1>')+1
                    indice_fim =  palavras.index('</e1>')-1
                    #print('indice_inicio:', indice_inicio)
                    #print('indice_fim:', indice_fim)
                    indices = [i for i in range(indice_inicio-1, indice_fim)]
                    tokens = frase[indice_inicio-1:indice_fim]
                    #print([tokens, indices, tag+'#######'])
                    entidades.append([tokens, indices, tag])
                    #entidades.append([tokens, indices, tag+'#######'])

            if len(entidades)>0:
                dic_predictions_span[num]=[frase_prevista[0],entidades]
            entidades=list()
        else:
            dic_predictions_span[num]=[dic_predictions[num][0], []]
            entidades=list()
        #if num>2:
        #    break
    return dic_predictions_span


def EntidadeUmaLetra(entidade):
    texto = entidade
    texto = ' '.join(texto).strip()
    #print(len(texto))
    if len(texto)>1:
        retorno = 0
    else:
        retorno = 1
    return retorno

def getCombinacaoEntidadesAllPosProc(dic_predictions, dicPosTagger, lista_postaggers_entidades, filtroPostagger=True):
    #  filtro (pos-processamento)
    palavrasDescontinua_underline=['AND_NOT', 'AND/OR', 'AS_WELL_AS', 'AND', 'OR', 'BUT_NOT', 'NEITHER_NOR', 'THAN']
    num=0
    numDescontinuas=0
    erro_corpus=0
    num_frases_sem_entidade=0
    lista_erro_corpus=list()
    combinacaoEntidadesAll = list()
    combinacaoEntidades = list()
    pulando_termos_postagger = list()
    numEntidadesRetiradas=0
    for key, value in dic_predictions.items():
        num=num+1
        combinacaoEntidades = list()
        #print('key:', key)
        #print(value)
        tokens=value[0].copy()
        so_tokens = [t[0] for t in tokens]
        entidades=value[1].copy()
        #print(so_tokens)
        #print(entidades)
        for entidade in entidades:
            erros_entidade = list()
            #print(entidade[1])
            texto_entidade=entidade[0]
            indices = entidade[1]
            tipo_entidade = entidade[2]
            if areConsecutive(indices): # ver se não é descontinua
                #print(entidade[1])
                #print(frase)
                frase = so_tokens.copy()
                inicio=indices[0]
                fim=indices[-1]
                frase.insert(inicio, '<e1>')
                frase.insert(fim+2, '</e1>')
                if ' '.join(texto_entidade).strip()=='-' or ' '.join(texto_entidade).strip()=='=' or ' '.join(texto_entidade).strip()=='+' or ' '.join(texto_entidade).strip()==':' or ' '.join(texto_entidade).strip()==',' or ' '.join(texto_entidade).strip()=="'" or ' '.join(texto_entidade).strip()=='"' or ' '.join(texto_entidade).strip()=='.' or ' '.join(texto_entidade).strip()==';' or ' '.join(texto_entidade).strip()=='/' or ' '.join(texto_entidade).strip()=='(' or ' '.join(texto_entidade).strip()==')' or ' '.join(texto_entidade).strip()=='[' or ' '.join(texto_entidade).strip()==']'or EntidadeUmaLetra(texto_entidade):
                    numEntidadesRetiradas=numEntidadesRetiradas+1
                    pass
                #print('--texto_entidade--:', texto_entidade)
                #print('frase:', frase)
                #print('frase[inicio+1:fim+2]:', frase[inicio+1:fim+2])
                #print(tokens[indice])
                # o corpus tem alguns problemas, ex tem entidade descontinua sem o CC
                # entao aqui, se não bater, não adicionar no arquivo de treinamento
                # ex frase "The inhibition of c - fos and c - jun expression by IL - 4 in LPS - treated cells was shown to be due to a lower transcription rate of the c - fos and c - jun genes ."
                #print("texto_entidade:", texto_entidade)
                #print("frase[inicio:fim]:", frase[inicio+1:fim+2])
                texto_entidade_comparar=' '.join(texto_entidade).strip().replace('/','').replace(')','').replace('(','').replace(']','').replace('[','').replace(',','').replace('.','').replace(';','').replace('-','').replace('+','').replace("'",'')
                texto_entidade_comparar = replaceWhiteSpaces(texto_entidade_comparar)
                texto_frase_comparar = ' '.join(frase[inicio+1:fim+2]).strip().replace('/','').replace(')','').replace('(','').replace(']','').replace('[','').replace(',','').replace('.','').replace(';','').replace('-','').replace('+','').replace("'",'')
                texto_frase_comparar = replaceWhiteSpaces(texto_frase_comparar)
                #tem = [palavra.lower() in texto_entidade_comparar for palavra in palavrasDescontinua]
                #tem = [' '+palavra.lower()+' ' in ' '+texto_entidade_comparar+' ' for palavra in palavrasDescontinua]
                #texto_frase_comparar2 = texto_frase_comparar.replace('and not', 'and_not').replace('as well as', 'as_well_as').replace('neither nor', 'neither_nor').replace('but not', 'but_not')
                #print(frase)
                #print('texto_frase_comparar2.split():', texto_frase_comparar2.split())
                #print('texto_frase_comparar2.split():', texto_frase_comparar2.split()[0])
                #tem = [texto_frase_comparar2.split()[-1]==palavra.lower() for palavra in palavrasDescontinua_underline]
                #tem = [' '+palavra.lower()+' ' in ' '+texto_frase_comparar2+' ' for palavra in palavrasDescontinua_underline]
                #tem_boolean = True in tem
                #print([' '.join(frase).strip(), tipo_entidade])
                #if not tem_boolean:
                #if 1==2:
                #    combinacaoEntidades.append([' '.join(frase).strip(), tipo_entidade]) # apendando entidades reais
                texto_entidade_comparar = texto_entidade_comparar.lower()
                texto_frase_comparar = texto_frase_comparar.lower()
                if ((texto_entidade_comparar == texto_frase_comparar) or (texto_entidade_comparar+'s' == texto_frase_comparar) or (texto_entidade_comparar+'es' == texto_frase_comparar) or (texto_entidade_comparar+'ies' == texto_frase_comparar) or (texto_entidade_comparar[:-1]+'ies' == texto_frase_comparar) or (texto_entidade_comparar[:-1] == texto_frase_comparar.replace(' /','/'))):
                    combinacaoEntidades.append([' '.join(frase).strip(), tipo_entidade]) # apendando entidades reais
                else:
                    #print('tem:', tem)
                    #print('texto_entidade_comparar:', texto_entidade_comparar)
                    #print('texto_frase_comparar:', texto_frase_comparar)
                    #print('texto_frase_comparar2.split()[-1]:', texto_frase_comparar2.split()[-1])
                    #print("' '.join(texto_entidade).strip()):", ' '.join(texto_entidade).strip())
                    #print("' '.join(frase[inicio:fim].strip():", ' '.join(frase[inicio+1:fim+2]).strip())
                    erro_corpus=erro_corpus+1
                    erros_entidade.append(indices)
                    #print('-----indices erro:----', indices)
                    lista_erro_corpus.append([' '.join(frase).strip(), tipo_entidade, ' '.join(so_tokens), entidade])
            else: # agora, qdo são descontinuas
                numDescontinuas=numDescontinuas+1
                #print(entidade[1])
                #frase = so_tokens.copy()
                #inicio=indices[0]
                #fim=indices[-1]
                #frase.insert(inicio, '<e1>')
                #frase.insert(fim+2, '</e1>')

        for entidade in entidades:
                indices = entidade[1]
                #print('indices:', indices)
                if indices in erros_entidade:
                    continue
                inicio=indices[0]
                fim=indices[-1]
                # agora, fazer a combinacao entre eles.. todas a seguir serão do tipo 'O'           
                for indice in indices:
                    for i in range(indice, fim+1):
                        # ver se nao tem antes
                        frase = so_tokens.copy()
                        termo = frase[indice:i+2]
                        frase.insert(indice, '<e1>')
                        #print(i)

                        frase.insert(i+2, '</e1>')
                        frase_string=' '.join(frase).strip()
                        #print(frase_string)
                        #if frase_string not in combinacaoEntidades:
                        # ver se frase não termina com pontuacao ('.',',',';','-',')','(',']','[','/','"',)
                        devePular = 0
                        if '. </e1>' in frase_string or ', </e1>' in frase_string  or '; </e1>' in frase_string or '- </e1>' in frase_string  or ': </e1>' in frase_string  or '= </e1>' in frase_string  or '/ </e1>' in frase_string  or '( </e1>' in frase_string  or ') </e1>' in frase_string  or '[ </e1>' in frase_string  or '] </e1>' in frase_string  or ': </e1>' in frase_string or 'and </e1>' in frase_string or 'or </e1>' in frase_string:
                            devePular=1
                        if '<e1> .' in frase_string or '<e1> ,' in frase_string  or '<e1> ;' in frase_string or '<e1> -' in frase_string  or '<e1> :' in frase_string  or '<e1> =' in frase_string  or '<e1> /' in frase_string  or '<e1> (' in frase_string  or '<e1> )' in frase_string  or '<e1> [' in frase_string  or '<e1> ]' in frase_string  or '<e1> :' in frase_string  or '<e1> and' in frase_string  or '<e1> or' in frase_string:
                            devePular=1
                        if re.search("<e1> [0-9]* </e1>", frase_string):
                            devePular=1
                        if re.search("<e1> . </e1>", frase_string):# só uma letra ou numero
                            devePular=1
                        if '</e1> factor' in frase_string or '</e1> receptor' in frase_string or '</e1> site' in frase_string or '</e1> cell line' in frase_string or '</e1> region' in frase_string or '</e1> cell' in frase_string or '</e1> enhancer' in frase_string or '</e1> element' in frase_string or '</e1> protein' in frase_string or '</e1> gene' in frase_string:
                            devePular=1
                            #print('aaaaaaaaa:', frase_string)
                        if (filtroPostagger):
                            pos_tagger_termo = tipoPostaggerTokens(termo, dicPosTagger)
                            if pos_tagger_termo not in lista_postaggers_entidades:
                                pulando_termos_postagger.append([termo, pos_tagger_termo])
                                devePular=1

                        tem_frase = 0
                        for frase in combinacaoEntidades:
                            if frase[0] == frase_string:
                                tem_frase=''
                                break
                        if devePular==1:
                            numEntidadesRetiradas=numEntidadesRetiradas+1
                        if tem_frase==0 and devePular==0:
                            #print('inserindo:', frase_string)
                            #print('indice:', indice)
                            combinacaoEntidades.append([frase_string, 'O'])
        # shuffle no combinacaoEntidades
        if len(combinacaoEntidades)>0:
            random.shuffle(combinacaoEntidades)
            combinacaoEntidadesAll.append([' '.join(so_tokens).strip(), combinacaoEntidades])
        else:
            num_frases_sem_entidade = num_frases_sem_entidade+1
            combinacaoEntidadesAll.append([])
            #print("key sem entidade", key)
            #print('len(combinacaoEntidades):', len(combinacaoEntidades))
            #print(so_tokens)
        combinacaoEntidades = list()
        if (num % 1000) ==0:
        #if (num %3) == 0:
            print('key:', key)
            #break
    print('numDescontinuas:', numDescontinuas)
    print('erro_corpus:', erro_corpus)
    print('num_frases_sem_entidade:', num_frases_sem_entidade)
    print('numEntidadesRetiradas:', numEntidadesRetiradas)
    return combinacaoEntidadesAll
    