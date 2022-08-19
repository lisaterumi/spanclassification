# coding: utf-8
# created by deng on 2019-01-23

import argparse
from tkinter import HIDDEN
from utils.torch_util import set_random_seed
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import os
import sys
import torch
import json
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datetime import datetime

import utils.json_util as ju
from utils.path_util import from_project_root, exists
from utils.torch_util import get_device
from dataset import InputFeatures, load_and_cache_examples
from utils.utils import get_label, load_tokenizer
from model import BertForChunkClassification
from eval import evaluate, evaluateMatriz
from transformers import AdamW, BertConfig, get_linear_schedule_with_warmup

NUM_EPOCHS = 50
EARLY_STOP = 5
#LR = 0.00005
#LR = 0.00003
LR = 0.0005
#LR = 0.00002
#BATCH_SIZE = 50
BATCH_SIZE = 32
MAX_GRAD_NORM = 5
FREEZE_WV = False
LOG_PER_BATCH = 20
#BERT_MODEL ='pucpr/biobertpt-all'
#BERT_MODEL_NER ='lisaterumi/genia-biobert-ent'
BERT_MODEL ='./model/biobertpt/'
#BERT_MODEL_NER ='./model2/'
#BERT_MODEL ='bert-base-uncased'
#BERT_MODEL ='bert-base-cased'
HIDDEN_SIZE = 768
#MAX_SEQ_LEN = 300
MAX_SEQ_LEN = 256
DATA_DIR='./model/data'
LABEL_FILE='label.txt'
RANDOM_SEED = 233
set_random_seed(RANDOM_SEED)


#TRAIN_URL = from_project_root("data/genia/genia.train")
#DEV_URL = from_project_root("data/genia/genia.dev")
#TEST_URL = from_project_root("data/genia/genia.test")

def train_chunkClassification(n_epochs=NUM_EPOCHS,
                  max_seq_len = MAX_SEQ_LEN,
                  learning_rate=LR,
                  batch_size=BATCH_SIZE,
                  early_stop=EARLY_STOP,
                  clip_norm=MAX_GRAD_NORM,
                  device='auto',
                  save_only_best=True,
                  bert_model=BERT_MODEL
                  ):
    """ 

    Args:
        n_epochs: number of epochs
        learning_rate: learning rate
        batch_size: batch_size
        early_stop: early stop for training
        clip_norm: whether to perform norm clipping, set to 0 if not need
        device: device for torch
        save_only_best: only save model of best performance
    """

    print(datetime.now().strftime("%c\n"))

    # print arguments
    arguments = json.dumps(vars(), indent=2)
    print("arguments", arguments)
    start_time = datetime.now()

    device = get_device(device)

    label_lst = get_label(DATA_DIR, LABEL_FILE)
    num_labels = len(label_lst)

    train_set = InputFeatures(bert_model, max_seq_len, device=device, mode="train", labels=label_lst)
    # retorna list_input_ids, list_attention_mask, list_token_type_ids, list_lista_label_id, list_lista_indices_e1
    
    train_loader = DataLoader(train_set, batch_size=batch_size, drop_last=False, collate_fn=train_set.collate_func)

    print('len train_loader (ou seja, numero de batchs):', len(train_loader))

    '''
    model = BertForChunkClassification(
        n_tags=num_labels,
        hidden_size=HIDDEN_SIZE,
        bert_model = bert_model
    )
    config = BertConfig.from_pretrained(
            bert_model,
            num_labels=num_labels,
            output_hidden_states=True,
            id2label={str(i): label for i, label in enumerate(label_lst)},
            label2id={label: i for i, label in enumerate(label_lst)},
        )
    model = BertForChunkClassification.from_pretrained(bert_model, config=config, hidden_size=HIDDEN_SIZE)
    '''
    config = BertConfig.from_pretrained(
        bert_model,
    )
    model = BertForChunkClassification(
        config=config,
        hidden_size=HIDDEN_SIZE,
        num_labels=num_labels
    )

    # congelando camadas de baixo do BERT, só treinando classificador TODO
    # https://github.com/huggingface/transformers/issues/400
    for param in model.bert.parameters():
        param.requires_grad = False

    if device.type == 'cuda':
        print("using gpu,", torch.cuda.device_count(), "gpu(s) available!\n")
        # model = nn.DataParallel(model)
    else:
        print("using cpu\n")
    model = model.to(device)

    criterion = F.cross_entropy
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    cnt = 0
    max_f1, max_f1_epoch = 0, 0
    best_model_url = None

    # Store the average loss after each epoch so we can plot them.
    loss_values = []
    loss_val_values = []
    n_epochs_exec=0
    # Vamos guardar os valores de perda do treinamento e validação, acurária e tempos de execução.
    training_stats = []

    for epoch in range(n_epochs):
        n_epochs_exec +=1
        # Mede quando tempo a época de treinamento demora
        t0 = datetime.now()
        # Reset the total loss for this epoch.
        total_train_loss = 0
        # switch to train mode
        model.train()
        batch_id = 0
        #for all_input_ids, all_attention_mask, all_token_type_ids, labels, indices, _ in train_loader:
        for all_input_ids, all_attention_mask, all_token_type_ids, all_tokens in train_loader:
            if len(all_tokens) == 0:  # skip no-region cases
                batch_id += 1
                print('Atenção!!!! caiu no if - sem labels - pulando batch')
                continue
            optimizer.zero_grad()

            indices = list()
            labels = list()
            for listatoken in all_tokens:
                for indice in listatoken.list_indices:
                    indices.append(indice)
                for label in listatoken.list_labels:
                    labels.append(label)

            pred_region_labels = model.forward(input_ids=all_input_ids, attention_mask=all_attention_mask, token_type_ids=all_token_type_ids, lista_indices_e1=indices)
            classification_loss = criterion(pred_region_labels, labels)
            loss = classification_loss

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value 
            # from the tensor.
            total_train_loss += loss.item()
            loss.backward()

            # gradient clipping
            #if clip_norm > 0:
            #    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.step()

            endl = '\n' if batch_id % LOG_PER_BATCH == 0 else '\r'
            sys.stdout.write("epoch #%d, batch #%d, loss: %.6f, %s%s" %
                             (epoch, batch_id, loss.item(), datetime.now().strftime("%X"), endl))
            sys.stdout.flush()
            batch_id += 1

        print('\n')

        cnt += 1
        # evaluating model use development dataset or and additional test dataset
        #precision, recall, f1, eval_loss = evaluate(model, bert_model, mode="dev").values()
        ret, eval_loss = evaluate(model, bert_model, mode="dev")
        f1 = ret['f1']
        if f1 > max_f1:
            max_f1, max_f1_epoch = f1, epoch
            name = 'span'
            if save_only_best and best_model_url:
                open(best_model_url, 'w').close() # salva com tamanho 0, pois no colab vai pra lixeira
                os.remove(best_model_url)
            best_model_url = from_project_root(
                "data/model/%s_model_epoch%d_%f.pt" % (name, epoch, f1))
            torch.save(model, best_model_url)
            model.save_pretrained(os.path.join(best_model_url, 'model'))
            cnt = 0

        # Calculate the average loss over the training data.
        avg_train_loss = total_train_loss / len(train_loader)       
        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)
        loss_val_values.append(eval_loss)
        # Mede quanto tempo levou essa época
        training_time = datetime.now() - t0

        print("maximum of f1 value: %.6f, in epoch #%d" % (max_f1, max_f1_epoch))
        print("training time:", str(datetime.now() - start_time).split('.')[0])
        print(datetime.now().strftime("%c\n"))

        # Imprime a acurácia final para a execução da validação.
        avg_val_accuracy = ret['accuracy']
        print("  Acurácia: {0:.2f}".format(avg_val_accuracy))

        # Mede quanto tempo levou a validação
        #validation_time = format_time(time.time() - t0)
        validation_time = datetime.now() - t0

        # Grava as estatísticas para esta época.
        training_stats.append(
            {
                'epoch': epoch + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': eval_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

        if cnt >= early_stop > 0:
            print("---Parando, atingiu o early stop---")
            break

    print('FIM treinamento!!')
    # testando o modelo no dataset de teste
    best_model = torch.load(best_model_url)
    print("best model url:", best_model_url)
    print("evaluating on test dataset: TESTE")
    evaluate(best_model, bert_model, mode="test")
    print("evaluating on test dataset: TESTE, sem ecxluir os O")
    best_model.save_pretrained(from_project_root('data/model/'))
    try:
        cm = evaluateMatriz(best_model, bert_model, mode="test")

        fig = plt.figure(figsize = (4,4))
        ax1 = fig.add_subplot(1,1,1)
        sns.set(font_scale=1.4) #for label size
        sns.heatmap(cm, annot=True, annot_kws={"size": 12}, cbar = False, cmap='Purples', fmt='d')
        ax1.set_ylabel('True Values',fontsize=14)
        ax1.set_xlabel('Predicted Values',fontsize=14)
        plt.savefig('matrix_confusao_teste.png')
        plt.close()
    except:
        print("erro ao gerar matriz de confusão")


    # Mostra números com duas casas decimais
    pd.set_option('precision', 2)

    # Cria um DataFrame das nossas estatísticas de treinamento
    df_stats = pd.DataFrame(data=training_stats)

    # Usa a época como o índice da linha
    df_stats = df_stats.set_index('epoch')

    # Forçar o agrupamento dos cabeçalho da coluna 
    #df = df.style.set_table_styles([dict(selector="th",props=[('max-width', '70px')])])

    # Mostra a tabela
    print(df_stats)
    df_stats.to_csv('df_stats.csv')

    try:
        epochs_range = range(n_epochs_exec)

        print('Plotando grafico de loss')
        #plt.subplot(1, 2, 2)
        figure1 = plt.figure()
        plt.plot(epochs_range, loss_values, label='Training Loss')
        plt.plot(epochs_range, loss_val_values, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.savefig('loss_train_val.png')
        plt.clf()
        
        print('Plotando grafico de loss 222')

        figure2 = plt.figure()
        # Use plot styling from seaborn.
        sns.set(style='darkgrid')

        # Increase the plot size and font size.
        sns.set(font_scale=1.5)
        plt.rcParams["figure.figsize"] = (12,6)

        # Plot the learning curve.
        plt.plot(loss_values, 'b-o')

        # Label the plot.
        plt.title("Training loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig('loss_train.png')
        plt.clf()

        figure3 = plt.figure()
        print('terceiro grafico')
        # Usando estilo
        sns.set(style='darkgrid')

        # Aumentando o tamanho e fonte 
        sns.set(font_scale=1.5)
        plt.rcParams["figure.figsize"] = (12,6)

        # Plotando a curva de aprendizagem
        plt.plot(df_stats['Training Loss'], 'b-o', label="Treinamento")
        plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validação")

        # Adicionando títulos
        plt.title("Perda de treinamento e validação")
        plt.xlabel("Época")
        plt.ylabel("Perda")
        plt.legend()
        plt.xticks([1, 2, 3, 4])

        plt.savefig('loss_train_grafico.png')
    except:
      print('erro ao plotar graficos')



def main(args):
    #format_time(time.time())
    #start_time = time.time()
    start_time = datetime.now()
    train_chunkClassification(n_epochs=args.n_epochs,batch_size=args.batch_size, max_seq_len=args.max_seq_len, bert_model=args.bert_model)
    print("finished in:", datetime.now()- start_time)
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_epochs", default=NUM_EPOCHS, type=str, help="Num epochs")
    parser.add_argument("--batch_size", default=BATCH_SIZE, type=str, help="batch_size")
    parser.add_argument("--max_seq_len", default=MAX_SEQ_LEN, type=str, help="max_seq_len")
    parser.add_argument("--bert_model", default=BERT_MODEL, type=str, help="bert_model")
 
    args = parser.parse_args()

    main(args)

