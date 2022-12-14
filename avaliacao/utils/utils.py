import logging
import os
import random

import numpy as np
import torch
from transformers import BertTokenizer

ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>"]

def get_label(data_dir, label_file):
    print('os.getcwd():', os.getcwd())
    return [label.strip() for label in open(os.path.join(data_dir, label_file), "r", encoding="utf-8")]


def load_tokenizer(model_name_or_path):
    #tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
    print('os.getcwd():', os.getcwd())
    tokenizer = BertTokenizer.from_pretrained(r"C:\Users\lisat\OneDrive\jupyter notebook\spanclassification\model\data\tokenizador")
    tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
    return tokenizer


def write_prediction(data_dir, label_file, output_file, preds):
    """
    For official evaluation script
    :param output_file: prediction_file_path (e.g. eval/proposed_answers.txt)
    :param preds: [0,1,0,2,18,...]
    """
    relation_labels = get_label(data_dir, label_file)
    with open(output_file, "w", encoding="utf-8") as f:
        for idx, pred in enumerate(preds):
            f.write("{}\t{}\n".format(idx, relation_labels[pred]))


def init_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


def set_seed(seed, no_cuda):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if not no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return acc_and_f1(preds, labels)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


