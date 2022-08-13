# coding: utf-8
# created by deng on 2019-03-05

import torch
import torch.nn as nn
import numpy as np
from transformers import BertModel, BertPreTrainedModel, BertForTokenClassification


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score

class BertForChunkClassification(BertPreTrainedModel):
    
    def __init__(self, config, hidden_size, num_labels):
        super().__init__(config)
        self.config = config

        #self.dropout = nn.Dropout(p=0.5)
        self.n_tags = num_labels
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(0.1)

        self.bert = BertModel(config=config)
        self.cls = BertOnlyNSPHead(config)
        
        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(self, input_ids, attention_mask, token_type_ids, lista_indices_e1):
        try:
          outputs = self.bert(
              input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
          )  # retorno BERT - sequence_output, pooled_output, (hidden_states), (attentions)
        except Exception as e:
          print('--erro na forward, erro:--', e)
          raise
        sequence_output = outputs[0]
        #sequence_output = self.dropout(sequence_output)

        regions = list()
        for hidden, indices in zip(sequence_output, lista_indices_e1): # para cada frase do batch
            # agora, para cada regiao
            for indice_region in indices:
                start = indice_region[0]
                end = indice_region[1] +1
                if start >= 0:
                    regions.append(hidden[start:end]) # tamanho 4(tam regiao em tokens) x 768

        region_outputs = []  # all not in-entity tokens, no candidate regions
        if len(regions) > 0:
            cat_regions = [torch.cat([hidden[0], torch.mean(hidden, dim=0), hidden[-1]], dim=-1).view(1, -1) for hidden in regions]
            cat_out = torch.cat(cat_regions, dim=0)
            #region_outputs = self.classifier(cat_out) # 495,6
            # shape of region_labels: (n_regions, n_classes)
            sequence_output_region = cat_out

        sequence_output_all = sequence_output + sequence_output_region # TODO - cat?
        pooled_output = sequence_output_all

        seq_relationship_scores = self.cls(pooled_output)


        return seq_relationship_scores

      
def main():
    pass

if __name__ == '__main__':
    main()
