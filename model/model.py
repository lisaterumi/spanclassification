# coding: utf-8
# created by deng on 2019-03-05

import torch
import torch.nn as nn
import numpy as np
from transformers import BertModel, BertPreTrainedModel, BertForTokenClassification

class BertForChunkClassification(BertPreTrainedModel):
    
    def __init__(self, config, hidden_size, num_labels):
        super().__init__(config)
        self.config = config

        #self.dropout = nn.Dropout(p=0.5)
        self.n_tags = num_labels
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(0.1)

        self.bert = BertModel(config=config)

        self.classifier= nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size * 3, num_labels),
        )
        
        # Initialize weights and apply final processing
        #self.post_init()
    
    def forward(self, input_ids, attention_mask, token_type_ids, lista_indices_e1):
        try:
          outputs = self.bert(
              input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
          )  # retorno BERT - sequence_output, pooled_output, (hidden_states), (attentions)
        except Exception as e:
          print('--erro na forward, erro:--', e)
          raise
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        regions = list()
        for hidden, indices in zip(sequence_output, lista_indices_e1): # para cada frase do batch
            # agora, para cada regiao
            for indice_region in indices:
                start = indice_region[0]
                end = indice_region[1] +1
                if start >= 0:
                    regions.append(hidden[start:end]) # tamanho 4(tam regiao em tokens) x 768
                    # TODO - aki, concatenar com tensor ao inves de lista

        region_outputs = []  # all not in-entity tokens, no candidate regions
        if len(regions) > 0:
            cat_regions = [torch.cat([hidden[0], torch.mean(hidden, dim=0), hidden[-1]], dim=-1).view(1, -1) for hidden in regions]
            cat_out = torch.cat(cat_regions, dim=0)
            region_outputs = self.classifier(cat_out) # 495,6
            # shape of region_labels: (n_regions, n_classes)

        return region_outputs

      
def main():
    pass

if __name__ == '__main__':
    main()
