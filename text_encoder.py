from transformers import Trainer, TrainingArguments,BertTokenizer, BertModel, BertPreTrainedModel, BertConfig
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

class BERT_Encoder(nn.Module):
    def __init__(self, hidden):
        super(BERT_Encoder, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)
        self.dropout = nn.Dropout(0.35)
        self.mlp = nn.Linear(768, hidden)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
            return_dict=True,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=return_dict,
        )
        pooled_output = outputs['pooler_output']
        pooled_output = self.dropout(pooled_output)
        logits = self.mlp(pooled_output)
        return logits
    
    
    def get_attention(self,
                    input_ids=None,
                    attention_mask=None,
                    token_type_ids=None):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        attentions = outputs.attentions
        return_attention = None
        for attention in attentions[-1]:
            normalized_attention = torch.softmax(attention, dim=2).squeeze().detach().numpy()
            return_attention = normalized_attention

        return return_attention