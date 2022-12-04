import torch
import torch.nn as nn
from transformers import BertModel
import transformers

class BERTNLIModel(nn.Module):
    def __init__(self,
                 bert_model,
                 output_dim,
                 ):
        super().__init__()
        self.bert = bert_model

        embedding_dim = bert_model.config.to_dict()['hidden_size']
        self.out = nn.Linear(embedding_dim, output_dim)
    def forward(self, sequence, attn_mask, token_type):
        embedded = self.bert(input_ids = sequence, attention_mask = attn_mask, token_type_ids= token_type)[1]
        output = self.out(embedded)
        return output

OUTPUT_DIM = 2
bert_model = BertModel.from_pretrained('bert-base-uncased')

model = BERTNLIModel(bert_model,OUTPUT_DIM)

model.load_state_dict(torch.load("bert_based_quora_model.pt ", map_location=torch.device('cpu')))
model.eval()


