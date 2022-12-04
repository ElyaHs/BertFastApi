import model
from transformers import BertTokenizer
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def tokenize_bert(sentence):
    tokens = tokenizer.tokenize(sentence)
    return tokens


def get_sent1_token_type(sent):
    try:
        return [0] * len(sent)
    except:
        return []


def get_sent2_token_type(sent):
    try:
        return [1] * len(sent)
    except:
        return []


def predict_inference(q1, q2):
    q1 = '[CLS] ' + str(q1) + ' [SEP]'
    q2 = str(q2) + ' [SEP]'

    q1_t = tokenize_bert(q1)
    q2_t = tokenize_bert(q2)

    q1_type = get_sent1_token_type(q1_t)
    q2_type = get_sent2_token_type(q2_t)

    indexes = q1_t + q2_t
    indexes = tokenizer.convert_tokens_to_ids(indexes)

    indexes_type = q1_type + q2_type

    attn_mask = get_sent2_token_type(indexes)

    indexes = torch.LongTensor(indexes).unsqueeze(0)
    indexes_type = torch.LongTensor(indexes_type).unsqueeze(0)
    attn_mask = torch.LongTensor(attn_mask).unsqueeze(0)

    prediction = model.model(indexes, attn_mask, indexes_type)
    prediction = prediction.argmax(dim=-1).item()
    return prediction
