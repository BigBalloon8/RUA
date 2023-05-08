import torch
import torch.nn as nn
import transformers
from transformers import BertTokenizer, GPT2Tokenizer

encoder_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
decoder_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def encoder_tokenizer(seq_list:list, batch_size=1, num_encoder_seqences=128):
    # takes list of sequnces as strings
    tokeninzed_sequnces = []
    for i in range(len(seq_list)//num_encoder_seqences):
        tokeninzed_sequnces.append(encoder_tokenizer(seq_list[i*num_encoder_seqences:num_encoder_seqences+i*num_encoder_seqences],
                                             return_tensors="pt", 
                                             padding="max_length", 
                                             truncation=True))
    batch_tokeninzed_sequnces = []

    for i in range(len(tokeninzed_sequnces)//batch_size):   
        batch_tokeninzed_sequnces.append(torch.stack(tokeninzed_sequnces[i*batch_size:batch_size+i*batch_size]))

    return batch_tokeninzed_sequnces


def decoder_tokenizer(seq_list:list, batch_size=1, num_decoder_seqences=128):
    seq_list = [seq + '[SEP]' for seq in seq_list]
    tokeninzed_sequnces = []
    for i in range(len(seq_list)//num_decoder_seqences):
        tokeninzed_sequnces.append(decoder_tokenizer(seq_list[i*num_decoder_seqences:num_decoder_seqences+i*num_decoder_seqences],
                                             return_tensors="pt", 
                                             max_length =1024,
                                             padding="max_length", 
                                             truncation=True))
    
    batch_tokeninzed_sequnces = []
    for i in range(len(tokeninzed_sequnces)//batch_size):
        batch_tokeninzed_sequnces.append(torch.stack(tokeninzed_sequnces[i*batch_size:batch_size+i*batch_size]))
    return batch_tokeninzed_sequnces
    
    