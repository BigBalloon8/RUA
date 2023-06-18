import torch
import torch.nn as nn
import transformers
from transformers import BertTokenizer, GPT2Tokenizer

encoder_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
decoder_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def encoder_tokenizer(seq_list:list, batch_size=1, num_encoder_sequences=128):
    # takes list of sequnces as strings
    tokeninzed_sequences = []
    for i in range(len(seq_list)//num_encoder_sequences):
        tokeninzed_sequences.append(encoder_tokenizer(seq_list[i*num_encoder_sequences:num_encoder_sequences+i*num_encoder_sequences],
                                             return_tensors="pt", 
                                             padding="max_length", 
                                             truncation=True))
    batch_tokeninzed_sequences = []

    for i in range(len(tokeninzed_sequences)//batch_size):   
        batch_tokeninzed_sequences.append(torch.stack(tokeninzed_sequences[i*batch_size:batch_size+i*batch_size]))

    return batch_tokeninzed_sequences


def decoder_tokenizer(seq_list:list, batch_size=1, num_decoder_sequences=128):
    seq_list = [seq + '[SEP]' for seq in seq_list]
    tokeninzed_sequences = []
    for i in range(len(seq_list)//num_decoder_sequences):
        tokeninzed_sequences.append(decoder_tokenizer(seq_list[i*num_decoder_sequences:num_decoder_sequences+i*num_decoder_sequences],
                                             return_tensors="pt", 
                                             max_length =1024,
                                             padding="max_length", 
                                             truncation=True))
    
    batch_tokeninzed_sequences = []
    for i in range(len(tokeninzed_sequences)//batch_size):
        batch_tokeninzed_sequences.append(torch.stack(tokeninzed_sequences[i*batch_size:batch_size+i*batch_size]))
    return batch_tokeninzed_sequences
    
    