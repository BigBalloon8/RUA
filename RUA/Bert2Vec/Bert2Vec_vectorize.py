from transformers import BertTokenizer, BertModel
import transformers
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import os
import json
import random
import os

transformers.logging.set_verbosity_error()

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_base = BertModel.from_pretrained("bert-base-uncased")
device = torch.device("cuda")
host = torch.device("cpu")



class SEDataset(Dataset):
    def __init__(self, file_name, batch_size):
        with open(f"/home/bigballoon/datasets/stack_exchange/{file_name}", "r") as file:
            str_data = file.readlines()
        self.data = []
        for i in str_data:
            self.data.append(json.loads(i))
        self.batch_size = batch_size
           
    
    def tokenize(self, idx):
        #Sequence
        seq = tokenizer([i["texts"] for i in self.data[idx*self.batch_size:idx*self.batch_size+self.batch_size]], 
                        return_tensors="pt", 
                        padding="max_length", 
                        truncation=True)
        return seq
        
    def __len__(self):
        return len(self.data)//self.batch_size
    
    def __getitem__(self, index):
        return self.tokenize(index) #working with a low memory system so cant pre tokenize data


def SEDataloader(batch_size=1):
    files = [i for i in os.listdir("/home/bigballoon/datasets/stack_exchange") if ".json" in i]
    for f_name in files:
        data = SEDataset(f_name, batch_size)
        for i in range(len(data)):
            if batch_size == 1:
                yield data[i]
            else:
                yield data[i]


class Bert2Vec(nn.Module):
    def __init__(self):
        super(Bert2Vec,self).__init__()
        self.bert = bert_base
    
    def forward(self, x):
        x = self.bert(**x)["last_hidden_state"]
        x = torch.mean(x, dim=1)
        return x


def vector_calc(model, data_loader):
    save = 0
    output_vectors = []
    for i, data in enumerate(data_loader):
        data = data.to(device)
        with torch.no_grad():
            vec = model(data).to(host)
            output_vectors.append(vec)
        if i % 10000 == 0:
            print(i)
        if i% 25000 ==0 and i:
            torch.save(output_vectors, "/home/bigballoon/datasets/stack_exchange/vectors{save}.h5")
            save += 1
            output_vectors = []

def main():
    loader = SEDataloader(batch_size=1)
    model = Bert2Vec()
    model.to(device)
    vecs = vector_calc(model, loader)




if __name__ == "__main__":
    main()

