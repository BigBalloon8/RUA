from transformers import BertModel
import transformers
from peft import get_peft_model, LoraConfig, TaskType
import torch
import torch.nn as nn
import math

#bert_base = BertModel.from_pretrained("bert-base-uncased", add_pooling_layer = False)

class bert_encoder_module(nn.Module):
    def __init__(self, decoder_sequence_length=1024):
        """
        The same bert and linear layers act on all 100 short-term sequences
        """
        super().__init__()
        self.decoder_sequence_factor = decoder_sequence_length/512
        if self.decoder_sequence_factor % 1.0 == 0.0:
            raise ValueError("decoder_sequence_factor must be a multiple of 512(bert sequence length)")
        self.bert_base = BertModel.from_pretrained("bert-base-uncased", add_pooling_layer = False)
        peft_config = LoraConfig(task_type=TaskType.QUESTION_ANS, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
        self.bert_base = get_peft_model(self.bert_base, peft_config)
        self.linear_in = nn.Linear(768, 2048)
        self.linear_out = nn.Linear(2048, int(768*self.decoder_sequence_factor) + 768*2)


    def forward(self, input_ids, attention_mask, token_type_ids):
        if input_ids.dim() == 2:
            input_ids = input_ids[None, :]
            attention_mask = attention_mask[None, :]
            token_type_ids = token_type_ids[None, :]

        x = torch.vmap(lambda a, b, c, : self.bert_base(input_ids=a, attention_mask=b, token_type_ids=c)["last_hidden_state"], in_dims=1, out_dims=1)(input_ids, attention_mask, token_type_ids)
        x = torch.vmap(lambda a: torch.vmap(lambda input: self.linear_out(self.linear_in(input)),in_dims=1, out_dims=1)(a))(x)
        print(x.shape)
        chunks = torch.chunk(x, int(self.decoder_sequence_factor), dim=3)
        x = torch.cat(chunks, dim=2)
        return x

class decoder_embedder(nn.Module):
    """
    A Lazy Mans Embedder will deal with later
    """
    def __init___(self):
        super().__init__()
        config = transformers.BertConfig(vocab_size = 30522, hidden_size = 768 ,intermediate_size = 3072,max_position_embeddings = 1024, type_vocab_size = 2)
        bert = BertModel(config)
        self.embed = bert.embeddings

    def forward(self, input_ids, token_type_ids):
        return self.embed(input_ids=input_ids, token_type_ids=token_type_ids)
        

class initial_decoder_module(nn.Module):
    def __init__(self):
        super().__init__()
        self.QKV_in = nn.Linear(768, 2048)
        self.QKV_out = nn.Linear(2048, 768*3)
        self.MHA = nn.MultiheadAttention(embed_dim=768, num_heads=8)
        self.SHA = nn.MultiheadAttention(embed_dim=768, num_heads=1, batch_first=True) #Single HA
        self.LN1 = nn.LayerNorm(768)
        self.LN2 = nn.LayerNorm(768)
        self.LN3 = nn.LayerNorm(768)
        self.ffn_in = nn.Linear(768, 2048)
        self.ffn_out = nn.Linear(2048, 768)


    def forward(self, x, K, V, a_mask):
        #a_mask = x["attention_mask"]
        QKV = torch.vmap(lambda input: self.QKV_out(self.QKV_in(input)),in_dims=1, out_dims=1)(x)
        #QKV = torch.permute(QKV, (1,0,2))
        xQ, xK, xV = torch.tensor_split(QKV, 3 , dim=2)
        SHA_out = self.MHA(xQ,xK,xV,need_weights=False)[0]
        x = SHA_out + x
        Q = self.LN1(x)
        #print(Q.shape,K.shape,V.shape)
        x = torch.vmap(lambda k,v : self.SHA(Q,k,v)[0],in_dims=1, out_dims=1)(K,V)  # ,attn_mask=a_mask attention_mask is buggin fix later
        x = torch.vmap(lambda a: a + Q ,in_dims=1, out_dims=1)(x)
        x = self.LN2(x)
        out = torch.vmap(lambda a: torch.vmap(lambda input: self.ffn_out(self.ffn_in(input)),in_dims=1, out_dims=1)(a))(x)
        x = out + x
        x = self.LN3(x)
        return x

class decoder_module(nn.Module):
    def __init__(self):
        super().__init__()
        self.QKV_in = nn.Linear(768, 2048)
        self.QKV_out = nn.Linear(2048, 768*3)
        self.SHA1 = nn.MultiheadAttention(embed_dim=768, num_heads=1, batch_first=True)
        self.SHA2 = nn.MultiheadAttention(embed_dim=768, num_heads=1, batch_first=True)
        self.LN1 = nn.LayerNorm(768)
        self.LN2 = nn.LayerNorm(768)
        self.LN3 = nn.LayerNorm(768)
        self.ffn_in = nn.Linear(768, 2048)
        self.ffn_out = nn.Linear(2048, 768)
        
    def forward(self, x, K, V, a_mask):
        QKV = torch.vmap(lambda a:torch.vmap(lambda input: self.QKV_out(self.QKV_in(input)),in_dims=1, out_dims=1)(a))(x)
        xQ, xK, xV = torch.tensor_split(QKV, 3 , dim=-1)
        SHA1_out = torch.vmap(lambda q, k, v : self.SHA1(q,k,v)[0], in_dims=1, out_dims=1)(xQ,xK,xV)  # ,attn_mask=a_mask attention_mask is buggin fix later
        x = SHA1_out + x
        Q = self.LN1(x)  # may have to be vmaped
        x = torch.vmap(lambda q,k,v : self.SHA2(q,k,v)[0], in_dims=1, out_dims=1)(Q,K,V)  # ,attn_mask=a_mask attention_mask is buggin fix later
        x = x + Q
        x = self.LN2(x)  # may have to be vmaped
        out = torch.vmap(lambda a: torch.vmap(lambda input: self.ffn_out(self.ffn_in(input)),in_dims=1,out_dims=1)(a))(x)
        x = out + x
        x = self.LN3(x) # may have to be vmaped
        return x

class compress_module(nn.Module):
    def __init__(self):
        super().__init__()
        self.Q_up = nn.Linear(768*2, 2048)
        self.Q_int = nn.Linear(2048, 2048)
        self.Q_down = nn.Linear(2048, 768)

        self.KV_up = nn.Linear(768*2, 2048)
        self.KV_int = nn.Linear(2048, 2048)
        self.KV_down = nn.Linear(2048, 768)
    
    def forward(self, x, K, V):
        x1,x2 = torch.tensor_split(x, 2, dim=1)
        x = torch.cat((x1,x2), dim=-1)
        x = torch.vmap(lambda a: torch.vmap(lambda input: self.Q_down(self.Q_int(self.Q_up(input))),in_dims=1, out_dims=1)(a))(x)

        K1,K2 = torch.tensor_split(K, 2, dim=1)
        K = torch.cat((K1,K2), dim=-1)
        K = torch.vmap(lambda a: torch.vmap(lambda input: self.KV_down(self.KV_int(self.KV_up(input))),in_dims=1, out_dims=1)(a))(K)

        V1,V2 = torch.tensor_split(V, 2, dim=1)
        V = torch.cat((V1,V2), dim=-1)
        V = torch.vmap(lambda a: torch.vmap(lambda input: self.KV_down(self.KV_int(self.KV_up(input))),in_dims=1, out_dims=1)(a))(V)
        
        return x , K, V


class RUA(nn.Module):  # Read, Understand, Answer
    def __init__(self, num_encoder_sequences = 128, num_final_decoders=1, decoder_sequence_length=1024):
        super().__init__()

        config = transformers.BertConfig(vocab_size = 30522, hidden_size = 768 ,intermediate_size = 3072, max_position_embeddings = 1024, type_vocab_size = 2)
        bert = BertModel(config)
        self.embed = bert.embeddings

        self.encoder = bert_encoder_module(decoder_sequence_length=decoder_sequence_length)
        self.decoder_Q = initial_decoder_module()
        self.decoder_intermediate = decoder_module()
        self.compressed_decoders = nn.ModuleList([decoder_module() for _ in range(int(math.log2(num_encoder_sequences)))])
        self.compress = compress_module()
        self.final_decoders = nn.ModuleList([decoder_module() for _ in range(num_final_decoders)])#decoder_module()
    
    def forward(self, x, encoder_input_ids, encoder_token_type_ids, encoder_attention_mask):
        """
        if x[0]["input_ids"].shape[-2:] != (1024,768):
            raise AttributeError("Input seq")
        """

        input_ids = x["input_ids"]
        token_type_ids = x["token_type_ids"]
        attention_mask = x["attention_mask"].bool()

        x = self.embed(input_ids, token_type_ids)
        K_V = self.encoder(input_ids=encoder_input_ids, 
                           token_type_ids=encoder_token_type_ids, 
                           attention_mask=encoder_attention_mask)
        K,V = torch.tensor_split(K_V, 2, dim=3)

        x = self.decoder_Q(x,K,V, attention_mask)
        x = self.decoder_intermediate(x,K,V,attention_mask)
        

        for dec in self.compressed_decoders:
            x = dec(x,K,V,attention_mask)
            if x.size(dim=0) != 1:
                x, K, V = self.compress(x, K, V)
        
        for dec in self.final_decoders:
            x = dec(x,K,V,attention_mask)
        return x
        