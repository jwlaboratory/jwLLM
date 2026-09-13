import torch
import torch.nn as nn
import math


class Tokenizer():
    def __init__ (self):
        self.tokens = {}
        self.counter = 1 # start at 1, becuase pad is 0

    def forward(self, input):

        # get the max token size and pad them all out
        max_seq_len = 0
        for ar in input:
            max_seq_len = max(len(ar), max_seq_len)
        

        out = []
        for ar in input:
            temp = []
            for word in ar:
                if word in self.tokens:
                    temp.append(self.tokens[word])
                else:
                    self.tokens[word] = self.counter
                    temp.append(self.tokens[word])
                    self.counter += 1

            # pad
            while len(temp) < max_seq_len:
                temp.append(0)
            
            out.append(temp)
        return torch.tensor(out)
            

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)

        # embedding table is vocab_size cols, d_model rows
        # you index in based on vocab_index and out get a d_model size row

        # 0th indexed, can get index out of range. need to use torch.tensor in it


    def forward(self, x):
        # tokenized input
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len, dropout):

        super().__init__()
        self.d_model = d_model
        self.seq_len = max_seq_len
        self.dropout = nn.Dropout(dropout) #randomly will drop out with p, to make 0, to help with overfitting

        # we want one unique positional encoding for each position in sequence lenght. therfor the rows = # of sequence length
        # we want each positional encoding to add to the original encoding

        pe = torch.zeros(max_seq_len, d_model ) # seq i, seq i+1, seq i+2 is next row etc . # dmodel size for each
        # position 0 -> [?, ?, ?, ?]
        # position 1 -> [?, ?, ?, ?]
        # ...

        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        # creates a [0, 1, 2, 3, 4, ... seq_len]
        # unsqueeze creates dimension at index 1, so 
        # [[0], 
        #  [1]
        #  [2]
        #  [3]]
        # to a new shape 5x1
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # torch.arange (0, d_model, 2) is skip every 2, go from 0 to d_model like [0,2,4,6]
        # so this is length of d_model / 2
        # this makes sense bc we are only doing half for sin and half for cos

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        # this register buffer says its a stored in the export, but not needed to be updated/ not trainable 
        self.register_buffer('pe', pe)


        # note we build this on torch so that it could theretically stay on gpu. otherwise, if we create an array then convert it, it would need to start in cpu
    def forward(self, x):

        # x is gonna be [batch size, sequence length, d_model]
        # [num unique sequences at once, sequence legnth, d_model for each part of sequence]


        # this means: on the [batch size dim], do all.
        # on the [sequence length dim], do up to the sequence length
        # on the 3rd dem, it should be d_model so we want to do all
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        # requires grad is off because we dont need to train the positional encoding, tis fixed
        # note that this is still register_buffer so its exported 
        

        # dropout for training sometimes
        return self.dropout(x)


class LayerNorm(nn.Module):
    # layernorm is normalizing per each batch item
    # we take all per the batch and noramlzie based on mean and std. we give it alpha (multipliciative) and bias (additive) as learned parameters so the model can tune it it self

    def __init__(self, eps: float= 10 ** -6):
        super().__init__()

        self.eps = eps

                                    # tensor of size 1
                                    # paramenter makes it learnable
        self.alpha = nn.Parameter(torch.ones(1)) #multiply
        self.bias = nn.Parameter(torch.zeros(1)) # add


    def forward(self, x):

        # assuming shape [batch, sequence, embedding]
        # dim=-1 means the embedding dimension. so this gets mean PER token in sequence in batch. this isn't averaging accross a single sequences, or multiple sequences 


        # dim = -1 is last dimension
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim=True)

        # keep dims means this. rn we found average in the last dimension, so we'd expect a tensor of shape:
        # [batch size, sequence size, ], NOT, [batch size, sequence size, embeddings]; because we just averaged embeddings
        # keep dims just keeps the dimensionally size=1. so itll be:
        # [batch size, sequence size, 1]

        # this helps w subtraction later so we have same sizes.

        return self.alpha * ( x-mean) / (std + self.eps )  + self.bias

        # epsiolon so we dont get very small numbers that underflow or errors diviing by zero


class FFN(nn.Module):

    def __init__(self, dmodel, dff, dropout):
        super().__init__()

        self.dmodel = dmodel
        self.dff = dff
        self.dropout = nn.Dropout(dropout)

        self.linear1 = nn.Linear(dmodel, dff, bias=True) #w1 and b1 (bias included)
        self.linear2 = nn.Linear(dff, dmodel, bias=True) #w2 and b2

        #FFN(x) = max(0,xW1 + b1)W2 + b2  --> from pape. where it takes dmodel in, dmodel out, but middle is dff
        # the max (0, x) is basically a ReLU . so ReLU in the middle

    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class MultiHeadAttention(nn.Module):
    #Each head gets the entire sequence context across all tokens, but only a slice (d k) of the embedding dimension (d model) per token.
    def __init__(self, dmodel, heads, dropout):
        super().__init__()

        self.dmodel = dmodel
        self.heads = heads

        # we gonna slice dmodel into heads. we want it accross all seuqences. so therfor dmodel must be divisable by heads
        if dmodel % heads != 0:
            raise ValueError("dmodel must be divisble by the # of heads evenly")

        self.d_k = dmodel // heads #basically the size of each head

        # weights of each k/q/v
        self.w_q = nn.Linear(dmodel, dmodel)
        self.w_k = nn.Linear(dmodel, dmodel)
        self.w_v = nn.Linear(dmodel, dmodel)


        self.w_o =  nn.Linear(dmodel, dmodel) # this is a final matrix multiply after the heads

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def selfAttention(query, key, value, mask, dropout):
        d_k = query.shape[-1]

        attention_scores = query @ (key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)

        attention_scores = attention_scores.softmax(dim = -1)
        # softmax on last dimension, the idmensions should be: 
        if dropout != None:
            attention_scores = dropout(attention_scores)

        return attention_scores @ value, attention_scores

    def forward(self, k, q, v, mask):
        # expect x to be [batch, sequence, embeddings]

        query = self.w_q(q)
        value = self.w_v(v)
        key = self.w_k(k) #[batch, sequence dmodel] --> batch, sequence, dmodel

        # now we need to split up for heads
        # key is the heads are splti accross sequences.. so multipel words 
        
        print("q [batch, sequence, dmodel]" , query.shape)
        # batch,seq,dmodel --> batch,seq, h, d_k
        query = query.view(query.shape[0], query.shape[1], self.heads, self.d_k)
        print("q, [batch, sequence, heads, dk] ", query.shape)
        # for example, imagine previous dmodel is [a,b,c,d]
        # this view now makes the 3rd dim, (the dmodel dim): [a, b], [c,d]
        # so there is 2 heads here. 


        # batch,seq, h, d_k--> batch, h, seq, d_k
        # so that h can watch over sequence length and d_k
        query = query.transpose(1, 2)

        # this reshapes it. so we went from batch, sqeuence, heads, dk
        #  imagine from prev result, at batch size 1: [ (batch dim)
        #                                                [ (seq dim)
                                                         #
                                                            # this is each token in seuqence (now broken into 2 heads)
        #                                                   [[a,b], [c,d]],
        #                                                   [],
        #                                                   []
                                                              
        #                                                  ]
        #                                                ]

        #### NOW##### we swap the head dim with seq dim

        #                                               [ batch dim (2)
        #                                                    [
        #                                                      # seq dim (3)
        #                                                        [ [dk], [dk], [dk] ]
        #                                                       ] head one
        #                                                    [....] head two
        #                                                ]

        print("q new view slices [batch, heads, sequence, dk]", query.shape)
        print("each head is touching the entire sequence but a slice of the dk")


        # same fr key and value
        key = key.view(query.shape[0], query.shape[1], self.heads, self.d_k).transpose(1, 2)
        value = value.view(query.shape[0], query.shape[1], self.heads, self.d_k).transpose(1, 2)

        x, self_attention_scores = MultiHeadAttention.selfAttention(query, key, value, mask, self.dropout)

        #transpose it back so dk is last again
        x = x.transpose(1,2).contigious().view(x.shape[0], -1)

    #view does not calculate anything, add parameters, or copy values conceptually. It changes how PyTorch interprets the same values in memory. The actual learned transformation happened earlier

def main():
    # text to tokens
    d_model = 4
    max_seq_len = 512
    dropout = .2
    attn_heads = 2
    input = [["hi", "my", "name", "is", "shrey"], ["goodbye", "shrey"]]
    tokenizer = Tokenizer()
    tokens = tokenizer.forward(input) # expected a 2d array of batch size
    vocab_count = tokenizer.counter # counter includes the reserved PAD ID 0

    # embed the results to 3 per token
    embedding = InputEmbedding(vocab_size=vocab_count, d_model=d_model)
    embed_out = embedding.forward(tokens)
    #print(embed_out) # note that this still contains a grad_fn=<MulBackward0> to note that the operation was created by a multiplication. if we aren't doing backprop, we don't need, so we can do with torch.no_grad(). 
    # or we can do model.eval()/torch.inference mode --> useful for changing nature of droupout/batch normalization to infernece mode
    # [n, n+1, n+2] -> [na, nb, nc]
    #                  [n1a, n1b, n1c]
    #                  [n2a, n2b, n2c]



    # note that the actual input sequence should not exceed max_seq_len otherwise pos encoding will be too shor tlol
    positional_encoding = PositionalEncoding(d_model=d_model, max_seq_len=max_seq_len, dropout=dropout)
    p_encoded = positional_encoding.forward(embed_out)
    print("p_encoded: batch size, sequence length, dmodel", p_encoded.shape, p_encoded)

    attention = MultiHeadAttention(dmodel=d_model, heads=attn_heads, dropout=dropout)
    post_attn = attention.forward(p_encoded,p_encoded,p_encoded, None)


if __name__ == "__main__":
    main()
