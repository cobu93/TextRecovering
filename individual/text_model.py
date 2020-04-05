import torch
import math
import torch.nn as nn

################################################################
###     Classes definitions                                  ###
################################################################
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TextModel(nn.Module):    
    def __init__(self, 
                 vocab_size,
                 embedding_size, 
                 num_heads, 
                 encoder_layers, 
                 decoder_layers, 
                 dim_feedforward,
                 dropout=0.5
        ):
        
        super(TextModel, self).__init__()
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.pos_encoder = PositionalEncoding(embedding_size, dropout)
        '''
        self.encoder = nn.TransformerEncoder(
                        nn.TransformerEncoderLayer(embedding_size, num_heads, dim_feedforward, dropout), 
                        encoder_layers
                    )

        self.decoder = nn.TransformerDecoder(
                    nn.TransformerDecoderLayer(embedding_size, num_heads, dim_feedforward, dropout), 
                    decoder_layers
                    )
        '''
        
        self.transformer = nn.Transformer(
            d_model=embedding_size, 
            nhead=num_heads, 
            num_encoder_layers=encoder_layers, 
            num_decoder_layers=decoder_layers, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout
        )
        self.linear = nn.Linear(embedding_size, vocab_size)    
        self.init_weights()    

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        #self.embedding.bias.data.zero_()
        

    def forward(self, src, tgt):        
        src = self.embedding(src) #* math.sqrt(self.embedding_size)
        tgt = self.embedding(tgt) #* math.sqrt(self.embedding_size)
        
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
        
        output = self.transformer(src, tgt)
        
        output = self.linear(output)
        return output    

################################################################
###     Functions                                            ###
################################################################

def get_loss_function():
    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    
    def my_loss_fn(target, predicted):
        predicted_size = predicted.size()
        flatten_size = predicted_size[0] * predicted_size[1]
        
        target_ = target.reshape([flatten_size])
        predicted_ = predicted.reshape([flatten_size, predicted_size[2]])
        return loss_fn(predicted_, target_)
    
    return my_loss_fn