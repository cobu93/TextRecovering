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


class MultiTextClassifierModel(nn.Module):    
    def __init__(self, 
                 authors,
                 sequences_len,
                 vocab_size,
                 embedding_size, 
                 num_heads, 
                 encoder_layers,  
                 dim_feedforward,
                 dropout=0.5
        ):


        super(MultiTextClassifierModel, self).__init__()

        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.pos_encoder = PositionalEncoding(embedding_size, dropout)
        self.encoder = nn.TransformerEncoder(
                        nn.TransformerEncoderLayer(embedding_size, num_heads, dim_feedforward, dropout), 
                        encoder_layers
                    )

        self.dropout_1 = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(embedding_size * sequences_len, len(authors))
        self.bn_1 = nn.BatchNorm1d(len(authors))
        self.activation_1 = nn.ReLU()

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        #self.embedding.bias.data.zero_()
        

    def forward(self, source):      
        out = self.embedding(source) 
        out = self.pos_encoder(out) 
        out = self.encoder(out)
 
        out = self.dropout_1(out)   
        out = self.linear_1(out.view(out.size()[0], out.size()[1] * out.size()[2]))        
        out = self.bn_1(out)        
        out = self.activation_1(out)

        return out




class MultiTextModel(nn.Module):    
    def __init__(self, 
                 authors,
                 vocab_size,
                 embedding_size, 
                 num_heads, 
                 encoder_layers, 
                 decoder_layers, 
                 dim_feedforward,
                 dropout=0.5
        ):
        
        super(MultiTextModel, self).__init__()
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.pos_encoder = PositionalEncoding(embedding_size, dropout)
        self.encoder = nn.TransformerEncoder(
                        nn.TransformerEncoderLayer(embedding_size, num_heads, dim_feedforward, dropout), 
                        encoder_layers
                    )

        self.decoders = nn.ModuleList()
        self.linears = nn.ModuleList()
        self.num_decoders = len(authors)

        for _ in authors:
            decoder = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(embedding_size, num_heads, dim_feedforward, dropout), 
                decoder_layers
            )
            linear = nn.Linear(embedding_size, vocab_size)    
            
            self.decoders.append(decoder)
            self.linears.append(linear)

        self.init_weights()


    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        #self.embedding.bias.data.zero_()
        

    def forward(self, source, target):      
        outputs = [None] * self.num_decoders
        srcs = [None] * self.num_decoders
        tgts = [None] * self.num_decoders

        for idx in range(self.num_decoders):
            srcs[idx] = self.embedding(source[idx]) 
            tgts[idx] = self.embedding(target[idx]) 
        
            srcs[idx] = self.pos_encoder(srcs[idx])
            tgts[idx] = self.pos_encoder(tgts[idx])
        
            outputs[idx] = self.decoders[idx](tgts[idx], self.encoder(srcs[idx]))
            outputs[idx]  = self.linears[idx](outputs[idx])
        
        return outputs





################################################################
###     Functions                                            ###
################################################################

def get_loss_function(num_authors):

    loss_fns = []
    for _ in range(num_authors):
        loss_fns.append(nn.CrossEntropyLoss(reduction='mean'))
    
    def my_loss_fn(targets, predicted_outputs):
        '''
        predicted = predicted_outputs[current_training]
        predicted_size = predicted.size()
        flatten_size = predicted_size[0] * predicted_size[1]        
        target_ = target.reshape([flatten_size])
        predicted_ = predicted.reshape([flatten_size, predicted_size[2]])
        return loss_fn(predicted_, target_)
        '''

        total_loss = 0

        for idx, predicted in enumerate(predicted_outputs):
            predicted_size = predicted.size()
            flatten_size = predicted_size[0] * predicted_size[1]        
            total_loss += loss_fns[idx](
                predicted.reshape([flatten_size, predicted_size[2]]), 
                targets[idx].reshape([flatten_size])
            )

        return total_loss / len(loss_fns)
    
    return my_loss_fn


def get_classifier_loss_function(num_authors):

    loss_fn = nn.CrossEntropyLoss()
    
    def my_loss_fn(targets, predicted_outputs):

        return loss_fn(predicted_outputs, targets)
    
    return my_loss_fn