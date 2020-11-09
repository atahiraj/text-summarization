import torch.nn.functional as F
import torch.nn as nn

class Encoder(nn.Module):
    
    def __init__(self, input_size, hidden_size, embedding_size, batch_size, output_size, pretrained_embedding, num_layers, padding_idx, bidirectional):
        super().__init__()
        
        # ajouter des asserts
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.batch_size = batch_size 
        self.output_size = output_size
        self.pretrained_embedding = pretrained_embedding
        self.num_layers = num_layers
        self.padding_idex = padding_idex
        self.directions = 2 if bidirectional else 1
        
        # BIBI ALOVIOU SO METCH :kiss: :kiss_closed_eyes:
        self.embedding_layer = nn.Embedding(self.input_size, self.embedding_size, padding_idx = self.padding_idx)
        self.embedding_layer.weight.data.copy_(self.pretrained_embedding.weight.data)
        self.rnn = nn.lstm(
            input_size = self.embedding_size,
            hidden_size = self.hidden_size,
            num_layers = self.num_layers,
            bidirectional = bidirectional
        )
        
        
        