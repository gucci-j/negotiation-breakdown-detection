import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math


class SelfAttention(nn.Module):
    def __init__(self, attention_size, attention_type='dot_product'):
        super(SelfAttention, self).__init__()

        if attention_type == 'dot_product':
            self.scale = 1. / math.sqrt(attention_size)
        else:
            raise NotImplementedError()


    def forward(self, query, key, value, mask=None):
        """
        Args: 
            query: (batch_size, hidden_dim)
            key: (seq_len, batch_size, hidden_dim)
            value: (seq_len, batch_size, hidden_dim)
            mask: (batch_size, seq_len)
        
        Returns:
            attention_output: (batch_size, hidden_dim)
            attention_weight: (seq_len, batch_size)
        """

        # set up
        query = query.unsqueeze(1) # -> (batch_size, 1, hidden_dim)
        key = key.permute(1, 2, 0) # -> (batch_size, hidden_dim, seq_len)
        value = value.transpose(0, 1) # -> (batch_size, seq_len, hidden_dim)

        # compute a scaled dot product
        attention_weight = torch.bmm(query, key).mul_(self.scale) 
        ## -> (batch_size, 1, seq_len)

        # apply a mask
        if mask is not None:
            mask = mask.unsqueeze(1) # -> (batch_size, 1, seq_len)
            attention_weight.masked_fill((1 - mask).bool(), float('-inf'))
        
        # compute attention weight
        attention_weight = F.softmax(attention_weight, dim=2)
        
        # compute output
        attention_output = torch.bmm(attention_weight, value) 
        ## -> (batch_size, 1, hidden_dim)
        attention_output = attention_output.squeeze(1) 
        ## -> (batch_size, hidden_dim)

        return attention_output, attention_weight


class Model(nn.Module):
    def __init__(self, rnn_type, input_dim, hidden_dim, num_layers, 
                 recur_d_rate, dense_d_rate, 
                 bidirectional=False, is_attention=False):
        """
        Args:
            rnn_type (str): 'LSTM' or 'GRU'
            input_dim (int): the dim of input
            hidden_dim (int): the dim of hidden states
            num_layers (int): the number of layers
            recur_d_rate (float): the ratio of a recurrent dropout
            dense_d_rate (float): the ratio of a dense dropout
            bidirectional (bool): whether to consider bidirectional
            is_attention (bool): whether to consider an attention
        """
        super(Model, self).__init__()
        self.dropout = nn.Dropout(dense_d_rate)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(input_dim, hidden_dim, num_layers, 
                                            dropout=recur_d_rate, bidirectional=bidirectional)
        
        if bidirectional is True:
            self.decoder = nn.Linear(hidden_dim * 2, 1)
            if is_attention is True:
                self.selfattention = SelfAttention(hidden_dim * 2)
        else:
            self.decoder = nn.Linear(hidden_dim, 1)
            if is_attention is True:
                    self.selfattention = SelfAttention(hidden_dim)

        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.is_attention = is_attention

    
    def forward(self, x, attention_mask=None):
        """
        Args:  
            x: (seq_len, batch_size, 10)
            attention_mask: (batch_size, seq_len)

        Returns:  
            (batch_size, 1)
        """

        output, hidden = self.rnn(x)
        # output -> (seq_len, batch_size, hidden_dim)
        # hidden -> GRU (num_layers, batch_size, hidden_dim)
        #           LSTM (h_n: (num_layers, batch_size, hidden_dim),
        #                 c_n: (num_layers, batch_size, hidden_dim))

        if self.rnn_type == 'LSTM':
            if self.bidirectional is True:
                last_hidden = self.dropout(torch.cat((hidden[0][-2,:,:], hidden[0][-1,:,:]), dim=1))
                # -> (batch_size, hidden_dim * 2)
                if self.is_attention:
                    last_hidden, attn_weights = self.selfattention(query=last_hidden, 
                                                                   key=output, 
                                                                   value=output, 
                                                                   mask=attention_mask)
            else:
                last_hidden = self.dropout(hidden[0][-1,:,:]) # -> (batch_size, hidden_dim)
                if self.is_attention:
                    last_hidden, attn_weights = self.selfattention(query=last_hidden, 
                                                                   key=output, 
                                                                   value=output, 
                                                                   mask=attention_mask)

        else: # GRU
            if self.bidirectional is True:
                last_hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
                # -> (batch_size, hidden_dim * 2)
                if self.is_attention:
                    last_hidden, attn_weights = self.selfattention(query=last_hidden, 
                                                                   key=output, 
                                                                   value=output, 
                                                                   mask=attention_mask)
            else:
                last_hidden = self.dropout(hidden[-1,:,:]) # -> (batch_size, hidden_dim)
                if self.is_attention:
                    last_hidden, attn_weights = self.selfattention(query=last_hidden, 
                                                                   key=output,
                                                                   value=output, 
                                                                   mask=attention_mask)
        
        decoded = self.decoder(last_hidden) # -> (batch_size, 1)
        return decoded.squeeze(1)