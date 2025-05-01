import torch
import torch.nn as nn
import math
import pandas as pd
from PositionalEncoding import PositionalEncoding
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ImageCaptionModelConvNeXT(nn.Module):
    def __init__(self, n_head, n_decoder_layer, vocab_size, embedding_size, convnext_feature_size):
        super(ImageCaptionModelConvNeXT, self).__init__()
        self.convnext_feature_proj = nn.Linear(convnext_feature_size, embedding_size)
        self.pos_encoder = PositionalEncoding(embedding_size, 0.1)
        self.TransformerDecoderLayer = nn.TransformerDecoderLayer(d_model =  embedding_size, nhead = n_head)
        self.TransformerDecoder = nn.TransformerDecoder(decoder_layer = self.TransformerDecoderLayer, num_layers = n_decoder_layer)
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(vocab_size , embedding_size)
        self.last_linear_layer = nn.Linear(embedding_size, vocab_size)
        self.init_weights()
        self.dropout = nn.Dropout(0.5) # Add dropout
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.last_linear_layer.bias.data.zero_()
        self.last_linear_layer.weight.data.uniform_(-initrange, initrange)
        self.convnext_feature_proj.bias.data.zero_()
        self.convnext_feature_proj.weight.data.uniform_(-initrange, initrange)

    def generate_Mask(self, size, decoder_inp):
        decoder_input_mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        decoder_input_mask = decoder_input_mask.float().masked_fill(decoder_input_mask == 0, float('-inf')).masked_fill(decoder_input_mask == 1, float(0.0))
        decoder_input_pad_mask = decoder_inp.float().masked_fill(decoder_inp == 0, float(0.0)).masked_fill(decoder_inp > 0, float(1.0))
        decoder_input_pad_mask_bool = decoder_inp == 0
        return decoder_input_mask, decoder_input_pad_mask, decoder_input_pad_mask_bool

    def forward(self, encoded_image, decoder_inp):
        # Project ConvNeXT features and reshape
        encoded_image = self.convnext_feature_proj(encoded_image.squeeze(1).squeeze(1))  # Remove extra dimensions
        #The following line was changed
        encoded_image = encoded_image.unsqueeze(0)  # Reshape for Transformer

        decoder_inp_embed = self.embedding(decoder_inp)* math.sqrt(self.embedding_size)
        decoder_inp_embed = self.pos_encoder(decoder_inp_embed)
        decoder_inp_embed = self.dropout(decoder_inp_embed)
        decoder_inp_embed = decoder_inp_embed.permute(1,0,2)

        decoder_input_mask, decoder_input_pad_mask, decoder_input_pad_mask_bool = self.generate_Mask(decoder_inp.size(1), decoder_inp)
        decoder_input_mask = decoder_input_mask.to(device)
        decoder_input_pad_mask = decoder_input_pad_mask.to(device)
        decoder_input_pad_mask_bool = decoder_inp == 0

        decoder_output = self.TransformerDecoder(tgt = decoder_inp_embed, memory = encoded_image, tgt_mask = decoder_input_mask, tgt_key_padding_mask = decoder_input_pad_mask_bool)
        decoder_output = self.dropout(decoder_output)
        final_output = self.last_linear_layer(decoder_output)
        final_output = self.log_softmax(final_output)
        return final_output,  decoder_input_pad_mask