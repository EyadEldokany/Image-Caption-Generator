import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import math
import pandas as pd
from PositionalEncoding import PositionalEncoding
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Define the Transformer Decoder model that generates captions from image features
# Define the Transformer Decoder model that generates captions from image features
class ImageCaptionModel(nn.Module):
    def __init__(self, n_head, n_decoder_layer, vocab_size, embedding_size, encoder_output_dim):
        super(ImageCaptionModel, self).__init__()
        self.pos_encoder = PositionalEncoding(embedding_size, 0.1)
        self.TransformerDecoderLayer = nn.TransformerDecoderLayer(d_model=embedding_size, nhead=n_head)
        self.TransformerDecoder = nn.TransformerDecoder(decoder_layer=self.TransformerDecoderLayer, num_layers=n_decoder_layer)
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.last_linear_layer = nn.Linear(embedding_size, vocab_size)

        # Projection layer to match encoder output with decoder input
        self.encoder_projection = nn.Linear(encoder_output_dim, embedding_size)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.last_linear_layer.bias.data.zero_()
        self.last_linear_layer.weight.data.uniform_(-initrange, initrange)
        self.encoder_projection.weight.data.uniform_(-initrange, initrange)
        self.encoder_projection.bias.data.zero_()

    def generate_Mask(self, size, decoder_inp):
        decoder_input_mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        decoder_input_mask = decoder_input_mask.float().masked_fill(decoder_input_mask == 0, float('-inf')).masked_fill(decoder_input_mask == 1, float(0.0))

        decoder_input_pad_mask = decoder_inp.float().masked_fill(decoder_inp == 0, float(0.0)).masked_fill(decoder_inp > 0, float(1.0))
        decoder_input_pad_mask_bool = decoder_inp == 0

        return decoder_input_mask, decoder_input_pad_mask, decoder_input_pad_mask_bool

    def forward(self, encoded_image, decoder_inp):
        # Project encoder output to match decoder input size
        encoded_image = self.encoder_projection(encoded_image)  # shape: (batch, seq_len, embedding_size)
        encoded_image = encoded_image.permute(1, 0, 2)  # shape: (seq_len, batch, embedding_size)

        # Decoder input embedding + positional encoding
        decoder_inp_embed = self.embedding(decoder_inp) * math.sqrt(self.embedding_size)
        decoder_inp_embed = self.pos_encoder(decoder_inp_embed)
        decoder_inp_embed = decoder_inp_embed.permute(1, 0, 2)

        # Generate masks
        decoder_input_mask, decoder_input_pad_mask, decoder_input_pad_mask_bool = self.generate_Mask(decoder_inp.size(1), decoder_inp)
        decoder_input_mask = decoder_input_mask.to(device)
        decoder_input_pad_mask_bool = decoder_input_pad_mask_bool.to(device)

        # Transformer decoding
        decoder_output = self.TransformerDecoder(
            tgt=decoder_inp_embed,
            memory=encoded_image,
            tgt_mask=decoder_input_mask,
            tgt_key_padding_mask=decoder_input_pad_mask_bool
        )

        final_output = self.last_linear_layer(decoder_output)

        return final_output, decoder_input_pad_mask