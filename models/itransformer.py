import torch
import torch.nn as nn
from layers.transformer_encdec import Encoder, EncoderLayer
from layers.self_attention_family import FullAttention, AttentionLayer
from layers.embed import DataEmbedding_inverted


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.input_length
        self.pred_len = configs.output_length
        self.mlp_hidden_dim = configs.mlp_hidden_dim
        self.c_mark = configs.c_mark 

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.input_length, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        
        # Decoder / Projection Head
        flatten_dim = (configs.input_vars + configs.c_mark) * configs.d_model
        
        self.output_projection = nn.Sequential(
            nn.Linear(flatten_dim, self.mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(self.mlp_hidden_dim, self.pred_len)
        )

            
    def forecast(self, x_enc, x_mark_enc):
        
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        
        B, N_vars, D_model = enc_out.shape
        dec_out = enc_out.reshape(B, N_vars * D_model)
        dec_out = self.output_projection(dec_out)
        dec_out = dec_out.unsqueeze(-1)
        
        return dec_out

    def forward(self, history_power, history_nwp, future_power, future_nwp):
       
        history_nwp_vars = history_nwp

        B, L, _ = history_nwp.shape
        x_mark_enc = torch.zeros([B, L, self.c_mark]).to(history_nwp.device)
        
        x_enc = torch.cat([history_nwp_vars, history_power], dim=-1)

        dec_out = self.forecast(x_enc, x_mark_enc)
        
        return dec_out