import torch
from torch import nn
from layers.transformer_encdec import Encoder, EncoderLayer
from layers.self_attention_family import FullAttention, AttentionLayer
from layers.embed import PatchEmbedding


class Transpose(nn.Module):
    
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
   
    def __init__(self, args):
        super().__init__()
        
        self.pred_len = args.output_length
        self.seq_len = args.input_length
        self.patch_len = args.patch_len
        self.stride = args.stride
        self.d_model = args.d_model
        
        padding = self.stride
        self.patch_embedding = PatchEmbedding(
            self.d_model, self.patch_len, self.stride, padding, args.dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, args.n_heads, attention_dropout=args.dropout, output_attention=False), 
                        self.d_model, 
                        args.n_heads
                    ),
                    self.d_model,
                    args.d_ff,
                    dropout=args.dropout,
                    activation=args.activation
                ) for l in range(args.e_layers)
            ],
            norm_layer=nn.Sequential(
                Transpose(1, 2), 
                nn.BatchNorm1d(self.d_model), 
                Transpose(1, 2)
            )
        )

        self.patch_num = int((self.seq_len - self.patch_len) / self.stride + 2)
        head_nf = self.d_model * self.patch_num
        self.head = FlattenHead(n_vars=-1, nf=head_nf, target_window=self.pred_len, head_dropout=args.dropout)
        self.projection = None

    def _create_projection_mlp(self, n_vars):
        
        hidden_dim = (n_vars + 1) // 2
        
        self.projection = nn.Sequential(
            nn.Linear(n_vars, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        ).to(next(self.parameters()).device)
        print(f"Dynamically created projection MLP for n_vars={n_vars}")


    def forward(self, history_power, history_nwp, future_power, future_nwp):
        
        x_enc = torch.cat([history_nwp, history_power], dim=-1)
        n_vars = x_enc.shape[-1] 
        if self.projection is None:
            self._create_projection_mlp(n_vars)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        x_enc = x_enc.permute(0, 2, 1)
        enc_out, _ = self.patch_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)
        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = self.projection(dec_out)

        return dec_out