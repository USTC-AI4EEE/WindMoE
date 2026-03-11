import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.seq_len = args.input_length
        self.pred_len = args.output_length
        self.enc_in = args.input_vars
        self.d_model = args.d_model
        self.dropout = args.dropout
        self.seg_len = args.seg_len
        self.rnn_hidden_size = args.d_model 
        self.linear_patch = nn.Linear(self.seg_len, self.d_model)
        self.relu = nn.ReLU()
        
        self.gru = nn.GRU(input_size=self.d_model, hidden_size=self.rnn_hidden_size, batch_first=True)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.linear_out = nn.Linear(self.rnn_hidden_size, self.pred_len)
        
        proj_hidden = max(4, (self.enc_in + 1) // 2)
        self.projection = nn.Sequential(
            nn.Linear(self.enc_in, proj_hidden),
            nn.GELU(),
            nn.Linear(proj_hidden, 1)
        )

    def forward(self, history_power, history_nwp, future_power, future_nwp):
        x = torch.cat([history_nwp, history_power], dim=-1)
        B, L, C = x.shape
        x = x.permute(0, 2, 1).reshape(B * C, L, 1)
        if L % self.seg_len != 0:
            pad_len = self.seg_len - (L % self.seg_len)
            x = torch.cat([x, x[:, -1:, :].repeat(1, pad_len, 1)], dim=1)
        seg_num = x.shape[1] // self.seg_len
        x = x.view(B * C, seg_num, self.seg_len)
        x = self.linear_patch(x)
        x = self.relu(x)
        x = self.dropout_layer(x)
        _, hn = self.gru(x) 
        hn = hn.squeeze(0)
        y_pred = self.linear_out(hn)
        y_pred = y_pred.reshape(B, C, self.pred_len).permute(0, 2, 1)
        out = self.projection(y_pred)
        return out