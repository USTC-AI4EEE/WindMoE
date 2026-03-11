import torch
import torch.nn as nn
import torch.nn.functional as F

class MovingAvg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class MIC(nn.Module):
    def __init__(self, feature_size, n_heads=4, dropout=0.05, decomp_kernel=[32]):
        super(MIC, self).__init__()
        self.feature_size = feature_size
        self.conv_kernel = decomp_kernel
        
        self.isometric_conv = nn.ModuleList([
            nn.Sequential(
                nn.AvgPool1d(kernel_size=k, stride=k, padding=0),
                nn.Conv1d(feature_size, feature_size, kernel_size=1), 
                nn.Upsample(scale_factor=k, mode='linear', align_corners=True) 
            ) for k in self.conv_kernel
        ])
        
        self.conv = nn.Conv1d(feature_size, feature_size, kernel_size=3, padding=1, bias=False)
        
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        
        self.fusion = nn.Conv1d(feature_size * (len(self.conv_kernel) + 1), feature_size, kernel_size=1)

    def forward(self, x):
        x_local = self.conv(x)
        
        x_global_list = []
        L = x.shape[-1] 
        
        for i, layer in enumerate(self.isometric_conv):
            k = self.conv_kernel[i]
            
            if L < k or L % k != 0:
                target_len = k * ((L // k) + 1)
                pad_len = target_len - L
                
                x_padded = F.pad(x, (0, pad_len), mode='replicate')
                
                out = layer(x_padded)
                
                out = out[:, :, :L]
                x_global_list.append(out)
            else:
                x_global_list.append(layer(x))
            
        x_all = torch.cat([x_local] + x_global_list, dim=1) 
        x_out = self.fusion(x_all)
        return self.activation(x_out)

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.seq_len = args.input_length
        self.pred_len = args.output_length
        self.enc_in = args.input_vars
        self.d_model = args.d_model
        
        self.decomp = SeriesDecomp(args.kernel_size)
        
        self.mic_blocks = nn.ModuleList([
            MIC(feature_size=self.enc_in, decomp_kernel=args.micn_conv_kernel)
            for _ in range(args.e_layers)
        ])
        
        self.seasonal_pred = nn.Linear(self.seq_len, self.pred_len)
        self.trend_pred = nn.Linear(self.seq_len, self.pred_len)
        proj_hidden = max(4, (self.enc_in + 1) // 2)
        self.projection = nn.Sequential(
            nn.Linear(self.enc_in, proj_hidden),
            nn.GELU(),
            nn.Linear(proj_hidden, 1)
        )

    def forward(self, history_power, history_nwp, future_power, future_nwp):
        x = torch.cat([history_nwp, history_power], dim=-1)
        
        seasonal_init, trend_init = self.decomp(x)
        
        # --- Trend Prediction ---
        trend_part = self.trend_pred(trend_init.permute(0, 2, 1)).permute(0, 2, 1)
        
        # --- Seasonal Prediction ---
        x_s = seasonal_init.permute(0, 2, 1) # [B, C, L]
        
        for mic in self.mic_blocks:
            x_s = x_s + mic(x_s) 
            
        seasonal_part = self.seasonal_pred(x_s).permute(0, 2, 1) # [B, Pred_Len, C]
        
        # --- Final Combine ---
        y_pred_multi = seasonal_part + trend_part
        
        # --- Projection ---
        out = self.projection(y_pred_multi)
        
        return out