import torch
import torch.nn as nn
from typing import Tuple


class MovingAvg(nn.Module):
    
    def __init__(self, kernel_size: int, stride: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        front_padding = (self.kernel_size - 1) // 2
        end_padding = self.kernel_size - 1 - front_padding
        front = x[:, 0:1, :].repeat(1, front_padding, 1)
        end = x[:, -1:, :].repeat(1, end_padding, 1)
        x = torch.cat([front, x, end], dim=1)
        x = x.permute(0, 2, 1)
        x = self.avg(x)
        x = x.permute(0, 2, 1)
        return x


class SeriesDecomp(nn.Module):
    
    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Model(nn.Module):
    
    def __init__(self, args):
        super().__init__()
        
        self.seq_len = args.input_length
        self.pred_len = args.output_length
        self.individual = args.individual
        
        n_vars = args.input_vars

        self.decomposition = SeriesDecomp(args.kernel_size)
        
        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for _ in range(n_vars):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
            
        hidden_dim = (n_vars + 1) // 2
        self.projection = nn.Sequential(
            nn.Linear(n_vars, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, history_power, history_nwp, future_power, future_nwp):

        x_enc = torch.cat([history_nwp, history_power], dim=-1)
        n_vars = x_enc.shape[-1]
        
        seasonal_init, trend_init = self.decomposition(x_enc)
        
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        if self.individual:
            seasonal_output = torch.zeros((x_enc.size(0), n_vars, self.pred_len), device=x_enc.device)
            trend_output = torch.zeros((x_enc.size(0), n_vars, self.pred_len), device=x_enc.device)
            for i in range(n_vars):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        prediction_multi = seasonal_output + trend_output
        
        prediction_multi = prediction_multi.permute(0, 2, 1)
        
        prediction_single = self.projection(prediction_multi)

        return prediction_single