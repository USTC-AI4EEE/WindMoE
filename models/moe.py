import torch
import torch.nn as nn
import torch.nn.functional as F
from models import dlinear, itransformer, patchtst

EXPERT_MODEL_REGISTRY = {
    'dlinear': dlinear.Model,
    'itransformer': itransformer.Model,
    'patchtst': patchtst.Model,
}


class TimeSeriesAttentionEncoder(nn.Module):
    
    def __init__(self, input_dim: int, rnn_hidden_dim: int, rnn_layers: int, 
                 n_heads: int, dropout: float):
        super().__init__()
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=dropout if rnn_layers > 1 else 0,
            bidirectional=False
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=rnn_hidden_dim, 
            num_heads=n_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        self.summary_query = nn.Parameter(torch.randn(1, 1, rnn_hidden_dim))

        self.norm1 = nn.LayerNorm(rnn_hidden_dim)
        self.norm2 = nn.LayerNorm(rnn_hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        gru_outputs, _ = self.gru(x)
        gru_outputs = self.norm1(gru_outputs)
        B = x.size(0)
        query = self.summary_query.expand(B, -1, -1)
        keys = gru_outputs
        values = gru_outputs
        
        attn_output, _ = self.attention(query=query, key=keys, value=values)
        attended_vector = self.norm2(attn_output)
        final_vector = attended_vector.squeeze(1)
        
        return final_vector


class Router(nn.Module):
   
    def __init__(self, 
                 rnn_hidden_dim: int, 
                 rnn_layers: int,
                 n_heads: int,
                 pred_len: int, 
                 pred_mlp_hidden_dim: int,
                 fusion_hidden_dim: int,
                 num_experts: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.num_experts = num_experts
        self.initialized = False

        self.history_encoder = None 
        self.future_encoder = None
        
        self.pred_encoder = nn.Sequential(
            nn.Linear(pred_len, pred_mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(pred_mlp_hidden_dim, pred_mlp_hidden_dim),
            nn.LayerNorm(pred_mlp_hidden_dim)
        )

        total_feature_dim = rnn_hidden_dim * 2 + pred_mlp_hidden_dim * num_experts
        self.fusion_mlp = nn.Sequential(
            nn.Linear(total_feature_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.GELU(),
        )
        self.output_layer = nn.Linear(fusion_hidden_dim // 2, num_experts)
        
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_layers = rnn_layers
        self.n_heads = n_heads
        self.dropout = dropout

    def _lazy_init(self, history_nwp, history_power, future_nwp):

        history_input_dim = history_nwp.shape[-1] + history_power.shape[-1]
        future_input_dim = future_nwp.shape[-1]

        self.history_encoder = TimeSeriesAttentionEncoder(
            input_dim=history_input_dim,
            rnn_hidden_dim=self.rnn_hidden_dim,
            rnn_layers=self.rnn_layers,
            n_heads=self.n_heads,
            dropout=self.dropout
        )
        self.future_encoder = TimeSeriesAttentionEncoder(
            input_dim=future_input_dim,
            rnn_hidden_dim=self.rnn_hidden_dim,
            rnn_layers=self.rnn_layers,
            n_heads=self.n_heads,
            dropout=self.dropout
        )
        
        self.to(history_power.device)
        self.initialized = True
        print("--- Router Lazy Initialization (Attention-based) Complete ---")

    def forward(self, history_nwp: torch.Tensor, history_power: torch.Tensor, future_nwp: torch.Tensor, 
                pred1: torch.Tensor, pred2: torch.Tensor) -> torch.Tensor:
        
        if not self.initialized:
            self._lazy_init(history_nwp, history_power, future_nwp)

        history_data = torch.cat([history_nwp, history_power], dim=-1)
        history_feat = self.history_encoder(history_data) 
        future_feat = self.future_encoder(future_nwp)     

        pred1_flat = pred1.squeeze(-1)
        pred2_flat = pred2.squeeze(-1)
        pred1_feat = self.pred_encoder(pred1_flat)
        pred2_feat = self.pred_encoder(pred2_flat)

        combined_features = torch.cat([history_feat, future_feat, pred1_feat, pred2_feat], dim=1)

        fused_output = self.fusion_mlp(combined_features)
        logits = self.output_layer(fused_output)
        weights = F.softmax(logits, dim=1)
        
        return weights


class Model(nn.Module):
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        self.expert1 = self._build_expert(args.expert1)
        self.expert2 = self._build_expert(args.expert2)
        
        self.router = Router(
            rnn_hidden_dim=args.router_rnn_hidden_dim,
            rnn_layers=args.router_rnn_layers,
            n_heads=args.router_attention_heads,
            pred_len=args.output_length,
            pred_mlp_hidden_dim=args.router_pred_mlp_hidden_dim,
            fusion_hidden_dim=args.router_fusion_hidden_dim,
            num_experts=2,
            dropout=args.dropout
        )

    def _build_expert(self, model_name):
        if model_name not in EXPERT_MODEL_REGISTRY:
            raise ValueError(f"Expert model '{model_name}' is not registered.")
        return EXPERT_MODEL_REGISTRY[model_name](self.args)

    def forward(self, history_power, history_nwp, future_power, future_nwp):

        pred1 = self.expert1(history_power, history_nwp, future_power, future_nwp)
        pred2 = self.expert2(history_power, history_nwp, future_power, future_nwp)
        
        weights = self.router(history_nwp, history_power, future_nwp,
                              pred1.detach(), pred2.detach())
        
        return {
            'expert1_pred': pred1,
            'expert2_pred': pred2,
            'weights': weights
        }