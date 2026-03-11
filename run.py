import os
import torch
import random
import warnings
import argparse
import numpy as np
from exps.exp_forecasting import ExpForecasting
from exps.exp_moe import ExpMOE

def main():
    warnings.filterwarnings('ignore')
    
    parser = argparse.ArgumentParser(description='Wind Power Forecasting Framework')

    parser.add_argument('--optimize_router', action='store_true', 
                        help='If set, run hyperparameter optimization for the MoE router instead of training.')
    parser.add_argument('--opt_seeds', type=int, nargs='+', default=None,
                        help='A list of seeds to use for each optimization trial.')
    parser.add_argument('--exp_type', type=str, default='moe', choices=['base','moe', 'gan', 'transfer'],
                        help='Experiment type: Mixture of Experts, GAN-augmented, or Transfer Learning.')
    parser.add_argument('--is_training', type=lambda x: (str(x).lower() == 'true'), default=True, help='train or test')
    parser.add_argument('--model', type=str, default='itransformer', help='model name, e.g.itransformer')
    parser.add_argument('--dataset', type=str, default='goldwind', help='dataset name, e.g.goldwind')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id to use')

    parser.add_argument('--data_path', type=str, default='./datasets/goldwind/datanpy/', help='root path of the .npy data files')
    parser.add_argument('--weather', type=str, default='coldwave', help='weather condition')
    parser.add_argument('--station', type=int, default=225, help='station id')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    
    parser.add_argument('--input_length', type=int, default=32, help='input sequence length (e.g., 24 hours)')
    parser.add_argument('--output_length', type=int, default=16, help='prediction sequence length')
    parser.add_argument('--input_vars', type=int, default=7, help='Encoder input size (total number of variables). Auto-detected by the experiment class.')
    
    parser.add_argument('--expert1', type=str, default='dlinear', help='Ensemble: name of the first expert model')
    parser.add_argument('--expert2', type=str, default='dlinear', help='Ensemble: name of the second expert model')
    parser.add_argument('--router_hidden_dim', type=int, default=64, help='Ensemble: hidden dimension of the router')
    parser.add_argument('--router_feat_dim', type=int, default=32, help='Ensemble: feature dimension for the router')
    parser.add_argument('--router_rnn_hidden_dim', type=int, default=64, help='Router: hidden dimension of the GRU encoder.')
    parser.add_argument('--router_rnn_layers', type=int, default=1, help='Router: number of layers for the GRU encoder.')
    parser.add_argument('--router_attention_heads', type=int, default=4, help='Router: number of heads for the attention mechanism.')
    parser.add_argument('--router_pred_mlp_hidden_dim', type=int, default=32, help='Router: hidden dimension for the MLP processing expert predictions.')
    parser.add_argument('--router_fusion_hidden_dim', type=int, default=128, help='Router: hidden dimension of the final fusion MLP.')
    parser.add_argument('--hard_sample_weight_factor', type=float, default=5.0, 
                    help='Boosting factor for training Expert2 based on Expert1s errors.')
    parser.add_argument('--router_pred_gru_hidden_dim', type=int, default=32, help='Router: hidden dimension for the prediction GRU encoder')
    parser.add_argument('--router_pred_gru_layers', type=int, default=1, help='Router: number of layers for the prediction GRU encoder')
        
    parser.add_argument('--router_patch_len', type=int, default=16, help='Router: length of a patch for the router input')
    parser.add_argument('--router_stride', type=int, default=16, help='Router: stride between patches for the router input')
    
    parser.add_argument('--kde_sigma', type=float, default=5.0,
                        help='Sigma (bandwidth) for the Gaussian kernel in the "kde" weighting scheme. Controls the smoothness.')
    
    parser.add_argument('--save_preds', type=lambda x: (str(x).lower() == 'true'), default=False, 
                        help='Enable saving of test set predictions to .npy files.')
    parser.add_argument('--pred_save_path', type=str, default='./plot/datanpy', 
                        help='Path to save the prediction .npy files.')
    
    parser.add_argument('--patch_len', type=int, default=8, help='PatchTST: length of a patch')
    parser.add_argument('--stride', type=int, default=8, help='PatchTST: stride between patches')
    
    parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='number of heads in multi-head attention')
    parser.add_argument('--e_layers', type=int, default=3, help='number of encoder layers')
    parser.add_argument('--d_ff', type=int, default=256, help='dimension of feedforward network')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--activation', type=str, default='gelu', help='activation function (e.g., relu, gelu)')
    
    parser.add_argument('--individual', action='store_true', help='DLinear: individual linear layer for each channel. Set this flag to use it.')
    parser.add_argument('--kernel_size', type=int, default=25, help='DLinear: moving average kernel size')
    
    parser.add_argument('--input_features', type=int, default=16, help='Encoder input size (total number of variables). Auto-detected by the experiment class.')
    parser.add_argument('--c_mark', type=int, default=4, help='Number of time features for the embedding layer (placeholder dimension)')
    parser.add_argument('--mlp_hidden_dim', type=int, default=128, help='iTransformer: hidden dimension of the final MLP projection layer')
    parser.add_argument('--embed', type=str, default='timeF', help='Time feature embedding type [timeF, fixed, learned]')
    parser.add_argument('--freq', type=str, default='h', help='Frequency for time features [s, t, h, d, w, m]')
    parser.add_argument('--factor', type=int, default=3, help='Attention factor for some attention mechanisms')

    parser.add_argument('--gan_epochs', type=int, default=50, help='Epochs for each phase of TimeGAN training')
    parser.add_argument('--gan_hidden_dim', type=int, default=24, help='Hidden dimension for TimeGAN (encoder, generator etc.)')
    parser.add_argument('--gan_num_layers', type=int, default=3, help='Number of layers for TimeGAN RNNs')
    
    parser.add_argument('--ssa_pop_size', type=int, default=20, help='Population size for Sparrow Search Algorithm')
    parser.add_argument('--ssa_max_iter', type=int, default=10, help='Max iterations for Sparrow Search Algorithm')
    
    parser.add_argument('--seg_len', type=int, default=12, help='SegRNN: length of segment (similar to patch_len)')
    parser.add_argument('--micn_conv_kernel', type=list, default=[12, 24], help='MICN: downsampling conv kernel list')
    
    parser.add_argument('--epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size of train input data')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='optimizer learning rate')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--clip_value', type=float, default=1, help='gradient clipping value')
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    
    args = parser.parse_args()
    
    fix_seed = args.seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(fix_seed)
        torch.cuda.manual_seed_all(fix_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    if args.exp_type.lower() == 'moe':
        print("Experiment Type: Mixture of Experts")
        Exp = ExpMOE
    else:
        print("Experiment Type: Standard Forecasting")
        Exp = ExpForecasting

    exp = Exp(args)

    if args.is_training:
        print(">>>>>>>>>>  Starting Training  >>>>>>>>>>")
        exp.train()
        print("\n>>>>>>>>>>  Training Finished, Starting Testing on Best Model  >>>>>>>>>>")
        exp.test()
    else:
        print(">>>>>>>>>>  Starting Testing  >>>>>>>>>>")
        exp.test()

if __name__ == '__main__':
    main()