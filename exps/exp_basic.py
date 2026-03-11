import os
import shutil
import torch
from models import itransformer, patchtst, dlinear, segrnn, micn

class ExpBasic:
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        
        self.model_dict = {
            'itransformer': itransformer,
            'patchtst': patchtst,
            'dlinear': dlinear,
            'segrnn': segrnn,
            'micn': micn,
        }
        self.model = self._build_model().to(self.device)
        
        self.exp_name = '_'.join([
            self.args.dataset, 
            self.args.model, 
            f"station{self.args.station}",
            f"in{self.args.input_length}",
            f"out{self.args.output_length}",
            f"seed{self.args.seed}"
        ])
        
        self.output_dir = os.path.join(self.args.checkpoints, self.exp_name)

        print('='*40)
        print('Exp:', self.exp_name)
        print('Mode:', 'Train' if args.is_training else 'Test')
        
        if args.is_training:
            os.makedirs(self.output_dir, exist_ok=True)
            print('Checkpoint Save Dir:', self.output_dir)
        print('='*40)
        
    def _acquire_device(self):
        if self.args.gpu is not None and torch.cuda.is_available():
            print(f"Using GPU: cuda:{self.args.gpu}")
            return torch.device(f'cuda:{self.args.gpu}')
        else:
            print("Using CPU")
            return torch.device('cpu')

    def _build_model(self):
        raise NotImplementedError

    def _get_data(self, use):
        raise NotImplementedError

    def vali(self, vali_loader, criterion):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError