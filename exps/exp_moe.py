import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import joblib
from scipy.ndimage import gaussian_filter1d

from exps.exp_basic import ExpBasic
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping
from utils.metrics import CustomLoss, calculate_metrics
from models import moe

class WeightedDataset(Dataset):
    def __init__(self, original_dataset, weights):
        if len(original_dataset) != len(weights):
            raise ValueError("Dataset and weights must have the same length.")
        self.original_dataset = original_dataset
        self.weights = torch.from_numpy(weights).float()
        print(f"Wrapped dataset of size {len(original_dataset)} with pre-calculated weights.")

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, index):
        original_data = self.original_dataset[index]
        weight = self.weights[index]
        return (*original_data, weight)

class ExpMOE(ExpBasic):
    def __init__(self, args):
        super().__init__(args)
        
        self.scaler = self._load_scaler()
                
        self.exp_name = '_'.join([
            'EnsembleStaged_KDE',
            f'exp1_{self.args.expert1}',
            f'exp2_{self.args.expert2}',
            f"station{self.args.station}",
            f"in{self.args.input_length}", 
            f"out{self.args.output_length}",
            f"seed{self.args.seed}"
        ])
        
        self.output_dir = os.path.join(self.args.checkpoints, self.exp_name)
        if args.is_training:
            os.makedirs(self.output_dir, exist_ok=True)
            print(f'Ensemble Checkpoint Save Dir updated to: {self.output_dir}')

    def _load_scaler(self):
        if self.args.dataset == 'fujian':
            station_id_str = f"{self.args.station:03d}"
        
        elif self.args.dataset == 'jilin':
            station_id_str = f"{self.args.station:03d}"
        
        elif self.args.dataset == 'goldwind':
            station_id_str = str(self.args.station)
            
        else:
            print(f"Warning: Unhandled dataset '{self.args.dataset}' for scaler loading. "
                  f"Using station ID without padding.")
            station_id_str = str(self.args.station)

        scaler_path = os.path.join(
            self.args.data_path,
            f'scaler_station_{station_id_str}.joblib'
        )
        try:
            scaler = joblib.load(scaler_path)
            print(f"Scaler loaded successfully from {scaler_path}")
            return scaler
        except FileNotFoundError:
            print(f"Error: Scaler file not found at {scaler_path}")
            return None

    def _build_model(self):
        model = moe.Model(self.args)
        print(f"Building Ensemble Model for Staged Training with Expert 1: {self.args.expert1} and Expert 2: {self.args.expert2}")
        return model

    def _get_data(self, use):
        return data_provider(self.args, use)

    def _select_optimizer(self, model_params):
        return optim.AdamW(model_params, lr=self.args.learning_rate)
    
    def _select_criterion(self):
        return CustomLoss()

    def _train_single_component(self, component_name, train_loader, vali_loader, component, loss_fn, checkpoint_path):
        print(f"\n----- Starting Training for Component: {component_name} -----")
        optimizer = self._select_optimizer(component.parameters())
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, path=checkpoint_path)
        for epoch in range(self.args.epochs):
            component.train()
            train_losses = []
            for batch_data in train_loader:
                optimizer.zero_grad()
                loss = loss_fn(batch_data, component)
                if loss is None: continue
                train_losses.append(loss.item())
                loss.backward()
                if self.args.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(component.parameters(), self.args.clip_value)
                optimizer.step()
            avg_train_loss = np.average(train_losses)
            avg_val_loss = self.vali_single_component(vali_loader, component, loss_fn)
            print(f"[{component_name}] Epoch: {epoch+1} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
            early_stopping(avg_val_loss, component)
            if early_stopping.early_stop:
                print(f"[{component_name}] Early stopping.")
                break
        print(f"----- Finished Training for {component_name} -----")
        print(f"Loading best model for {component_name} from {checkpoint_path}")
        component.load_state_dict(torch.load(checkpoint_path))
        
    def vali_single_component(self, vali_loader, component, loss_fn):
        component.eval()
        val_losses = []
        with torch.no_grad():
            for batch_data in vali_loader:
                loss = loss_fn(batch_data, component, is_train=False) 
                if loss is not None:
                    val_losses.append(loss.item())
        return np.average(val_losses) if val_losses else 0
    
    def _precompute_kde_weights(self, dataset, expert1_model, loss_fn_unreduced, set_name: str):
        
        print(f"--- Pre-computing KDE weights for '{set_name}' set... ---")
        if not dataset or len(dataset) == 0:
            print(f"Warning: '{set_name}' dataset is empty. Skipping weight calculation.")
            return None

        sequential_loader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers
        )
        
        all_losses = []
        with torch.no_grad():
            for batch_data in sequential_loader:
                history_power, history_nwp, future_power, future_nwp = (d.to(self.device) for d in batch_data)
                pred1 = expert1_model(history_power, history_nwp, future_power, future_nwp)
                loss1_per_sample = loss_fn_unreduced(pred1, future_power).mean(dim=(1, 2))
                all_losses.append(loss1_per_sample.cpu())
        
        base_losses_np = torch.cat(all_losses).numpy()
        smoothed_losses = gaussian_filter1d(base_losses_np, sigma=self.args.kde_sigma)
        mean_loss, std_loss = np.mean(smoothed_losses), np.std(smoothed_losses)
        std_loss = std_loss if std_loss > 1e-6 else 1.0
        normalized_losses = (smoothed_losses - mean_loss) / std_loss
        final_weights = 1.0 + self.args.hard_sample_weight_factor * normalized_losses
        final_weights = np.maximum(final_weights, 0.0)
        
        print(f"--- Finished pre-computing weights for '{set_name}' set. ---")
        return final_weights

    def train(self):
        file_prefix = '_'.join([self.args.dataset, f"station{self.args.station}", f"in{self.args.input_length}", f"out{self.args.output_length}", f"seed{self.args.seed}"])
        criterion = self._select_criterion()

        _, exp1_train_loader = self._get_data(use='train')
        _, exp1_vali_loader = self._get_data(use='valid')
        exp1_checkpoint_path = os.path.join(self.output_dir, f'{file_prefix}_expert1_best.pth')
        expert1_model = self.model.expert1

        if callable(getattr(expert1_model, '_lazy_init', None)) and not getattr(expert1_model, 'initialized', True):
            dummy_batch = next(iter(exp1_train_loader))
            history_power, history_nwp, future_power, future_nwp = [d.to(self.device) for d in dummy_batch]
            with torch.no_grad(): expert1_model.eval(); expert1_model(history_power, history_nwp, future_power, future_nwp)

        def loss_fn_exp1(batch_data, model, is_train=True):
            history_power, history_nwp, future_power, future_nwp = (d.to(self.device) for d in batch_data)
            pred = model(history_power, history_nwp, future_power, future_nwp)
            return criterion(pred, future_power)
        
        self._train_single_component("Expert1", exp1_train_loader, exp1_vali_loader, expert1_model, loss_fn_exp1, exp1_checkpoint_path)
        expert1_model.eval()
        for param in expert1_model.parameters(): param.requires_grad = False
    
        exp2_train_dataset, _ = self._get_data(use='train')
        exp2_vali_dataset, _ = self._get_data(use='valid')
        exp2_checkpoint_path = os.path.join(self.output_dir, f'{file_prefix}_expert2_best.pth')
        expert2_model = self.model.expert2
        
        if callable(getattr(expert2_model, '_lazy_init', None)) and not getattr(expert2_model, 'initialized', True):
            dummy_batch = next(iter(exp1_train_loader))
            history_power, history_nwp, future_power, future_nwp = [d.to(self.device) for d in dummy_batch]
            with torch.no_grad(): expert2_model.eval(); expert2_model(history_power, history_nwp, future_power, future_nwp)
            

        loss_fn_unreduced = nn.HuberLoss(reduction='none')
        train_weights = self._precompute_kde_weights(exp2_train_dataset, expert1_model, loss_fn_unreduced, "train")
        vali_weights = self._precompute_kde_weights(exp2_vali_dataset, expert1_model, loss_fn_unreduced, "validation")
        
        weighted_train_dataset = WeightedDataset(exp2_train_dataset, train_weights)
        exp2_train_loader_weighted = DataLoader(
            weighted_train_dataset,
            batch_size=self.args.batch_size, shuffle=True,
            num_workers=self.args.num_workers, drop_last=True
        )
        
        weighted_vali_dataset = WeightedDataset(exp2_vali_dataset, vali_weights)
        exp2_vali_loader_weighted = DataLoader(
            weighted_vali_dataset,
            batch_size=self.args.batch_size, shuffle=False,
            num_workers=self.args.num_workers
        )
            
        def loss_fn_exp2(batch_data, model, is_train=True):
            history_power, history_nwp, future_power, future_nwp, sample_weights = (d.to(self.device) for d in batch_data)
                
            pred2 = model(history_power, history_nwp, future_power, future_nwp)
            loss2_per_sample = loss_fn_unreduced(pred2, future_power).mean(dim=(1, 2))
            final_loss = (loss2_per_sample * sample_weights).mean()
            return final_loss

        self._train_single_component("Expert2", exp2_train_loader_weighted, exp2_vali_loader_weighted, expert2_model, loss_fn_exp2, exp2_checkpoint_path)

        self.model.expert2.eval()
        for param in self.model.expert2.parameters(): param.requires_grad = False
        
        _, router_train_loader = self._get_data(use='train')
        _, router_vali_loader = self._get_data(use='valid')
        
        router_model = self.model.router
        if callable(getattr(router_model, '_lazy_init', None)) and not getattr(router_model, 'initialized', True):
            dummy_batch_router = next(iter(router_train_loader))
            history_power, history_nwp, future_power, future_nwp = [d.to(self.device) for d in dummy_batch_router]
            
            pred_shape = (history_power.size(0), self.args.output_length, 1)
            dummy_pred = torch.zeros(pred_shape, device=self.device)
            with torch.no_grad():
                router_model.eval()
                router_model(history_nwp, history_power, future_nwp, dummy_pred, dummy_pred)
                
        router_params = self.model.router.parameters()
        router_optimizer = self._select_optimizer(router_params)
        router_checkpoint_path = os.path.join(self.output_dir, f'{file_prefix}_ensemble_best.pth') 
        router_early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, path=router_checkpoint_path)

        for epoch in range(self.args.epochs):
            self.model.router.train()
            train_losses = []
            for batch_data in router_train_loader:
                router_optimizer.zero_grad()
                history_power, history_nwp, future_power, future_nwp = (d.to(self.device) for d in batch_data)
                
                with torch.no_grad():
                    pred1 = self.model.expert1(history_power, history_nwp, future_power, future_nwp)
                    pred2 = self.model.expert2(history_power, history_nwp, future_power, future_nwp)
                
                weights = self.model.router(history_nwp, history_power, future_nwp, pred1.detach(), pred2.detach())
                
                w1 = weights[:, 0].unsqueeze(-1).unsqueeze(-1)
                w2 = weights[:, 1].unsqueeze(-1).unsqueeze(-1)
                final_prediction = w1 * pred1.detach() + w2 * pred2.detach()
                
                loss = criterion(final_prediction, future_power)
                train_losses.append(loss.item())
                loss.backward()
                router_optimizer.step()

            avg_train_loss = np.average(train_losses)
            avg_val_loss = self.vali(router_vali_loader, criterion)
            print(f"[Router] Epoch: {epoch+1} | Ensemble Train Loss: {avg_train_loss:.6f} | Ensemble Val Loss: {avg_val_loss:.6f}")
            router_early_stopping(avg_val_loss, self.model)
            if router_early_stopping.early_stop:
                print("[Router] Early stopping.")
                break
                
        self.model.load_state_dict(torch.load(router_checkpoint_path))

    def vali(self, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for batch_data in vali_loader:
                history_power, history_nwp, future_power, future_nwp = (d.to(self.device) for d in batch_data)
                outputs = self.model(history_power, history_nwp, future_power, future_nwp)
                pred1, pred2, weights = outputs['expert1_pred'], outputs['expert2_pred'], outputs['weights']
                w1 = weights[:, 0].unsqueeze(-1).unsqueeze(-1)
                w2 = weights[:, 1].unsqueeze(-1).unsqueeze(-1)
                final_prediction = w1 * pred1 + w2 * pred2
                loss = criterion(final_prediction, future_power)
                total_loss.append(loss.item())
        return np.average(total_loss)

    def test(self):
        file_prefix = '_'.join([
            self.args.dataset, f"station{self.args.station}",
            f"in{self.args.input_length}", f"out{self.args.output_length}",
            f"seed{self.args.seed}"
        ])
        checkpoint_path = os.path.join(self.output_dir, f'{file_prefix}_ensemble_best.pth')
        
        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint file not found at {checkpoint_path}. Testing with the final model state.")
        else:
            print(f"Loading best model from: {checkpoint_path}")
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))

        _, test_loader = self._get_data(use='test')

        print("\n--- Testing on Normal Test Set ---")
        self._run_test_on_loader(test_loader, "Normal Test")

    def _run_test_on_loader(self, dataloader, test_name):
        if self.scaler is None:
            print(f"[{test_name}] Error: Scaler is not loaded. Cannot proceed.")
            return

        self.model.eval()
        all_preds, all_trues = [], []
        with torch.no_grad():
            for i, batch_data in enumerate(dataloader):
                batch_data_device = [d.to(self.device) for d in batch_data]
                history_power, history_nwp, future_power, future_nwp = batch_data_device
                outputs = self.model(history_power, history_nwp, future_power, future_nwp)
                w1 = outputs['weights'][:, 0].unsqueeze(-1).unsqueeze(-1)
                w2 = outputs['weights'][:, 1].unsqueeze(-1).unsqueeze(-1)
                final_prediction = w1 * outputs['expert1_pred'] + w2 * outputs['expert2_pred']
                all_preds.append(final_prediction.cpu())
                all_trues.append(future_power.cpu())
        
        preds = torch.cat(all_preds, dim=0).numpy()
        trues = torch.cat(all_trues, dim=0).numpy()
        
        num_features = self.scaler.n_features_in_
        preds_reshaped = preds.reshape(-1, 1)
        trues_reshaped = trues.reshape(-1, 1)
        preds_template = np.zeros((preds_reshaped.shape[0], num_features))
        trues_template = np.zeros((trues_reshaped.shape[0], num_features))
        preds_template[:, -1] = preds_reshaped.flatten()
        trues_template[:, -1] = trues_reshaped.flatten()
        preds_inverted = self.scaler.inverse_transform(preds_template)[:, -1].reshape(preds.shape)
        trues_inverted = self.scaler.inverse_transform(trues_template)[:, -1].reshape(trues.shape)

        metrics = calculate_metrics(trues_inverted, preds_inverted)
        
        print(f"[{test_name}] Results:")
        print(f"MAE: {metrics['mae']*100:.2f}%, RMSE: {metrics['rmse']*100:.2f}%")