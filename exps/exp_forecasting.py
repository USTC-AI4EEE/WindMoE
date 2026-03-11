import time
import torch
import numpy as np
import torch.nn as nn
from torch import optim
import os
import joblib

from exps.exp_basic import ExpBasic
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping
from utils.metrics import CustomLoss, calculate_metrics

class ExpForecasting(ExpBasic):
    def __init__(self, args):
        super().__init__(args)
        
        self.scaler = self._load_scaler()
    
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
            print("Please run the data preprocessing script first to generate the scaler file.")
            return None

    def _build_model(self):
        if self.args.model not in self.model_dict:
            raise ValueError(f"Model {self.args.model} not found in model_dict.")
        model = self.model_dict[self.args.model].Model(self.args)
        return model

    def _get_data(self, use):
        _, data_loader = data_provider(self.args, use)
        return data_loader

    def _select_optimizer(self):
        return optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return CustomLoss()

    def train(self):
        train_loader = self._get_data(use='train')
        vali_loader = self._get_data(use='valid')
        
        if not train_loader or not vali_loader:
            print("Failed to get data, terminating training.")
            return

        if callable(getattr(self.model, '_lazy_init', None)) and not getattr(self.model, 'initialized', True):
            print("Model requires lazy initialization. Running a dummy forward pass...")
            
            try:
                dummy_batch = next(iter(train_loader))
            except StopIteration:
                print("Error: Training data loader is empty. Cannot initialize model.")
                return

            dummy_batch = [d.to(self.device) for d in dummy_batch]
            try:
                with torch.no_grad():
                    self.model.eval()
                    self.model(*dummy_batch)
                    self.model.train()
                print("Model initialized successfully.")
            except Exception as e:
                print(f"FATAL: Error during model's dummy forward pass for initialization: {e}")
                raise e

        optimizer = self._select_optimizer()
        criterion = self._select_criterion()

        checkpoint_path = os.path.join(self.output_dir, 'checkpoint.pth')
        early_stopping = EarlyStopping(
            patience=self.args.patience, 
            verbose=True,
            path=checkpoint_path
        )

        for epoch in range(self.args.epochs):
            self.model.train()
            train_loss = []
            
            for i, (history_power, history_nwp, future_power, future_nwp) in enumerate(train_loader):
                optimizer.zero_grad()
                history_power, history_nwp, future_power, future_nwp = (
                    d.to(self.device) for d in [history_power, history_nwp, future_power, future_nwp]
                )
                
                outputs = self.model(history_power, history_nwp, future_power, future_nwp)
                loss = criterion(outputs, future_power)
                train_loss.append(loss.item())
                
                loss.backward()
                if self.args.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_value)
                optimizer.step()
            
            avg_train_loss = np.average(train_loss)
            avg_val_loss = self.vali(vali_loader, criterion)
            
            print(f"Epoch: {epoch + 1} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

            early_stopping(avg_val_loss, self.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break


    def vali(self, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for i, (history_power, history_nwp, future_power, future_nwp) in enumerate(vali_loader):
                history_power, history_nwp, future_power, future_nwp = (
                    d.to(self.device) for d in [history_power, history_nwp, future_power, future_nwp]
                )

                outputs = self.model(history_power, history_nwp, future_power, future_nwp)
                loss = criterion(outputs, future_power)
                total_loss.append(loss.item())
        return np.average(total_loss)


    def test(self):
        checkpoint_path = os.path.join(self.output_dir, 'checkpoint.pth')
        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint file not found at {checkpoint_path}. Testing with the final model state.")
        else:
            print(f"Loading best model from: {checkpoint_path}")
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))

        test_loader = self._get_data(use='test')

        self._run_test_on_loader(test_loader, "Normal Test")


    def _run_test_on_loader(self, dataloader, test_name):
        if self.scaler is None:
            print(f"[{test_name}] Error: Scaler is not loaded. Cannot proceed.")
            return

        self.model.eval()
        all_preds, all_trues = [], []
        
        with torch.no_grad():
            for i, (history_power, history_nwp, future_power, future_nwp) in enumerate(dataloader):
                history_power, history_nwp, future_power, future_nwp = (
                    d.float().to(self.device) for d in [history_power, history_nwp, future_power, future_nwp]
                )
                
                outputs = self.model(history_power, history_nwp, future_power, future_nwp)

                all_preds.append(outputs.cpu())
                all_trues.append(future_power.cpu())
        
        preds_norm = torch.cat(all_preds, dim=0).numpy()
        trues_norm = torch.cat(all_trues, dim=0).numpy()

        num_features = self.scaler.n_features_in_
        
        preds_reshaped = preds_norm.reshape(-1, 1)
        trues_reshaped = trues_norm.reshape(-1, 1)

        preds_template = np.zeros((preds_reshaped.shape[0], num_features))
        trues_template = np.zeros((trues_reshaped.shape[0], num_features))

        target_feature_index = -1 
        preds_template[:, target_feature_index] = preds_reshaped.flatten()
        trues_template[:, target_feature_index] = trues_reshaped.flatten()

        preds_inversed = self.scaler.inverse_transform(preds_template)[:, target_feature_index]
        trues_inversed = self.scaler.inverse_transform(trues_template)[:, target_feature_index]

        preds_final = preds_inversed.reshape(preds_norm.shape)
        trues_final = trues_inversed.reshape(trues_norm.shape)

        metrics = calculate_metrics(trues_final, preds_final)
        
        print(f"[{test_name}] Results:")
        print(f"MAE: {metrics['mae']*100:.2f}%, RMSE: {metrics['rmse']*100:.2f}%")