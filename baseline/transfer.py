import numpy as np
import torch
import torch.nn as nn

class SparrowSearchAlgorithm:
    
    def __init__(self, fitness_func, dim, pop_size, max_iter, lb, ub, device, model, dataloader, target_layer, criterion):
        self.fitness_func = fitness_func
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.lb = lb
        self.ub = ub
        self.device = device
        self.model = model
        self.dataloader = dataloader
        self.target_layer = target_layer
        self.criterion = criterion

        self.ST = 0.8
        self.PD = 0.2
        self.SD = 0.1

        self.producer_count = int(pop_size * self.PD)
        self.scrounger_count = pop_size - self.producer_count

        self.g_best_pos = np.zeros(dim)
        self.g_best_fitness = np.inf

        self.positions = np.random.uniform(low=self.lb, high=self.ub, size=(self.pop_size, self.dim))
        self.fitness = np.full(self.pop_size, np.inf)

    def _update_model_weights(self, layer, weights_vector):
       
        weight_numel = layer.weight.numel()
        bias_numel = 0
        if layer.bias is not None:
            bias_numel = layer.bias.numel()

        expected_dim = weight_numel + bias_numel
        assert len(weights_vector) == expected_dim, f"Dimension mismatch: expected {expected_dim}, got {len(weights_vector)}"

        new_weight = torch.tensor(weights_vector[:weight_numel], dtype=torch.float32, device=self.device).view_as(layer.weight)

        with torch.no_grad():
            layer.weight.copy_(new_weight)
            if layer.bias is not None:
                new_bias = torch.tensor(weights_vector[weight_numel:], dtype=torch.float32, device=self.device).view_as(layer.bias)
                layer.bias.copy_(new_bias)

    def _calculate_fitness(self, position_vector):
        self._update_model_weights(self.target_layer, position_vector)
        return self.fitness_func(self.model, self.dataloader, self.criterion, self.device)

    def run(self):
        for i in range(self.pop_size):
            self.fitness[i] = self._calculate_fitness(self.positions[i])
            if self.fitness[i] < self.g_best_fitness:
                self.g_best_fitness = self.fitness[i]
                self.g_best_pos = self.positions[i].copy()
        
        for t in range(self.max_iter):
            sorted_indices = np.argsort(self.fitness)
            best_pos = self.positions[sorted_indices[0]].copy()
            worst_fitness = self.fitness[sorted_indices[-1]]
            worst_pos = self.positions[sorted_indices[-1]].copy()

            for i in range(self.producer_count):
                idx = sorted_indices[i]
                r2 = np.random.rand()
                if r2 < self.ST:
                    self.positions[idx, :] *= np.exp(-i / (np.random.rand() * self.max_iter))
                else:
                    self.positions[idx, :] += np.random.randn() * np.ones(self.dim)
            
            for i in range(self.producer_count, self.pop_size):
                idx = sorted_indices[i]
                if i > self.pop_size / 2:
                    self.positions[idx, :] = np.random.randn() * np.exp((worst_pos - self.positions[idx, :]) / (i**2))
                else:
                    A = np.ones((self.dim, 1))
                    A_plus = np.linalg.pinv(A.T @ A) @ A.T  
                    self.positions[idx, :] = best_pos + np.abs(self.positions[idx, :] - best_pos) * A_plus.squeeze()
    
            danger_indices = np.random.choice(self.pop_size, int(self.pop_size * self.SD), replace=False)
            for i in danger_indices:
                if self.fitness[i] > self.g_best_fitness:
                    self.positions[i, :] = self.g_best_pos + np.random.randn(self.dim) * np.abs(self.positions[i, :] - self.g_best_pos)
                else:
                    k = 2 * np.random.rand() - 1
                    self.positions[i, :] += k * ( (np.abs(self.positions[i, :] - worst_pos)) / (self.fitness[i] - worst_fitness + 1e-8) )

            self.positions = np.clip(self.positions, self.lb, self.ub)

            for i in range(self.pop_size):
                self.fitness[i] = self._calculate_fitness(self.positions[i])
                if self.fitness[i] < self.g_best_fitness:
                    self.g_best_fitness = self.fitness[i]
                    self.g_best_pos = self.positions[i].copy()
            
            print(f"Iteration {t+1}/{self.max_iter}, Best Fitness: {self.g_best_fitness}")
        return self.g_best_pos, self.g_best_fitness

def ssa_fitness_function(model, dataloader, criterion, device):
   
    model.eval()
    total_loss = 0.0
    if not dataloader:
        return float('inf')
        
    with torch.no_grad():
        for i, (history_power, history_nwp, future_power, future_nwp) in enumerate(dataloader):
            history_power, history_nwp, future_power, future_nwp = (
                d.to(device) for d in [history_power, history_nwp, future_power, future_nwp]
            )
            outputs = model(history_power, history_nwp, future_power, future_nwp)
            loss = criterion(outputs, future_power)
            total_loss += loss.item()

    if len(dataloader) == 0:
        return float('inf')
        
    return total_loss / len(dataloader)