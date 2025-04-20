"""
Training components for time series anomaly detection.
"""
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, restore_best_weights: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change in validation loss to be considered an improvement
            restore_best_weights: Whether to restore model weights to the best epoch
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_loss = float('inf')
        self.best_weights = None
        self.should_stop = False
        
        print(f"Initialized EarlyStopping with patience={patience}, min_delta={min_delta}")
    
    def __call__(self, model: nn.Module, current_loss: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            model: Current model
            current_loss: Current validation loss
            
        Returns:
            True if training should stop, False otherwise
        """
        if current_loss + self.min_delta < self.best_loss:
            # Improvement found
            self.best_loss = current_loss
            self.counter = 0
            
            if self.restore_best_weights:
                # Clone model weights
                self.best_weights = {name: param.clone() for name, param in model.state_dict().items()}
        else:
            # No improvement
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.should_stop = True
                print("Early stopping triggered")
        
        return self.should_stop
    
    def restore_weights(self, model: nn.Module) -> None:
        """
        Restore model weights to the best ones.
        
        Args:
            model: Model to restore weights for
        """
        if self.restore_best_weights and self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            print("Restored best model weights")


class Trainer:
    """Handles model training and evaluation."""
    
    def __init__(self, 
                 model: nn.Module, 
                 optimizer_name: str = "adam",
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-5,
                 device: Optional[str] = None):
        """
        Initialize the trainer.
        
        Args:
            model: PyTorch model to train
            optimizer_name: Name of optimizer to use ('adam', 'sgd', etc.)
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Move model to device
        self.model = model.to(self.device)
        
        # Set optimizer
        if optimizer_name.lower() == "adam":
            self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name.lower() == "sgd":
            self.optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        # Set loss function
        self.criterion = nn.MSELoss()
        
        # Initialize training history
        self.history = {
            "train_loss": [],
            "val_loss": []
        }
        
        print(f"Initialized Trainer with {optimizer_name} optimizer, learning_rate={learning_rate}, device={self.device}")
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        
        for batch in train_loader:
            # Move batch to device
            batch = batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch)
            loss = self.criterion(outputs, batch)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        self.history["train_loss"].append(avg_loss)
        
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate the model.
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                batch = batch.to(self.device)
                
                # Forward pass
                outputs = self.model(batch)
                loss = self.criterion(outputs, batch)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        self.history["val_loss"].append(avg_loss)
        
        return avg_loss
    
    def train(self, 
              train_loader: DataLoader, 
              val_loader: Optional[DataLoader] = None, 
              epochs: int = 100,
              patience: int = 10,
              verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data (if None, uses train_loader)
            epochs: Number of epochs to train for
            patience: Patience for early stopping
            verbose: Whether to print progress
            
        Returns:
            Dictionary containing training history
        """
        # Use train_loader for validation if val_loader is not provided
        if val_loader is None:
            val_loader = train_loader
            print("No validation loader provided, using training loader for validation")
        
        # Initialize early stopping
        early_stopping = EarlyStopping(patience=patience)
        
        # Training loop
        print(f"Starting training for {epochs} epochs")
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Train for one epoch
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            epoch_time = time.time() - epoch_start
            
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - "
                     f"Train Loss: {train_loss:.6f}, "
                     f"Val Loss: {val_loss:.6f}, "
                     f"Time: {epoch_time:.2f}s")
            
            # Check early stopping
            if early_stopping(self.model, val_loss):
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Restore best weights
        early_stopping.restore_weights(self.model)
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f}s")
        
        return self.history
    
    def get_reconstructions(self, data_loader: DataLoader) -> torch.Tensor:
        """
        Get reconstructions for the data.
        
        Args:
            data_loader: DataLoader containing data
            
        Returns:
            Tensor of reconstructions
        """
        self.model.eval()
        reconstructions = []
        
        with torch.no_grad():
            for batch in data_loader:
                # Move batch to device
                batch = batch.to(self.device)
                
                # Forward pass
                outputs = self.model(batch)
                
                # Move outputs to CPU
                reconstructions.append(outputs.cpu())
        
        return torch.cat(reconstructions, dim=0)