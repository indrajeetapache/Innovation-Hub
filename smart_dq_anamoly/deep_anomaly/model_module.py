"""
Model components for time series anomaly detection.
"""
import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    """LSTM Autoencoder for time series anomaly detection."""
    
    def __init__(self, 
                 input_dim: int = 1, 
                 hidden_dim: int = 64, 
                 layer_dim: int = 2, 
                 dropout: float = 0.2):
        """
        Initialize the LSTM Autoencoder.
        
        Args:
            input_dim: Number of input features
            hidden_dim: Size of hidden layers
            layer_dim: Number of LSTM layers
            dropout: Dropout probability
        """
        super(LSTMAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        
        # Encoder
        self.lstm_encoder = nn.LSTM(
            input_dim, hidden_dim, layer_dim, 
            batch_first=True, dropout=dropout if layer_dim > 1 else 0
        )
        
        # Decoder
        self.lstm_decoder = nn.LSTM(
            hidden_dim, hidden_dim, layer_dim, 
            batch_first=True, dropout=dropout if layer_dim > 1 else 0
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        
        print(f"Initialized LSTMAutoencoder with input_dim={input_dim}, "
              f"hidden_dim={hidden_dim}, layer_dim={layer_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input tensor of shape [batch_size, seq_length, input_dim]
            
        Returns:
            Reconstructed tensor of the same shape
        """
        batch_size = x.size(0)
        
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).to(x.device)
        
        # Encoder
        _, (hn, cn) = self.lstm_encoder(x, (h0, c0))
        
        # Use the last hidden state for all time steps in decoder input
        decoder_input = hn[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        
        # Decoder
        h0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).to(x.device)
        outputs, _ = self.lstm_decoder(decoder_input, (h0, c0))
        
        # Output layer
        outputs = self.output_layer(outputs)
        
        return outputs


class CNNLSTMAutoencoder(nn.Module):
    """
    CNN-LSTM Autoencoder for time series anomaly detection.
    This architecture uses CNN layers for feature extraction before LSTM layers.
    """
    
    def __init__(self, 
                 input_dim: int = 1, 
                 hidden_dim: int = 64, 
                 layer_dim: int = 2,
                 cnn_filters: int = 32,
                 kernel_size: int = 3,
                 dropout: float = 0.2):
        """
        Initialize the CNN-LSTM Autoencoder.
        
        Args:
            input_dim: Number of input features
            hidden_dim: Size of hidden layers
            layer_dim: Number of LSTM layers
            cnn_filters: Number of CNN filters
            kernel_size: Size of CNN kernel
            dropout: Dropout probability
        """
        super(CNNLSTMAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        
        # CNN encoder
        self.cnn_encoder = nn.Sequential(
            nn.Conv1d(input_dim, cnn_filters, kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.Conv1d(cnn_filters, cnn_filters, kernel_size, padding=kernel_size//2),
            nn.ReLU()
        )
        
        # LSTM encoder
        self.lstm_encoder = nn.LSTM(
            cnn_filters, hidden_dim, layer_dim, 
            batch_first=True, dropout=dropout if layer_dim > 1 else 0
        )
        
        # LSTM decoder
        self.lstm_decoder = nn.LSTM(
            hidden_dim, hidden_dim, layer_dim, 
            batch_first=True, dropout=dropout if layer_dim > 1 else 0
        )
        
        # CNN decoder (transpose convolution)
        self.cnn_decoder = nn.Sequential(
            nn.Conv1d(hidden_dim, cnn_filters, kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.Conv1d(cnn_filters, input_dim, kernel_size, padding=kernel_size//2)
        )
        
        print(f"Initialized CNNLSTMAutoencoder with input_dim={input_dim}, "
              f"hidden_dim={hidden_dim}, layer_dim={layer_dim}, cnn_filters={cnn_filters}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input tensor of shape [batch_size, seq_length, input_dim]
            
        Returns:
            Reconstructed tensor of the same shape
        """
        batch_size, seq_length, _ = x.size()
        
        # CNN requires [batch, channels, length] format
        x_cnn = x.transpose(1, 2)  # Now [batch, input_dim, seq_length]
        
        # Apply CNN
        cnn_features = self.cnn_encoder(x_cnn)  # [batch, cnn_filters, seq_length]
        
        # Convert back to LSTM format [batch, seq_length, features]
        lstm_input = cnn_features.transpose(1, 2)  # Now [batch, seq_length, cnn_filters]
        
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).to(x.device)
        
        # Encoder
        _, (hn, cn) = self.lstm_encoder(lstm_input, (h0, c0))
        
        # Use the last hidden state for all time steps in decoder input
        decoder_input = hn[-1].unsqueeze(1).repeat(1, seq_length, 1)
        
        # Decoder
        h0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).to(x.device)
        lstm_outputs, _ = self.lstm_decoder(decoder_input, (h0, c0))
        
        # Apply CNN decoder
        cnn_input = lstm_outputs.transpose(1, 2)  # [batch, hidden_dim, seq_length]
        outputs = self.cnn_decoder(cnn_input)  # [batch, input_dim, seq_length]
        
        # Convert back to [batch, seq_length, input_dim]
        outputs = outputs.transpose(1, 2)
        
        return outputs


class ModelFactory:
    """Factory class for creating different types of anomaly detection models."""
    
    @staticmethod
    def create_model(model_type: str, **kwargs) -> nn.Module:
        """
        Create and return a model instance based on the specified type.
        
        Args:
            model_type: Type of model to create ('lstm_ae', 'cnn_lstm_ae', etc.)
            **kwargs: Additional parameters to pass to the model constructor
            
        Returns:
            PyTorch model instance
        """
        model_type = model_type.lower()
        
        if model_type == "lstm_ae":
            model = LSTMAutoencoder(**kwargs)
        elif model_type == "cnn_lstm_ae":
            model = CNNLSTMAutoencoder(**kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        print(f"Created model of type {model_type}")
        return model