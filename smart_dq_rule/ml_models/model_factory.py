"""
Model Factory for Smart DQ Rule System.

This module implements the Factory pattern for creating model instances
based on configuration parameters. It centralizes model creation logic
and makes it easy to switch between different model implementations.

The factory supports:
1. Local HuggingFace models (loaded directly)
2. HuggingFace API-based models (accessed via API)
3. Future expansion to other model types

By using this factory, other components don't need to know the details
of how each model type is initialized or configured.
"""

from typing import Dict, Any, Optional
import logging

# Import base model interface
from ml_models.base_model import BaseModel

# # Configure logger
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)

class ModelFactory:
    """Factory for creating ML model instances."""
    
    def create_model(self, 
                    model_type: str, 
                    model_name: str, 
                    **kwargs) -> BaseModel:
        """
        Create a model instance.
        
        Args:
            model_type: Type of model to create ('huggingface', 'huggingface_api', etc.)
            model_name: Name of the model to use
            **kwargs: Additional model parameters
            
        Returns:
            Model instance
            
        Raises:
            ValueError: If the model type is unknown
        """
        print(f"Creating {model_type} model: {model_name}")
        
        if model_type.lower() == "huggingface":
            # Import here to avoid circular imports
            from ml_models.huggingface_model import HuggingFaceModel
            return HuggingFaceModel(model_name=model_name, **kwargs)
        
        elif model_type.lower() == "huggingface_api":
            # Import here to avoid circular imports
            from ml_models.huggingface_api_model import HuggingFaceAPIModel
            return HuggingFaceAPIModel(model_name=model_name, **kwargs)
        
        else:
            error_msg = f"Unknown model type: {model_type}"
            print(error_msg)
            raise ValueError(error_msg)
    
    @staticmethod
    def get_default_model(model_config: Optional[Dict[str, Any]] = None) -> BaseModel:
        """
        Get a default model based on config or sensible defaults.
        
        Args:
            model_config: Optional configuration for the model
            
        Returns:
            Default model instance
        """
        config = model_config or {}
        model_type = config.get("type", "huggingface")
        model_name = config.get("name", "mistralai/Mixtral-8x7B-Instruct-v0.1")
        
        factory = ModelFactory()
        return factory.create_model(model_type=model_type, model_name=model_name, **config)