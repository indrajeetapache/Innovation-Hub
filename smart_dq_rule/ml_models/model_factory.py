"""
Factory for creating ML model instances.
"""
from typing import Dict, Any

from smart_dq_rule.ml_models.base_model import BaseModel
from smart_dq_rule.ml_models.huggingface_model import HuggingFaceModel


class ModelFactory:
    """Factory for creating model instances."""
    
    @staticmethod
    def create_model(model_type: str, config: Dict[str, Any]) -> BaseModel:
        """
        Create a model instance based on type and configuration.
        
        Args:
            model_type: Type of model to create ('huggingface', 'huggingface_api', etc.)
            config: Configuration for the model
            
        Returns:
            An instance of a model
            
        Raises:
            ValueError: If model_type is not supported
        """
        print(f"Creating model of type: {model_type}")
        
        if model_type.lower() in ['huggingface', 'transformers']:
            return HuggingFaceModel(config)
        elif model_type.lower() == 'huggingface_api':
            # Import API model only when needed to avoid circular imports
            from smart_dq_rule.ml_models.huggingface_api_model import HuggingFaceAPIModel
            return HuggingFaceAPIModel(config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")