"""Model interfaces for DQ rule recommendation."""

import logging

# Configure logger


# Import components (using conditional imports to avoid circular dependencies)
def _import_components():
    # Using a function to delay imports until needed
    global BaseModel, HuggingFaceModel, HuggingFaceAPIModel, ModelFactory
    
    from .base_model import BaseModel
    from .huggingface_model import HuggingFaceModel
    from .huggingface_api_model import HuggingFaceAPIModel
    from .model_factory import ModelFactory

# Only populate the namespace when explicitly requested
__all__ = ['BaseModel', 'HuggingFaceModel', 'HuggingFaceAPIModel', 'ModelFactory']

print("ml_models module initialized")