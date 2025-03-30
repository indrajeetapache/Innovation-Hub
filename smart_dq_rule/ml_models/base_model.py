from abc import ABC, abstractmethod
from typing import Dict, List, Any

class BaseModel(ABC):
    """Abstract base class for all ML model interfaces."""
    
    @abstractmethod
    def __init__(self, config: Dict[str, Any]):
        pass
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        pass
    
    @abstractmethod
    def analyze_column(self, column_name: str, column_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        pass