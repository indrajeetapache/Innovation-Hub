# To resolve circular imports, use one of these approaches:

# APPROACH 1: Reorganize imports in huggingface_model.py
# Instead of importing at the module level, import inside functions:

def generate(self, prompt: str) -> str:
    """Generate text using the HuggingFace model."""
    # Import only when needed, not at module level
    from rule_engines.rule_catalog import RuleCatalog
    
    # Rest of the function code...
    
# APPROACH 2: Create a separate module for shared data types

# 1. Create a file called data_types.py with shared enums/types
"""
from enum import Enum

class PIIType(Enum):
    NAME = "name"
    EMAIL = "email"
    # etc...
"""

# 2. In both modules, import from data_types instead of each other
"""
from data_types import PIIType
"""

# APPROACH 3: Use import with a relative path in your notebook
import sys
sys.path.append('/content/Innovation-Hub/smart_dq_rule')

# Then be careful with the order of imports
from rule_engines.rule_catalog import RuleCatalog
# Then import model related components
from ml_models.model_factory import ModelFactory