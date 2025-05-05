"""
Base Model interface for Smart DQ Rule System.

This module defines the abstract base class that all model implementations must follow.
It ensures consistent APIs across different types of models, whether they are:
- Local HuggingFace models
- API-based models
- Other types of ML models

The BaseModel defines methods for:
1. Text generation from prompts
2. PII type detection in columnar data
3. Common utility functions for model interaction
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class BaseModel(ABC):
    """Abstract base class for ML models used in the rule engine."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text based on a prompt.
        
        Args:
            prompt: The prompt to generate from
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        pass
    
    def detect_pii_types(self, column_name: str, sample_data: List[str]) -> Dict[str, float]:
        """
        Detect PII types in a column.
        
        Args:
            column_name: Name of the column
            sample_data: Sample data from the column
            
        Returns:
            Dictionary mapping PII types to confidence scores
        """
        # Default implementation that can be overridden by subclasses
        # This uses the generate method and a standard prompt
        prompt = self._create_pii_detection_prompt(column_name, sample_data)
        response = self.generate(prompt)
        return self._parse_pii_response(response)
    
    def _create_pii_detection_prompt(self, column_name: str, sample_data: List[str]) -> str:
        """
        Create a prompt for PII detection.
        
        Args:
            column_name: Name of the column
            sample_data: Sample data from the column
            
        Returns:
            Prompt for PII detection
        """
        samples = "\n".join([str(s) for s in sample_data[:10]])
        
        return f"""
        Analyze the following column data and determine if it contains personally identifiable information (PII).
        
        Column Name: {column_name}
        
        Sample Data:
        {samples}
        
        For each PII type, provide a confidence score between 0 and 1 (0 means not PII, 1 means definitely PII).
        Return the results in the following JSON format:
        
        {{
            "name": <confidence score if this contains names>,
            "email": <confidence score if this contains email addresses>,
            "phone": <confidence score if this contains phone numbers>,
            "address": <confidence score if this contains physical addresses>,
            "ssn": <confidence score if this contains SSNs>,
            "date_of_birth": <confidence score if this contains dates of birth>,
            "ip_address": <confidence score if this contains IP addresses>,
            "credit_card": <confidence score if this contains credit card numbers>,
            "account_number": <confidence score if this contains account numbers>,
            "username": <confidence score if this contains usernames>,
            "password": <confidence score if this contains passwords>
        }}
        
        Only include PII types with non-zero confidence scores.
        """
    
    def _parse_pii_response(self, response: str) -> Dict[str, float]:
        """
        Parse the response from the model to extract PII confidence scores.
        
        Args:
            response: Model's response to the PII detection prompt
            
        Returns:
            Dictionary mapping PII types to confidence scores
        """
        # Default implementation that attempts to extract JSON from the response
        import json
        import re
        
        try:
            # Look for JSON pattern in the response
            json_match = re.search(r'({[\s\S]*})', response)
            if json_match:
                json_str = json_match.group(1)
                # Parse the JSON string
                return json.loads(json_str)
            else:
                return {}
        except Exception as e:
            print(f"Error parsing PII response: {e}")
            return {}