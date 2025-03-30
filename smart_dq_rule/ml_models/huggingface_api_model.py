"""
HuggingFace API model interface for LLMs used in DQ rule recommendation.
"""
import json
import requests
from typing import Dict, List, Any

from smart_dq_rule.ml_models.base_model import BaseModel
from smart_dq_rule.rule_engines.rule_catalog import RuleCatalog


class HuggingFaceAPIModel(BaseModel):
    """Interface for HuggingFace models via API."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the HuggingFace API model interface.
        
        Args:
            config: Dictionary containing model configuration including:
                - model_id: HuggingFace model ID
                - api_token: HuggingFace API token
                - api_url: API endpoint URL (optional)
        """
        self.model_id = config.get("model_id")
        self.api_token = config.get("api_token")
        self.api_url = config.get("api_url", "https://api-inference.huggingface.co/models/")
        self.headers = {"Authorization": f"Bearer {self.api_token}"}
        self.rule_catalog = RuleCatalog()
        
        print(f"Initialized HuggingFace API model interface for {self.model_id}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response from the model based on the prompt.
        
        Args:
            prompt: The input prompt for the model
            kwargs: Additional keyword arguments for generation
                - max_length: Maximum length of generated response
                - temperature: Sampling temperature
        
        Returns:
            The generated text response
        """
        url = f"{self.api_url}{self.model_id}"
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": kwargs.get("max_length", 512),
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.95),
                "return_full_text": False
            }
        }
        
        print(f"Sending request to API for prompt: {prompt[:100]}...")
        
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "")
            
            return str(result)
        except Exception as e:
            print(f"API request error: {str(e)}")
            return f"Error: {str(e)}"
    
    def _create_column_analysis_prompt(self, column_name: str, column_metadata: Dict[str, Any]) -> str:
        """
        Create a prompt for column analysis based on metadata.
        
        Args:
            column_name: Name of the column
            column_metadata: Dictionary containing column metadata and statistics
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""
                    I need to identify appropriate data quality rules for a column in a financial dataset.

                    Column Name: {column_name}

                    Column Metadata:
                    - Data Type: {column_metadata.get('data_type', 'Unknown')}
                    - Description: {column_metadata.get('description', 'No description available')}
                    - Is PII: {column_metadata.get('is_pii', False)}
                    - Nullable: {column_metadata.get('nullable', True)}

                    Column Statistics:
                    - Null Count: {column_metadata.get('null_count', 'Unknown')}
                    - Null Percentage: {column_metadata.get('null_percentage', 'Unknown')}%
                    - Distinct Count: {column_metadata.get('distinct_count', 'Unknown')}
                    - Distinct Percentage: {column_metadata.get('distinct_percentage', 'Unknown')}%
                    - Min Value: {column_metadata.get('min', 'N/A')}
                    - Max Value: {column_metadata.get('max', 'N/A')}
                    - Mean Value: {column_metadata.get('mean', 'N/A')}
                    - Sample Values: {', '.join(str(v) for v in column_metadata.get('sample_values', [])[:5])}

                    Based on this information, recommend the most appropriate data quality rules for this column.
                    For each rule, provide:
                    1. Rule type (e.g., completeness, validity, consistency, etc.)
                    2. Rule definition in plain English
                    3. Implementation guidance (e.g., regex pattern, threshold values)
                    4. Severity level (Critical, High, Medium, Low)

                    Return your response as a JSON list where each item has the following structure:
                    {{
                    "rule_type": "string",
                    "rule_name": "string",
                    "rule_description": "string",
                    "implementation": "string",
                    "severity": "string",
                    "rationale": "string"
                    }}
                    """
        return prompt
    
    def _parse_model_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse the model's response to extract rule recommendations.
        
        Args:
            response: Model-generated response text
            
        Returns:
            List of dictionaries containing rule recommendations
        """
        try:
            # Extract JSON from response (handle case where model adds text before/after JSON)
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            
            if json_start == -1 or json_end == 0:
                # Try to find JSON object if not a list
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                
                if json_start == -1 or json_end == 0:
                    print("Could not extract JSON from response")
                    return []
                
                # If it's a single object, wrap it in a list
                json_str = response[json_start:json_end]
                return [json.loads(json_str)]
            
            json_str = response[json_start:json_end]
            return json.loads(json_str)
            
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON from response: {e}")
            return []
    
    def analyze_column(self, column_name: str, column_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Analyze a data column and recommend appropriate DQ rules.
        
        Args:
            column_name: Name of the column
            column_metadata: Dictionary containing column metadata and statistics
            
        Returns:
            List of dictionaries containing recommended DQ rules
        """
        print(f"Analyzing column: {column_name}")
        prompt = self._create_column_analysis_prompt(column_name, column_metadata)
        response = self.generate(prompt, max_length=1024, temperature=0.2)
        
        rule_recommendations = self._parse_model_response(response)
        print(f"Extracted {len(rule_recommendations)} rule recommendations for column {column_name}")
        
        # Validate and enhance rule recommendations
        validated_rules = []
        for rule in rule_recommendations:
            # Check if rule_type is in our catalog
            if rule.get("rule_type") in self.rule_catalog.get_rule_types():
                # Add additional metadata
                rule["column_name"] = column_name
                rule["source"] = "ml_recommendation"
                rule["model_id"] = self.model_id
                validated_rules.append(rule)
            else:
                print(f"Rule type '{rule.get('rule_type')}' not in catalog, skipping")
        
        print(f"Returning {len(validated_rules)} validated rules for column {column_name}")
        return validated_rules