"""
HuggingFace model interface for LLMs used in DQ rule recommendation.
"""
import json
from typing import Dict, List, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from ml_models.base_model import BaseModel
from rule_engines.rule_catalog import RuleCatalog


class HuggingFaceModel(BaseModel):
    """Interface for locally loaded HuggingFace models."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the HuggingFace model interface.
        
        Args:
            config: Dictionary containing model configuration including:
                - model_id: HuggingFace model ID
                - tokenizer: Optional pre-loaded tokenizer
                - model: Optional pre-loaded model
                - device: Device to run model on ('cuda', 'cpu')
        """
        self.model_id = config.get("model_id")
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.rule_catalog = RuleCatalog()
        
        # Use pre-loaded tokenizer and model if provided, otherwise load them
        self.tokenizer = config.get("tokenizer")
        self.model = config.get("model")
        
        if self.tokenizer is None or self.model is None:
            print(f"Loading tokenizer and model for {self.model_id}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, 
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device
            )
            print(f"Model loaded on {self.device}")
        
        print(f"Initialized HuggingFace model interface for {self.model_id}")
    
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
        print(f"Generating response for prompt: {prompt[:100]}...")
        
        # Format prompt based on model type
        if "mistral" in self.model_id.lower():
            formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        elif "llama" in self.model_id.lower():
            formatted_prompt = f"<|system|>\nYou are a helpful AI assistant specialized in data quality.\n<|user|>\n{prompt}\n<|assistant|>"
        else:
            formatted_prompt = prompt
        
        # Prepare inputs
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        
        # Set generation parameters
        generation_config = {
            "max_new_tokens": kwargs.get("max_length", 512),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.95),
            "do_sample": kwargs.get("temperature", 0.7) > 0,
            "pad_token_id": self.tokenizer.eos_token_id
        }
        
        # Generate response
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                **generation_config
            )
        
        # Decode response
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract assistant's response for chat models
        if "mistral" in self.model_id.lower():
            parts = response.split("[/INST]")
            if len(parts) > 1:
                response = parts[-1].strip()
        elif "llama" in self.model_id.lower():
            parts = response.split("<|assistant|>")
            if len(parts) > 1:
                response = parts[-1].strip()
        
        print(f"Generated response length: {len(response)} chars")
        return response
    
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
