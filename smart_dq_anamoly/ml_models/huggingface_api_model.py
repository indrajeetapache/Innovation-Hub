"""
HuggingFace API Model for Smart DQ Rule System.

This module provides a class for using HuggingFace models via their API
(instead of loading models locally). This is useful for:
1. Accessing larger models than can fit in local memory
2. Using models without having to install all dependencies
3. Faster startup time since models are pre-loaded on the API

The class implements the BaseModel interface for consistent usage
across different model implementations.
"""

import requests
import json
from typing import Dict, Any, Optional, List
import logging
import time
import os

# Import base model interface
from ml_models.base_model import BaseModel

# Configure logger
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)

class HuggingFaceAPIModel(BaseModel):
    """Class for using HuggingFace models via their API."""
    
    def __init__(self, 
                model_name: str,
                api_key: Optional[str] = None,
                max_new_tokens: int = 512,
                temperature: float = 0.1,
                top_p: float = 0.95,
                api_url: Optional[str] = None,
                **kwargs):
        """
        Initialize a HuggingFace API model.
        
        Args:
            model_name: Name of the model to use (e.g., 'mistralai/Mixtral-8x7B-Instruct-v0.1')
            api_key: HuggingFace API key (if not provided, will look for HF_API_KEY env var)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling (lower = more deterministic)
            top_p: Top-p sampling parameter
            api_url: Custom API URL (if not provided, will use HuggingFace Inference API)
            **kwargs: Additional model parameters
        """
        self.name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.model_kwargs = kwargs
        
        # Get API key from kwargs, environment variable, or raise error
        self.api_key = api_key or os.environ.get("HF_API_KEY")
        if not self.api_key:
            print("No HuggingFace API key provided - API calls may fail")
        
        # Set API URL
        self.api_url = api_url or f"https://api-inference.huggingface.co/models/{model_name}"
        
        print(f"Initialized HuggingFace API model '{model_name}'")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using the HuggingFace API.
        
        Args:
            prompt: The prompt to generate from
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        print(f"Generating text via API for prompt: {prompt[:100]}...")
        
        try:
            # Override default parameters with any provided in kwargs
            max_new_tokens = kwargs.get("max_new_tokens", self.max_new_tokens)
            temperature = kwargs.get("temperature", self.temperature)
            top_p = kwargs.get("top_p", self.top_p)
            
            # Format the prompt if needed (for instruction-tuned models)
            formatted_prompt = self._format_prompt(prompt)
            
            # Prepare the payload
            payload = {
                "inputs": formatted_prompt,
                "parameters": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "do_sample": temperature > 0,
                    **self.model_kwargs
                }
            }
            
            # Prepare headers
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Make the API request
            response = requests.post(self.api_url, headers=headers, json=payload)
            
            # Check for errors
            if response.status_code != 200:
                error_msg = f"API request failed with status {response.status_code}: {response.text}"
                print(error_msg)
                
                # If the model is still loading, wait and retry once
                if response.status_code == 503 and "is currently loading" in response.text:
                    lprint("Model is loading, waiting 10 seconds and retrying...")
                    time.sleep(10)
                    response = requests.post(self.api_url, headers=headers, json=payload)
                    
                    if response.status_code != 200:
                        print(f"Retry failed with status {response.status_code}: {response.text}")
                        return f"Error: API request failed after retry - {response.text}"
                else:
                    return f"Error: {response.text}"
            
            # Parse the response
            result = response.json()
            
            # Extract the generated text
            if isinstance(result, list) and len(result) > 0:
                if "generated_text" in result[0]:
                    generated_text = result[0]["generated_text"]
                else:
                    generated_text = result[0].get("text", "")
            elif isinstance(result, dict):
                generated_text = result.get("generated_text", "")
            else:
                generated_text = str(result)
            
            # Remove the prompt from the generated text if it's included
            if generated_text.startswith(formatted_prompt):
                generated_text = generated_text[len(formatted_prompt):].strip()
            
            print(f"Generated text length: {len(generated_text)}")
            return generated_text
            
        except Exception as e:
            error_msg = f"Error generating text via API: {str(e)}"
            print(error_msg, exc_info=True)
            return error_msg
    
    def _format_prompt(self, prompt: str) -> str:
        """
        Format the prompt for the specific model architecture.
        
        Args:
            prompt: Raw prompt
            
        Returns:
            Formatted prompt suitable for the model
        """
        # Mixtral-specific formatting
        if "mixtral" in self.name.lower():
            return f"<s>[INST] {prompt} [/INST]"
        # Llama-specific formatting
        elif "llama" in self.name.lower():
            return f"<s>[INST] {prompt} [/INST]"
        # Default formatting
        else:
            return prompt