"""
ML Rule Engine for Smart DQ Rule System.

This module defines the MLRuleEngine which connects ML models with rule suggestions.
It uses ML models to analyze column data and suggest appropriate data quality rules.

The key functionality includes:
1. Creating prompts for ML models based on column profiles and samples
2. Processing model responses to extract confidence scores for different PII types
3. Converting these confidence scores into rule suggestions
4. Providing an interface for generating rules for entire datasets

This component is separate from rule_catalog.py to avoid circular imports.
"""

from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import json
import re
import logging

# Import from our own modules - avoid circular imports
from common.data_types import PIIType
from rule_engines.rule_catalog import BaseRule, RuleCatalog

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class MLRuleEngine:
    """Engine that uses ML models to suggest data quality rules."""
    
    def __init__(self, model, rule_catalog: RuleCatalog):
        """
        Initialize the ML rule engine.
        
        Args:
            model: ML model for PII detection (Mixtral-8x7B-Instruct-v0.1)
            rule_catalog: Catalog of available rules
        """
        self.model = model
        self.rule_catalog = rule_catalog
        logger.info(f"Initialized ML Rule Engine with model: {getattr(model, 'name', type(model).__name__)}")
    
    def generate_rules_for_dataset(self, 
                                 dataframe: pd.DataFrame, 
                                 profiles: Dict[str, Dict[str, Any]]) -> Dict[str, List[BaseRule]]:
        """
        Generate rules for all columns in a dataframe.
        
        Args:
            dataframe: Input dataframe
            profiles: Dictionary mapping column names to their profiles
            
        Returns:
            Dictionary mapping column names to lists of suggested rules
        """
        rules_by_column = {}
        
        for column_name in dataframe.columns:
            if column_name in profiles:
                # Get sample values from the column
                column_sample = dataframe[column_name].dropna().sample(
                    min(10, len(dataframe[column_name].dropna())),
                    random_state=42
                )
                
                # Generate rules for this column
                rules = self.generate_rule_suggestions(
                    column_name, 
                    profiles[column_name], 
                    column_sample
                )
                
                rules_by_column[column_name] = rules
        
        return rules_by_column
    
    def generate_rule_suggestions(self, 
                                column_name: str,
                                column_profile: Dict[str, Any],
                                column_sample: pd.Series) -> List[BaseRule]:
        """
        Generate rule suggestions for a column.
        
        Args:
            column_name: Name of the column
            column_profile: Profile of the column
            column_sample: Sample data from the column
            
        Returns:
            List of suggested rules
        """
        logger.info(f"Generating rule suggestions for column: {column_name}")
        
        # Create a prompt for the ML model
        logger.debug("Creating prompt for ML model")
        prompt = self._create_column_prompt(column_name, column_profile, column_sample)
        
        # Get model predictions
        logger.debug("Getting predictions from ML model")
        predictions = self._get_model_predictions(prompt)
        logger.info(f"Received predictions for column {column_name}: {predictions}")
        
        # Get validation suggestions
        validation_prompt = self._create_validation_prompt(column_name, column_profile, column_sample)
        validation_suggestions = self._get_validation_suggestions(validation_prompt)
        
        # Combine PII and validation suggestions
        all_suggestions = {
            **predictions,
            **validation_suggestions
        }
        
        # Convert predictions to rule suggestions
        logger.debug("Converting predictions to rule suggestions")
        rules = self.rule_catalog.suggest_rules_for_column(column_profile, all_suggestions)
        logger.info(f"Generated {len(rules)} rule suggestions for column {column_name}")
        
        return rules
    
    def _create_column_prompt(self, 
                             column_name: str,
                             column_profile: Dict[str, Any],
                             column_sample: pd.Series) -> str:
        """
        Create a prompt for the ML model to analyze a column for PII.
        
        Args:
            column_name: Name of the column
            column_profile: Profile of the column
            column_sample: Sample data from the column
            
        Returns:
            Prompt string for the ML model
        """
        # Convert sample to string
        sample_values = column_sample.tolist()
        sample_str = "\n".join([str(x) for x in sample_values])
        
        # Format statistics for the prompt
        stats = []
        for key, value in column_profile.items():
            if isinstance(value, (int, float, str, bool)):
                stats.append(f"- {key}: {value}")
        
        stats_str = "\n".join(stats)
        
        # Create the final prompt
        prompt = f"""
You are a data quality expert tasked with identifying personally identifiable information (PII) in a dataset column.

Given the following information about a data column, determine if it contains personal identifiable information (PII).

Column Name: {column_name}

Column Statistics:
{stats_str}

Sample Values:
{sample_str}

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

Only include PII types with non-zero confidence scores. If you don't think the column contains any PII, return an empty JSON object.
"""
        
        return prompt
    
    def _create_validation_prompt(self,
                                 column_name: str,
                                 column_profile: Dict[str, Any],
                                 column_sample: pd.Series) -> str:
        """
        Create a prompt for the ML model to suggest validation rules.
        
        Args:
            column_name: Name of the column
            column_profile: Profile of the column
            column_sample: Sample data from the column
            
        Returns:
            Prompt string for the ML model
        """
        # Convert sample to string
        sample_values = column_sample.tolist()
        sample_str = "\n".join([str(x) for x in sample_values])
        
        # Format statistics for the prompt
        stats = []
        for key, value in column_profile.items():
            if isinstance(value, (int, float, str, bool)):
                stats.append(f"- {key}: {value}")
        
        stats_str = "\n".join(stats)
        
        # Data type information
        data_type = column_profile.get("data_type", "unknown")
        
        # Create the final prompt
        prompt = f"""
You are a data quality expert tasked with suggesting appropriate validation rules for a dataset column.

Given the following information about a data column, suggest validation rules that would be appropriate for this type of data.

Column Name: {column_name}
Data Type: {data_type}

Column Statistics:
{stats_str}

Sample Values:
{sample_str}

Based on your analysis, suggest validation rules by providing a confidence score between 0 and 1 for each rule type:

Return your suggestions in the following JSON format:
{{
    "completeness": <confidence score for completeness check>,
    "uniqueness": <confidence score for uniqueness check>,
    "format_validation": <confidence score for format validation>,
    "range_check": <confidence score for range check>,
    "consistency": <confidence score for consistency check>
}}

If a format validation seems appropriate, also suggest a regex pattern:
{{
    "regex_pattern": "<suggested regex pattern>"
}}

Only include rule types with non-zero confidence scores.
"""
        
        return prompt
    
    def _get_model_predictions(self, prompt: str) -> Dict[str, float]:
        """
        Get PII predictions from the ML model.
        
        Args:
            prompt: Prompt for the model
            
        Returns:
            Dictionary mapping PII types to confidence scores
        """
        try:
            # Call the model with the prompt
            result = self.model.generate(prompt)
            
            # Parse the response
            # Extract JSON from the response if needed
            json_match = re.search(r'({.*})', result, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                confidence_scores = json.loads(json_str)
            else:
                # Fallback if no JSON found
                confidence_scores = {}
            
            return confidence_scores
        
        except Exception as e:
            logger.error(f"Error getting model predictions: {e}", exc_info=True)
            return {}  # Return empty dict if there was an error
    
    def _get_validation_suggestions(self, prompt: str) -> Dict[str, Any]:
        """
        Get validation rule suggestions from the ML model.
        
        Args:
            prompt: Prompt for the model
            
        Returns:
            Dictionary with validation rule suggestions
        """
        try:
            # Call the model with the prompt
            result = self.model.generate(prompt)
            
            # Parse the response
            # Extract JSON from the response if needed
            json_match = re.search(r'({.*})', result, re.DOTALL)
            suggestions = {}
            
            if json_match:
                json_str = json_match.group(1)
                suggestions = json.loads(json_str)
                
                # Look for a regex pattern suggestion in a separate JSON block
                regex_match = re.search(r'({[^{}]*"regex_pattern"[^{}]*})', result, re.DOTALL)
                if regex_match:
                    regex_json = regex_match.group(1)
                    try:
                        regex_data = json.loads(regex_json)
                        if "regex_pattern" in regex_data:
                            suggestions["regex_pattern"] = regex_data["regex_pattern"]
                    except:
                        pass
            
            return suggestions
        
        except Exception as e:
            logger.error(f"Error getting validation suggestions: {e}", exc_info=True)
            return {}
    
    def format_rule_recommendations(self, 
                                  rules_by_column: Dict[str, List[BaseRule]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Format rule recommendations for output.
        
        Args:
            rules_by_column: Dictionary mapping column names to lists of rules
            
        Returns:
            Formatted recommendations
        """
        formatted_recommendations = {}
        
        for column_name, rules in rules_by_column.items():
            column_recommendations = []
            
            for rule in rules:
                recommendation = {
                    "rule_name": rule.name,
                    "description": rule.description,
                    "category": rule.category.value,
                    "confidence": rule.confidence,
                    "severity": rule.severity,
                    "parameters": rule.parameters
                }
                
                if rule.pii_type:
                    recommendation["pii_type"] = rule.pii_type.value
                
                column_recommendations.append(recommendation)
            
            # Sort by confidence
            column_recommendations.sort(key=lambda x: x["confidence"], reverse=True)
            formatted_recommendations[column_name] = column_recommendations
        
        return formatted_recommendations