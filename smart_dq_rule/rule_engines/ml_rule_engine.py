"""
ML-based rule recommendation engine.
"""
from typing import Dict, List, Any, Optional

from smart_dq_rule.rule_engines.base_rule_engine import BaseRuleEngine
from smart_dq_rule.ml_models.base_model import BaseModel


class MLRuleEngine(BaseRuleEngine):
    """ML-based engine for recommending DQ rules."""
    
    def __init__(self, model: BaseModel, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ML rule engine.
        
        Args:
            model: ML model to use for rule recommendation
            config: Dictionary containing rule engine configuration
        """
        self.model = model
        self.config = config or {}
        print("Initialized ML-based rule engine")
    
    def suggest_rules(self, column_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Suggest DQ rules for a column based on metadata.
        
        Args:
            column_metadata: Dictionary containing column metadata and statistics
            
        Returns:
            List of dictionaries containing recommended DQ rules
        """
        column_name = column_metadata.get("column_name")
        print(f"Suggesting rules for column: {column_name}")
        
        try:
            # Use ML model to analyze column and suggest rules
            rule_recommendations = self.model.analyze_column(column_name, column_metadata)
            
            print(f"Suggested {len(rule_recommendations)} rules for column: {column_name}")
            return rule_recommendations
            
        except Exception as e:
            print(f"Error suggesting rules for column {column_name}: {str(e)}")
            return []
    
    def suggest_rules_for_table(self, table_metadata: Dict[str, Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Suggest DQ rules for all columns in a table.
        
        Args:
            table_metadata: Dictionary mapping column names to column metadata
            
        Returns:
            Dictionary mapping column names to lists of recommended DQ rules
        """
        print(f"Suggesting rules for {len(table_metadata)} columns")
        
        rule_recommendations = {}
        column_count = 0
        total_columns = len(table_metadata)
        
        for column_name, column_metadata in table_metadata.items():
            column_count += 1
            print(f"Processing column {column_count}/{total_columns}: {column_name}")
            rule_recommendations[column_name] = self.suggest_rules(column_metadata)
        
        total_rules = sum(len(rules) for rules in rule_recommendations.values())
        print(f"Completed rule suggestions: {total_rules} rules for {len(rule_recommendations)} columns")
        
        return rule_recommendations