from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Union, Any
import re
import pandas as pd
import numpy as np
import logging




class RuleCategory(Enum):
    """Enum defining categories of data quality rules."""
    COMPLETENESS = "completeness"
    FORMAT = "format"
    CONSISTENCY = "consistency"
    ACCURACY = "accuracy"
    UNIQUENESS = "uniqueness"
    PRIVACY = "privacy"
    CUSTOM = "custom"


class PIIType(Enum):
    """Enum defining types of PII data."""
    NAME = "name"
    EMAIL = "email"
    PHONE = "phone"
    ADDRESS = "address"
    SSN = "ssn"
    DOB = "date_of_birth"
    IP_ADDRESS = "ip_address"
    CREDIT_CARD = "credit_card"
    ACCOUNT_NUMBER = "account_number"
    USERNAME = "username"
    PASSWORD = "password"
    OTHER = "other"
    NOT_PII = "not_pii"


class BaseRule(ABC):
    """Abstract base class for all data quality rules."""
    
    def __init__(self, 
                 name: str, 
                 description: str, 
                 category: RuleCategory, 
                 pii_type: Optional[PIIType] = None,
                 severity: int = 1,
                 parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize a base rule with basic metadata.
        
        Args:
            name: Unique name for the rule
            description: Human-readable description of what the rule checks
            category: Category of the rule
            pii_type: PII type this rule is relevant for (if applicable)
            severity: Importance of this rule (1-5, with 5 being most severe)
            parameters: Additional parameters for the rule
        """
        self.name = name
        self.description = description
        self.category = category
        self.pii_type = pii_type
        self.severity = min(max(1, severity), 5)  # Clamp between 1 and 5
        self.parameters = parameters or {}
        self.confidence = 0.0  # ML-determined confidence that this rule applies
    
    @abstractmethod
    def validate(self, series: pd.Series) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate a column against this rule.
        
        Args:
            series: The pandas Series to validate
            
        Returns:
            Tuple of (passed, details) where:
            - passed: Boolean indicating if the column passed the validation
            - details: Dictionary with additional details about the validation
        """
        pass
    
    def set_confidence(self, confidence: float) -> None:
        """Set the confidence score that this rule applies to a column."""
        self.confidence = max(0.0, min(1.0, confidence))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the rule to a dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "pii_type": self.pii_type.value if self.pii_type else None,
            "severity": self.severity,
            "parameters": self.parameters,
            "confidence": self.confidence
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseRule':
        """Create a rule instance from a dictionary."""
        # This would be implemented by each specific rule subclass
        raise NotImplementedError("Must be implemented by subclasses")


class CompletenessRule(BaseRule):
    """Rule to check for completeness of data (missing values)."""
    
    def __init__(self, 
                 name: str = "completeness_check",
                 description: str = "Checks for missing values in the column",
                 pii_type: Optional[PIIType] = None,
                 threshold: float = 0.95,
                 severity: int = 3):
        """
        Initialize a completeness rule.
        
        Args:
            name: Name of the rule
            description: Description of the rule
            pii_type: Type of PII this rule is for (if applicable)
            threshold: Minimum acceptable completeness ratio (0.0-1.0)
            severity: Severity of violations (1-5)
        """
        parameters = {"threshold": threshold}
        super().__init__(name, description, RuleCategory.COMPLETENESS, 
                         pii_type, severity, parameters)
    
    def validate(self, series: pd.Series) -> Tuple[bool, Dict[str, Any]]:
        """Validate completeness of a column."""
        non_null_count = series.count()
        total_count = len(series)
        completeness_ratio = non_null_count / total_count if total_count > 0 else 1.0
        
        threshold = self.parameters.get("threshold", 0.95)
        passed = completeness_ratio >= threshold
        
        details = {
            "non_null_count": non_null_count,
            "total_count": total_count,
            "completeness_ratio": completeness_ratio,
            "threshold": threshold
        }
        
        return passed, details


class FormatRule(BaseRule):
    """Rule to check the format of values using a regular expression."""
    
    def __init__(self, 
                 name: str, 
                 description: str,
                 regex_pattern: str,
                 pii_type: Optional[PIIType] = None,
                 match_type: str = "all",  # "all", "any", or threshold as float
                 severity: int = 3):
        """
        Initialize a format rule with a regex pattern.
        
        Args:
            name: Name of the rule
            description: Description of the rule
            regex_pattern: Regular expression pattern for validation
            pii_type: Type of PII this rule is for (if applicable)
            match_type: How to apply the pattern ("all", "any", or float threshold)
            severity: Severity of violations (1-5)
        """
        parameters = {
            "regex_pattern": regex_pattern,
            "match_type": match_type
        }
        super().__init__(name, description, RuleCategory.FORMAT, 
                         pii_type, severity, parameters)
        
        # Compile the regex for better performance
        self._regex = re.compile(regex_pattern)
    
    def validate(self, series: pd.Series) -> Tuple[bool, Dict[str, Any]]:
        """Validate the format of values in a column."""
        # Filter out null values
        non_null_series = series.dropna()
        
        # Apply regex to each value
        matches = non_null_series.astype(str).apply(
            lambda x: bool(self._regex.match(x))
        )
        
        match_count = matches.sum()
        total_count = len(non_null_series)
        match_ratio = match_count / total_count if total_count > 0 else 1.0
        
        # Determine if passed based on match_type
        match_type = self.parameters.get("match_type", "all")
        if match_type == "all":
            passed = match_ratio == 1.0
        elif match_type == "any":
            passed = match_ratio > 0.0
        else:
            # Treat as threshold
            try:
                threshold = float(match_type)
                passed = match_ratio >= threshold
            except (ValueError, TypeError):
                passed = False
        
        details = {
            "match_count": match_count,
            "total_count": total_count,
            "match_ratio": match_ratio,
            "match_type": match_type,
            "regex_pattern": self.parameters.get("regex_pattern")
        }
        
        return passed, details


class UniquenessRule(BaseRule):
    """Rule to check uniqueness of values in a column."""
    
    def __init__(self, 
                 name: str = "uniqueness_check",
                 description: str = "Checks for uniqueness of values",
                 pii_type: Optional[PIIType] = None,
                 threshold: float = 1.0,
                 severity: int = 3):
        """
        Initialize a uniqueness rule.
        
        Args:
            name: Name of the rule
            description: Description of the rule
            pii_type: Type of PII this rule is for
            threshold: Minimum acceptable uniqueness ratio (0.0-1.0)
            severity: Severity of violations (1-5)
        """
        parameters = {"threshold": threshold}
        super().__init__(name, description, RuleCategory.UNIQUENESS, 
                         pii_type, severity, parameters)
    
    def validate(self, series: pd.Series) -> Tuple[bool, Dict[str, Any]]:
        """Validate uniqueness of values in a column."""
        non_null_series = series.dropna()
        unique_count = non_null_series.nunique()
        total_count = len(non_null_series)
        
        uniqueness_ratio = unique_count / total_count if total_count > 0 else 1.0
        
        threshold = self.parameters.get("threshold", 1.0)
        passed = uniqueness_ratio >= threshold
        
        details = {
            "unique_count": unique_count,
            "total_count": total_count,
            "uniqueness_ratio": uniqueness_ratio,
            "threshold": threshold
        }
        
        return passed, details


class PIIDetectionRule(BaseRule):
    """Rule to detect a specific type of PII in a column."""
    
    def __init__(self, 
                 pii_type: PIIType,
                 name: Optional[str] = None,
                 description: Optional[str] = None,
                 regex_pattern: Optional[str] = None,
                 custom_validator: Optional[callable] = None,
                 threshold: float = 0.8,
                 severity: int = 4):
        """
        Initialize a PII detection rule.
        
        Args:
            pii_type: Type of PII to detect
            name: Name of the rule (default: based on pii_type)
            description: Description of the rule (default: based on pii_type)
            regex_pattern: Optional regex pattern for validation
            custom_validator: Optional custom validation function
            threshold: Threshold for detection (ratio of matching values)
            severity: Severity of violations (1-5)
        """
        # Default name and description based on PII type
        if name is None:
            name = f"{pii_type.value}_detection"
        if description is None:
            description = f"Detects {pii_type.value} information in the column"
        
        parameters = {
            "threshold": threshold
        }
        
        if regex_pattern:
            parameters["regex_pattern"] = regex_pattern
        
        super().__init__(name, description, RuleCategory.PRIVACY, 
                         pii_type, severity, parameters)
        
        self._regex = re.compile(regex_pattern) if regex_pattern else None
        self._custom_validator = custom_validator
    
    def validate(self, series: pd.Series) -> Tuple[bool, Dict[str, Any]]:
        """Validate if the column contains the specific PII type."""
        non_null_series = series.dropna()
        
        # Use regex if provided
        if self._regex:
            matches = non_null_series.astype(str).apply(
                lambda x: bool(self._regex.search(x))
            )
        # Use custom validator if provided
        elif self._custom_validator:
            matches = non_null_series.apply(self._custom_validator)
        # Default to no matches if no validation method
        else:
            matches = pd.Series([False] * len(non_null_series))
        
        match_count = matches.sum()
        total_count = len(non_null_series)
        match_ratio = match_count / total_count if total_count > 0 else 0.0
        
        threshold = self.parameters.get("threshold", 0.8)
        passed = match_ratio >= threshold
        
        details = {
            "match_count": match_count,
            "total_count": total_count,
            "match_ratio": match_ratio,
            "threshold": threshold,
            "pii_type": self.pii_type.value
        }
        
        return passed, details


class RuleCatalog:
    """Repository of data quality rules."""
    
    def __init__(self):
        """Initialize the rule catalog."""
        self.rules = {}
        print("Initializing rule catalog with default rules")
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize the catalog with default rules."""
        # Add completeness rules
        print("Adding completeness rules to catalog")
        self.add_rule(CompletenessRule())
        
        # Add PII detection rules
        print("Adding PII detection rules to catalog")
        self._add_pii_rules()
        
        # Add uniqueness rules
        print("Adding uniqueness rules to catalog")
        self.add_rule(UniquenessRule())
    
    def _add_pii_rules(self):
        """Add standard PII detection rules to the catalog."""
        # Email rule
        print("Adding email detection rule")
        email_regex = r'^[\w\.-]+@[\w\.-]+\.\w+
        
        # Phone number rule
        phone_regex = r'(\+\d{1,3}[\s-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}'
        self.add_rule(PIIDetectionRule(
            pii_type=PIIType.PHONE,
            regex_pattern=phone_regex
        ))
        
        # SSN rule
        ssn_regex = r'^\d{3}-?\d{2}-?\d{4}$'
        self.add_rule(PIIDetectionRule(
            pii_type=PIIType.SSN,
            regex_pattern=ssn_regex,
            severity=5
        ))
        
        # Credit card rule
        cc_regex = r'^\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}$'
        self.add_rule(PIIDetectionRule(
            pii_type=PIIType.CREDIT_CARD,
            regex_pattern=cc_regex,
            severity=5
        ))
        
        # Date of birth rule
        dob_regex = r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$'
        self.add_rule(PIIDetectionRule(
            pii_type=PIIType.DOB,
            regex_pattern=dob_regex
        ))
        
        # IP address rule
        ip_regex = r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$'
        self.add_rule(PIIDetectionRule(
            pii_type=PIIType.IP_ADDRESS,
            regex_pattern=ip_regex
        ))

        # Name detection - this is tricky with regex alone, 
        # might need ML-based approach for better detection
        name_rule = PIIDetectionRule(
            pii_type=PIIType.NAME,
            threshold=0.7  # Lower threshold since name detection is less precise
        )
        self.add_rule(name_rule)
    
    def add_rule(self, rule: BaseRule) -> None:
        """
        Add a rule to the catalog.
        
        Args:
            rule: The rule to add
        """
        print(f"Adding rule '{rule.name}' of category '{rule.category.value}' to catalog")
        self.rules[rule.name] = rule
    
    def get_rule(self, rule_name: str) -> Optional[BaseRule]:
        """
        Get a rule by name.
        
        Args:
            rule_name: Name of the rule to retrieve
            
        Returns:
            The rule if found, None otherwise
        """
        rule = self.rules.get(rule_name)
        if rule is None:
            print(f"Rule '{rule_name}' not found in catalog")
        return rule
    
    def get_rules_by_category(self, category: RuleCategory) -> List[BaseRule]:
        """
        Get all rules of a specific category.
        
        Args:
            category: The category to filter by
            
        Returns:
            List of rules in the specified category
        """
        return [r for r in self.rules.values() if r.category == category]
    
    def get_rules_by_pii_type(self, pii_type: PIIType) -> List[BaseRule]:
        """
        Get all rules related to a specific PII type.
        
        Args:
            pii_type: The PII type to filter by
            
        Returns:
            List of rules for the specified PII type
        """
        return [r for r in self.rules.values() if r.pii_type == pii_type]
    
    def suggest_rules_for_column(self, 
                                column_profile: Dict[str, Any],
                                ml_confidence_scores: Optional[Dict[str, float]] = None) -> List[BaseRule]:
        """
        Suggest appropriate rules for a column based on its profile.
        
        Args:
            column_profile: Profile of the column
            ml_confidence_scores: Optional confidence scores from ML model
            
        Returns:
            List of suggested rules sorted by confidence
        """
        print(f"Suggesting rules for column with profile: {column_profile.get('name', 'unnamed')}")
        print(f"Column profile: {column_profile}")
        print(f"ML confidence scores: {ml_confidence_scores}")
        
        suggested_rules = []
        
        # Add completeness rule for all columns
        completeness_rule = self.get_rule("completeness_check")
        if completeness_rule:
            suggested_rules.append(completeness_rule)
        
        # Check if column might contain PII based on ML model predictions
        if ml_confidence_scores:
            for pii_type_str, confidence in ml_confidence_scores.items():
                try:
                    pii_type = PIIType(pii_type_str)
                    pii_rules = self.get_rules_by_pii_type(pii_type)
                    
                    for rule in pii_rules:
                        rule_copy = self._copy_rule(rule)
                        rule_copy.set_confidence(confidence)
                        suggested_rules.append(rule_copy)
                except ValueError:
                    # Invalid PII type, skip
                    print(f"Invalid PII type: {pii_type_str}, skipping")
                    continue
        
        # Add uniqueness rule if the column has high cardinality
        unique_ratio = column_profile.get("unique_ratio", 0.0)
        if unique_ratio > 0.9:
            uniqueness_rule = self.get_rule("uniqueness_check")
            if uniqueness_rule:
                rule_copy = self._copy_rule(uniqueness_rule)
                rule_copy.set_confidence(min(1.0, unique_ratio))
                suggested_rules.append(rule_copy)
        
        # Sort rules by confidence (highest first)
        suggested_rules.sort(key=lambda r: r.confidence, reverse=True)
        return suggested_rules
    
    def _copy_rule(self, rule: BaseRule) -> BaseRule:
        """Create a copy of a rule to avoid modifying the original."""
        # This is a simplified copying mechanism - in a real implementation
        # you might want to use copy.deepcopy or implement a proper clone method
        rule_dict = rule.to_dict()
        
        # Create a new instance of the same class
        if isinstance(rule, CompletenessRule):
            return CompletenessRule(
                name=rule_dict["name"],
                description=rule_dict["description"],
                pii_type=PIIType(rule_dict["pii_type"]) if rule_dict["pii_type"] else None,
                threshold=rule_dict["parameters"].get("threshold", 0.95),
                severity=rule_dict["severity"]
            )
        elif isinstance(rule, UniquenessRule):
            return UniquenessRule(
                name=rule_dict["name"],
                description=rule_dict["description"],
                pii_type=PIIType(rule_dict["pii_type"]) if rule_dict["pii_type"] else None,
                threshold=rule_dict["parameters"].get("threshold", 1.0),
                severity=rule_dict["severity"]
            )
        elif isinstance(rule, PIIDetectionRule):
            return PIIDetectionRule(
                name=rule_dict["name"],
                description=rule_dict["description"],
                pii_type=PIIType(rule_dict["pii_type"]) if rule_dict["pii_type"] else PIIType.OTHER,
                regex_pattern=rule_dict["parameters"].get("regex_pattern"),
                threshold=rule_dict["parameters"].get("threshold", 0.8),
                severity=rule_dict["severity"]
            )
        elif isinstance(rule, FormatRule):
            return FormatRule(
                name=rule_dict["name"],
                description=rule_dict["description"],
                regex_pattern=rule_dict["parameters"].get("regex_pattern", ""),
                pii_type=PIIType(rule_dict["pii_type"]) if rule_dict["pii_type"] else None,
                match_type=rule_dict["parameters"].get("match_type", "all"),
                severity=rule_dict["severity"]
            )
        else:
            # Default fallback - this should be expanded for other rule types
            return rule


class MLRuleEngine:
    """Engine that uses ML models to suggest data quality rules."""
    
    def __init__(self, model, rule_catalog: RuleCatalog):
        """
        Initialize the ML rule engine.
        
        Args:
            model: ML model for PII detection
            rule_catalog: Catalog of available rules
        """
        self.model = model
        self.rule_catalog = rule_catalog
        print(f"Initialized ML Rule Engine with model: {getattr(model, 'name', type(model).__name__)}")

    
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
        print(f"Generating rule suggestions for column: {column_name}")
        
        # Create a prompt for the ML model
        print("Creating prompt for ML model")
        prompt = self._create_column_prompt(column_name, column_profile, column_sample)
        
        # Get model predictions
        print("Getting predictions from ML model")
        predictions = self._get_model_predictions(prompt)
        print(f"Received predictions for column {column_name}: {predictions}")
        
        # Convert predictions to rule suggestions
        print("Converting predictions to rule suggestions")
        rules = self.rule_catalog.suggest_rules_for_column(column_profile, predictions)
        print(f"Generated {len(rules)} rule suggestions for column {column_name}")
        
        return rules
    
    def _create_column_prompt(self, 
                             column_name: str,
                             column_profile: Dict[str, Any],
                             column_sample: pd.Series) -> str:
        """
        Create a prompt for the ML model to analyze a column.
        
        Args:
            column_name: Name of the column
            column_profile: Profile of the column
            column_sample: Sample data from the column
            
        Returns:
            Prompt string for the ML model
        """
        # Convert sample to string
        sample_str = "\n".join([str(x) for x in column_sample.head(10).tolist()])
        
        # Format statistics for the prompt
        stats = []
        for key, value in column_profile.items():
            if isinstance(value, (int, float, str, bool)):
                stats.append(f"- {key}: {value}")
        
        stats_str = "\n".join(stats)
        
        # Create the final prompt
        prompt = f"""
        Analyze the following column and determine if it contains personal identifiable information (PII).
        
        Column Name: {column_name}
        
        Column Statistics:
        {stats_str}
        
        Sample Values:
        {sample_str}
        
        For each PII type, provide a confidence score between 0 and 1 (0 means not PII, 1 means definitely PII).
        Return the results in the following format:
        
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
        """
        
        return prompt
    
    def _get_model_predictions(self, prompt: str) -> Dict[str, float]:
        """
        Get predictions from the ML model.
        
        Args:
            prompt: Prompt for the model
            
        Returns:
            Dictionary mapping PII types to confidence scores
        """
        try:
            # Call the model with the prompt
            result = self.model.generate(prompt)
            
            # Parse the response
            # Assuming the model returns a JSON string that can be parsed
            import json
            import re
            
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
            print(f"Error getting model predictions: {e}", exc_info=True)
            return {}  # Return empty dict if there was an error

        self.add_rule(PIIDetectionRule(
            pii_type=PIIType.EMAIL,
            regex_pattern=email_regex
        ))
        
        # Phone number rule
        phone_regex = r'(\+\d{1,3}[\s-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}'
        self.add_rule(PIIDetectionRule(
            pii_type=PIIType.PHONE,
            regex_pattern=phone_regex
        ))
        
        # SSN rule
        ssn_regex = r'^\d{3}-?\d{2}-?\d{4}$'
        self.add_rule(PIIDetectionRule(
            pii_type=PIIType.SSN,
            regex_pattern=ssn_regex,
            severity=5
        ))
        
        # Credit card rule
        cc_regex = r'^\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}$'
        self.add_rule(PIIDetectionRule(
            pii_type=PIIType.CREDIT_CARD,
            regex_pattern=cc_regex,
            severity=5
        ))
        
        # Date of birth rule
        dob_regex = r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$'
        self.add_rule(PIIDetectionRule(
            pii_type=PIIType.DOB,
            regex_pattern=dob_regex
        ))
        
        # IP address rule
        ip_regex = r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$'
        self.add_rule(PIIDetectionRule(
            pii_type=PIIType.IP_ADDRESS,
            regex_pattern=ip_regex
        ))

        # Name detection - this is tricky with regex alone, 
        # might need ML-based approach for better detection
        name_rule = PIIDetectionRule(
            pii_type=PIIType.NAME,
            threshold=0.7  # Lower threshold since name detection is less precise
        )
        self.add_rule(name_rule)
    
    def add_rule(self, rule: BaseRule) -> None:
        """
        Add a rule to the catalog.
        
        Args:
            rule: The rule to add
        """
        self.rules[rule.name] = rule
    
    def get_rule(self, rule_name: str) -> Optional[BaseRule]:
        """
        Get a rule by name.
        
        Args:
            rule_name: Name of the rule to retrieve
            
        Returns:
            The rule if found, None otherwise
        """
        return self.rules.get(rule_name)
    
    def get_rules_by_category(self, category: RuleCategory) -> List[BaseRule]:
        """
        Get all rules of a specific category.
        
        Args:
            category: The category to filter by
            
        Returns:
            List of rules in the specified category
        """
        return [r for r in self.rules.values() if r.category == category]
    
    def get_rules_by_pii_type(self, pii_type: PIIType) -> List[BaseRule]:
        """
        Get all rules related to a specific PII type.
        
        Args:
            pii_type: The PII type to filter by
            
        Returns:
            List of rules for the specified PII type
        """
        return [r for r in self.rules.values() if r.pii_type == pii_type]
    
    def suggest_rules_for_column(self, 
                                column_profile: Dict[str, Any],
                                ml_confidence_scores: Optional[Dict[str, float]] = None) -> List[BaseRule]:
        """
        Suggest appropriate rules for a column based on its profile.
        
        Args:
            column_profile: Profile of the column
            ml_confidence_scores: Optional confidence scores from ML model
            
        Returns:
            List of suggested rules sorted by confidence
        """
        suggested_rules = []
        
        # Add completeness rule for all columns
        completeness_rule = self.get_rule("completeness_check")
        if completeness_rule:
            suggested_rules.append(completeness_rule)
        
        # Check if column might contain PII based on ML model predictions
        if ml_confidence_scores:
            for pii_type_str, confidence in ml_confidence_scores.items():
                try:
                    pii_type = PIIType(pii_type_str)
                    pii_rules = self.get_rules_by_pii_type(pii_type)
                    
                    for rule in pii_rules:
                        rule_copy = self._copy_rule(rule)
                        rule_copy.set_confidence(confidence)
                        suggested_rules.append(rule_copy)
                except ValueError:
                    # Invalid PII type, skip
                    continue
        
        # Add uniqueness rule if the column has high cardinality
        unique_ratio = column_profile.get("unique_ratio", 0.0)
        if unique_ratio > 0.9:
            uniqueness_rule = self.get_rule("uniqueness_check")
            if uniqueness_rule:
                rule_copy = self._copy_rule(uniqueness_rule)
                rule_copy.set_confidence(min(1.0, unique_ratio))
                suggested_rules.append(rule_copy)
        
        # Sort rules by confidence (highest first)
        suggested_rules.sort(key=lambda r: r.confidence, reverse=True)
        return suggested_rules
    
    def _copy_rule(self, rule: BaseRule) -> BaseRule:
        """Create a copy of a rule to avoid modifying the original."""
        # This is a simplified copying mechanism - in a real implementation
        # you might want to use copy.deepcopy or implement a proper clone method
        rule_dict = rule.to_dict()
        
        # Create a new instance of the same class
        if isinstance(rule, CompletenessRule):
            return CompletenessRule(
                name=rule_dict["name"],
                description=rule_dict["description"],
                pii_type=PIIType(rule_dict["pii_type"]) if rule_dict["pii_type"] else None,
                threshold=rule_dict["parameters"].get("threshold", 0.95),
                severity=rule_dict["severity"]
            )
        elif isinstance(rule, UniquenessRule):
            return UniquenessRule(
                name=rule_dict["name"],
                description=rule_dict["description"],
                pii_type=PIIType(rule_dict["pii_type"]) if rule_dict["pii_type"] else None,
                threshold=rule_dict["parameters"].get("threshold", 1.0),
                severity=rule_dict["severity"]
            )
        elif isinstance(rule, PIIDetectionRule):
            return PIIDetectionRule(
                name=rule_dict["name"],
                description=rule_dict["description"],
                pii_type=PIIType(rule_dict["pii_type"]) if rule_dict["pii_type"] else PIIType.OTHER,
                regex_pattern=rule_dict["parameters"].get("regex_pattern"),
                threshold=rule_dict["parameters"].get("threshold", 0.8),
                severity=rule_dict["severity"]
            )
        elif isinstance(rule, FormatRule):
            return FormatRule(
                name=rule_dict["name"],
                description=rule_dict["description"],
                regex_pattern=rule_dict["parameters"].get("regex_pattern", ""),
                pii_type=PIIType(rule_dict["pii_type"]) if rule_dict["pii_type"] else None,
                match_type=rule_dict["parameters"].get("match_type", "all"),
                severity=rule_dict["severity"]
            )
        else:
            # Default fallback - this should be expanded for other rule types
            return rule


class MLRuleEngine:
    """Engine that uses ML models to suggest data quality rules."""
    
    def __init__(self, model, rule_catalog: RuleCatalog):
        """
        Initialize the ML rule engine.
        
        Args:
            model: ML model for PII detection
            rule_catalog: Catalog of available rules
        """
        self.model = model
        self.rule_catalog = rule_catalog
    
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
        # Create a prompt for the ML model
        prompt = self._create_column_prompt(column_name, column_profile, column_sample)
        
        # Get model predictions
        predictions = self._get_model_predictions(prompt)
        
        # Convert predictions to rule suggestions
        return self.rule_catalog.suggest_rules_for_column(column_profile, predictions)
    
    def _create_column_prompt(self, 
                             column_name: str,
                             column_profile: Dict[str, Any],
                             column_sample: pd.Series) -> str:
        """
        Create a prompt for the ML model to analyze a column.
        
        Args:
            column_name: Name of the column
            column_profile: Profile of the column
            column_sample: Sample data from the column
            
        Returns:
            Prompt string for the ML model
        """
        # Convert sample to string
        sample_str = "\n".join([str(x) for x in column_sample.head(10).tolist()])
        
        # Format statistics for the prompt
        stats = []
        for key, value in column_profile.items():
            if isinstance(value, (int, float, str, bool)):
                stats.append(f"- {key}: {value}")
        
        stats_str = "\n".join(stats)
        
        # Create the final prompt
        prompt = f"""
        Analyze the following column and determine if it contains personal identifiable information (PII).
        
        Column Name: {column_name}
        
        Column Statistics:
        {stats_str}
        
        Sample Values:
        {sample_str}
        
        For each PII type, provide a confidence score between 0 and 1 (0 means not PII, 1 means definitely PII).
        Return the results in the following format:
        
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
        """
        
        return prompt
    
    def _get_model_predictions(self, prompt: str) -> Dict[str, float]:
        """
        Get predictions from the ML model.
        
        Args:
            prompt: Prompt for the model
            
        Returns:
            Dictionary mapping PII types to confidence scores
        """
        try:
            # Call the model with the prompt
            result = self.model.generate(prompt)
            
            # Parse the response
            # Assuming the model returns a JSON string that can be parsed
            import json
            import re
            
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
            print(f"Error getting model predictions: {e}")
            return {}  # Return empty dict if there was an error