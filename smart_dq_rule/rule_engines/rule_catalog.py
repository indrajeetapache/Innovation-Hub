class RuleCatalog:
    """Repository of data quality rules."""
    
    def __init__(self):
        """Initialize the rule catalog."""
        self.rules = {}
        logger.info("Initializing rule catalog with default rules")
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize the catalog with default rules."""
        # Add completeness rules
        logger.debug("Adding completeness rules to catalog")
        self.add_rule(CompletenessRule())
        
        # Add PII detection rules
        logger.debug("Adding PII detection rules to catalog")
        self._add_pii_rules()
        
        # Add uniqueness rules
        logger.debug("Adding uniqueness rules to catalog")
        self.add_rule(UniquenessRule())
    
    def _add_pii_rules(self):
        """Add standard PII detection rules to the catalog."""
        # Email rule
        logger.debug("Adding email detection rule")
        email_regex = r'^[\w\.-]+@[\w\.-]+\.\w+$'  # FIXED: Added closing quote and $ to end the pattern
        self.add_rule(PIIDetectionRule(
            pii_type=PIIType.EMAIL,
            regex_pattern=email_regex
        ))
        
        # Phone number rule
        logger.debug("Adding phone detection rule")
        phone_regex = r'(\+\d{1,3}[\s-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}'
        self.add_rule(PIIDetectionRule(
            pii_type=PIIType.PHONE,
            regex_pattern=phone_regex
        ))
        
        # SSN rule
        logger.debug("Adding SSN detection rule")
        ssn_regex = r'^\d{3}-?\d{2}-?\d{4}$'
        self.add_rule(PIIDetectionRule(
            pii_type=PIIType.SSN,
            regex_pattern=ssn_regex,
            severity=5
        ))
        
        # Credit card rule
        logger.debug("Adding credit card detection rule")
        cc_regex = r'^\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}$'
        self.add_rule(PIIDetectionRule(
            pii_type=PIIType.CREDIT_CARD,
            regex_pattern=cc_regex,
            severity=5
        ))
        
        # Date of birth rule
        logger.debug("Adding DOB detection rule")
        dob_regex = r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$'
        self.add_rule(PIIDetectionRule(
            pii_type=PIIType.DOB,
            regex_pattern=dob_regex
        ))
        
        # IP address rule
        logger.debug("Adding IP address detection rule")
        ip_regex = r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$'
        self.add_rule(PIIDetectionRule(
            pii_type=PIIType.IP_ADDRESS,
            regex_pattern=ip_regex
        ))

        # Name detection - this is tricky with regex alone, 
        # might need ML-based approach for better detection
        logger.debug("Adding name detection rule")
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
        logger.debug(f"Adding rule '{rule.name}' of category '{rule.category.value}' to catalog")
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
            logger.warning(f"Rule '{rule_name}' not found in catalog")
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
        logger.info(f"Suggesting rules for column with profile: {column_profile.get('name', 'unnamed')}")
        logger.debug(f"Column profile: {column_profile}")
        logger.debug(f"ML confidence scores: {ml_confidence_scores}")
        
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
                    logger.warning(f"Invalid PII type: {pii_type_str}, skipping")
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


# Note: The MLRuleEngine class has been moved to ml_rule_engine.py to break the circular dependency