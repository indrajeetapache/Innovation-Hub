"""
Common data types and enumerations for the Smart DQ Rule System.

This module contains shared data types used across different components of the 
Smart DQ Rule System. By centralizing these definitions, we eliminate circular
import dependencies between modules.

The main components defined here are:
1. RuleCategory - Enumeration of different categories of data quality rules
2. PIIType - Enumeration of different types of personally identifiable information

These are used by both the rule_engines module (for defining rules) and the
ml_models module (for processing model outputs related to these types).
"""

from enum import Enum




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
    """Enum defining types of personally identifiable information (PII)."""
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


# Additional helper functions related to these types can be added here
def get_pii_severity(pii_type: PIIType) -> int:
    """
    Get the default severity level for a PII type.
    
    Args:
        pii_type: The PII type to get the severity for
        
    Returns:
        Severity level (1-5, with 5 being most severe)
    """
    high_severity = {PIIType.SSN, PIIType.CREDIT_CARD, PIIType.PASSWORD}
    medium_severity = {PIIType.DOB, PIIType.EMAIL, PIIType.PHONE, PIIType.ADDRESS}
    
    if pii_type in high_severity:
        return 5
    elif pii_type in medium_severity:
        return 4
    else:
        return 3

print("Initialized common data types module")