"""Common data types for the Smart DQ Rule System."""
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
    """Enum defining types of PII data."""
    NAME = "name"
    EMAIL = "email"
    PHONE = "phone"
    # ... rest of your enum values