# Then, use this specific import sequence:
from common.data_types import PIIType, RuleCategory

# Import BaseRule and other rule classes first
from rule_engines.rule_catalog import (
    BaseRule,  # Import this *explicitly*
    CompletenessRule, 
    FormatRule,
    UniquenessRule, 
    PIIDetectionRule
)

# Then import RuleCatalog separately
from rule_engines.rule_catalog import RuleCatalog