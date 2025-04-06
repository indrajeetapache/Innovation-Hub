# Smart Data Quality Rule System

## Overview

The **Smart Data Quality (Smart DQ) Rule System** is an intelligent framework that automatically analyzes data columns and applies appropriate data quality rules based on column characteristics. The system combines rule-based heuristics with machine learning to provide comprehensive data quality validation that adapts to your specific data patterns.

## Key Features

- **Automatic Rule Detection**: Intelligently determines appropriate data quality rules for each column  
- **Type-Specific Rules**: Specialized rules for strings, integers, floats, dates, and boolean columns  
- **Statistical Confidence Scoring**: Uses statistical methods to assign confidence scores to rule applicability  
- **Machine Learning Integration**: XGBoost models learn patterns to improve rule recommendations  
- **PII Detection**: Identifies potentially sensitive data requiring special handling  
- **Cross-Validation**: Ensures rule model performance through k-fold validation  

## How It Works

### 1. Column Profiling

The system first extracts comprehensive features from each column, including:

- Basic statistics (count, null ratio, unique count, min/max values)
- Data type information
- Pattern detection (emails, phone numbers, dates, etc.)
- Distribution characteristics (mean, std dev, percentiles)
- Text metrics (length distribution, whitespace analysis)

```python
# Extract features for all columns
features_by_column = extract_dataframe_features(df)
2. Rule Determination
For each column, the system analyzes features to determine which rules should apply. Rules are categorized by data type:

Common Rules (All Types)

Completeness
Uniqueness
Type validation
String Column Rules

Format validation (email, phone, date, etc.)
String length
Allowed values
Whitespace checks
Data type consistency
Integer Column Rules

Range checks
Min/Max value validation
Float Column Rules

Range checks
Precision validation
Average value checks
Standard deviation checks
Date Column Rules

Date range validation
Future date checks
Boolean Column Rules

Boolean value validation
Flag alerting
PII Detection

Email, phone, SSN, credit card detection
Name and address detection
# Determine applicable rules for all columns
rules_by_column = {}
for column_name, column_features in features_by_column.items():
    rules_by_column[column_name] = determine_applicable_rules(column_features)
3. Statistical Confidence Calculation
The system assigns confidence scores (0.0–1.0) to each rule using statistical methods:

Completeness: Logistic function based on null ratio
Uniqueness: Z-score comparison to industry standard thresholds
Range Checks: Empirical rule (68–95–99.7) for normal distributions
Format Rules: Match ratio against expected patterns
String Consistency: Coefficient of variation for length and patterns
def calculate_rule_confidence(rule_type, column_features):
    if rule_type == 'completeness':
        null_ratio = column_features.get('null_ratio', 0)
        k = 10
        confidence = 1.0 / (1.0 + math.exp(k * (null_ratio - 0.5)))
        return confidence

    elif rule_type == 'range_check' and 'mean' in column_features and 'std' in column_features:
        mean = column_features.get('mean')
        std = column_features.get('std')
        min_val = column_features.get('min')
        max_val = column_features.get('max')

        if min_val >= mean - 3 * std and max_val <= mean + 3 * std:
            return 0.95
4. Machine Learning Enhancement
XGBoost models are trained to predict rule applicability based on column features:

Learns from historical rule assignments
Detects patterns missed by rule-based logic
Predicts rules for unseen columns
# Train XGBoost models with cross-validation
models, numeric_features, feature_importances, cv_results = train_rule_prediction_models(
    features_by_column, rules_by_column, n_folds=5
)

# Predict using XGBoost
def predict_rules_with_xgboost(column_features, models, numeric_features):
    # Predict rules with trained models
    pass
5. Integration and Validation
The system compares rule-based and ML-based recommendations:

Agreement statistics
Confidence-weighted rule recommendations
Feature importance insights
# Compare rule-based and ML-based predictions
results = compare_all_rule_predictions(features_by_column, rules_by_column, models, numeric_features)
Required Data for Optimal Results

Column Characteristics
Include the following features per column:

Basic statistics: count, null_ratio, unique_count, unique_ratio
Type indicators: is_string, is_integer, is_float, is_datetime, is_boolean, is_categorical
Numeric stats: min, max, mean, median, std, percentiles
Text features: min_length, max_length, mean_length, length_std, whitespace_only_ratio
Pattern matches: email_match_ratio, phone_match_ratio
Date features: min_date, max_date, date_range_days
Historical Data
Historical rule assignments improve ML accuracy
9+ months of column data recommended
At least 30–50 columns for robust training
Future Enhancements

Planned Improvements
Advanced ML Techniques
Ensemble and deep learning models
Active learning integration
Expanded Rule Types
Cross-column and domain-specific rules
Time series validations
Performance Optimization
Parallel processing
Feature selection and incremental learning
User Interaction
Feedback loops and rule customizations
Integration Options
API support
CI/CD pipeline validation
Database connectivity
Technical Requirements

Python 3.6+
pandas, numpy, scikit-learn
xgboost>=3.0
Installation

pip install -r requirements.txt
Usage Example

# Import modules
from smart_dq.profilers.column_profiler import ColumnProfiler
from smart_dq.profilers.profile_manager import ProfileManager
from smart_dq.rule_engines.rule_catalog import RuleCatalog

# Initialize and profile
profiler = ColumnProfiler()
profile_manager = ProfileManager(profiler)

df = pd.read_csv('your_data.csv')
table_metadata = profile_manager.profile_table(df)

features_by_column = extract_dataframe_features(df)

rules_by_column = {}
for column_name, column_features in features_by_column.items():
    rules_by_column[column_name] = determine_applicable_rules(column_features)

models, numeric_features, feature_importances, cv_results = train_rule_prediction_models(
    features_by_column, rules_by_column, n_folds=5
)

analyze_feature_importance(feature_importances)

results = compare_all_rule_predictions(
    features_by_column, rules_by_column, models, numeric_features
)

with open('dq_rules.json', 'w') as f:
    json.dump(rules_by_column, f, indent=2, cls=CustomJSONEncoder)
