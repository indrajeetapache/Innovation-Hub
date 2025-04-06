## How It Works

### 1. Column Profiling

The system first extracts comprehensive features from each column, including:

- Basic statistics (count, null ratio, unique count, min/max values)
- Data type information
- Pattern detection (emails, phone numbers, dates, etc.)
- Distribution characteristics (mean, std dev, percentiles)
- Text metrics (length distribution, whitespace analysis)

```python ```
# Extract features for all columns
features_by_column = extract_dataframe_features(df)


### 2. Rule Determination
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

```python ```
# Determine applicable rules for all columns
rules_by_column = {}
for column_name, column_features in features_by_column.items():
    rules_by_column[column_name] = determine_applicable_rules(column_features)
