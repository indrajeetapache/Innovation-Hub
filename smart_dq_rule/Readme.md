## How It Works

### 1. Column Profiling

The system first extracts comprehensive features from each column, including:

- Basic statistics (count, null ratio, unique count, min/max values)
- Data type information
- Pattern detection (emails, phone numbers, dates, etc.)
- Distribution characteristics (mean, std dev, percentiles)
- Text metrics (length distribution, whitespace analysis)

```python ```
Extract features for all columns
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
 Determine applicable rules for all columns
rules_by_column = {}
for column_name, column_features in features_by_column.items():
    rules_by_column[column_name] = determine_applicable_rules(column_features)

### 3. Statistical Confidence Calculation
The system assigns confidence scores (0.0–1.0) to each rule using statistical methods:

Completeness: Logistic function based on null ratio
Uniqueness: Z-score comparison to industry standard thresholds
Range Checks: Empirical rule (68–95–99.7) for normal distributions
Format Rules: Match ratio against expected patterns
String Consistency: Coefficient of variation for length and patterns

### 4. Machine Learning Enhancement
XGBoost models are trained to predict rule applicability based on column features:

Learns from historical rule assignments
Detects patterns missed by rule-based logic
Predicts rules for unseen columns


### 5. Integration and Validation
The system compares rule-based and ML-based recommendations:

Agreement statistics
Confidence-weighted rule recommendations
Feature importance insights
