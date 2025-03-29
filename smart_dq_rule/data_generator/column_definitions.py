import json

# Define database and table information
DB_NAME = "smartdq_db"
TABLE_NAME = "smartdq_table"

def generate_column_definitions():
    """Generate detailed column definitions for financial data"""
    print("Column names generation started")
    columns = [
        # Customer Information - PII columns
        {
            "name": "customer_id",
            "description": "Unique identifier for the customer",
            "data_type": "STRING",
            "pii_status": "NON-PII", 
            "nullable": False
        },
        # ... all your column definitions ...
    ]
    print(f" Column names {columns}")
    return columns

def save_column_definitions(columns, filepath="column_definitions.json"):
    """Save column definitions to a JSON file"""
    print("save_column_definitions started")
    column_data = {
        "database": DB_NAME,
        "table": TABLE_NAME,
        "columns": columns
    }
    print(f"DB_NAME : {DB_NAME} TABLE_NAME :{TABLE_NAME}")
    with open(filepath, 'w') as f:
        json.dump(column_data, f, indent=2)
    
    print(f"Column definitions saved to {filepath}")
    return column_data