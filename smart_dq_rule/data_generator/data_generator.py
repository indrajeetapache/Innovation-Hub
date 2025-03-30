import os
import pandas as pd
from datetime import datetime, timedelta
import random
import uuid
from faker import Faker

# Initialize Faker for generating realistic data
fake = Faker()

def generate_customer_data():
    """Generate a single synthetic customer record"""
    customer = {
        "customer_id": str(uuid.uuid4())[:8],
        "first_name": fake.first_name(),
        "last_name": fake.last_name(),
        "email": fake.email(),
        "phone_number": fake.phone_number(),
        "date_of_birth": fake.date_of_birth(minimum_age=18, maximum_age=90).strftime("%Y-%m-%d"),
        "ssn": f"xxx-xx-{random.randint(1000, 9999)}",
        "address_line1": fake.street_address(),
        "address_line2": fake.secondary_address() if random.random() > 0.7 else None,
        "city": fake.city(),
        "state": fake.state_abbr(),
        "zip_code": fake.zipcode(),
        "country": "US",
        "customer_segment": random.choice(["Premium", "Standard", "Basic"]),
        "loyalty_tier": random.choice(["Gold", "Silver", "Bronze", None]),
        "marketing_campaign_id": f"CAMP{random.randint(1000, 9999)}" if random.random() > 0.3 else None,
        "opt_in_marketing": random.choice(["Y", "N"]),
        "kyc_status": random.choice(["Verified", "Pending", "Not Verified"]),
    }
    return customer

def generate_account_data(customer_id):
    """Generate an account for a given customer"""
    account_type = random.choice(["Checking", "Savings", "Credit", "Investment"])
    account_status = random.choice(["Active", "Inactive", "Closed", "Suspended"])
    
    # Generate dates
    opening_date = fake.date_between(start_date="-5y", end_date="-1m")
    closing_date = None
    if account_status == "Closed":
        closing_date = fake.date_between(start_date=opening_date, end_date="today")
    
    account = {
        "account_id": f"ACC{random.randint(10000000, 99999999)}",
        "customer_id": customer_id,
        "account_type": account_type,
        "account_status": account_status,
        "opening_date": opening_date.strftime("%Y-%m-%d"),
        "closing_date": closing_date.strftime("%Y-%m-%d") if closing_date else None,
        "account_balance": round(random.uniform(0, 50000), 2),
        "interest_rate": round(random.uniform(0.01, 0.1), 4) if account_type in ["Savings", "Investment"] else None,
    }
    
    # Add credit-specific fields
    if account_type == "Credit":
        credit_limit = round(random.choice([1000, 2500, 5000, 10000, 25000]), 2)
        account["credit_limit"] = credit_limit
        account["available_credit"] = round(credit_limit - account["account_balance"], 2)
        account["minimum_payment"] = round(account["account_balance"] * 0.02, 2)
    
    return account

def generate_transaction_data(account_id, account_type, start_date, end_date, num_transactions):
    """Generate transaction data for an account"""
    print(f"  Generating {num_transactions} transactions for account {account_id} ({account_type})")
    
    transactions = []
    
    transaction_types = {
        "Checking": ["Deposit", "Withdrawal", "Transfer", "Fee", "Payment"],
        "Savings": ["Deposit", "Withdrawal", "Transfer", "Interest", "Fee"],
        "Credit": ["Purchase", "Payment", "Fee", "Interest", "Cash Advance"],
        "Investment": ["Deposit", "Withdrawal", "Dividend", "Fee", "Trade"]
    }
    
    merchant_categories = ["Retail", "Food", "Travel", "Transportation", "Utilities", 
                           "Entertainment", "Healthcare", "Education", "Technology"]
    
    # Generate transactions
    for i in range(num_transactions):
        if i > 0 and i % 100 == 0:
            print(f"    Generated {i}/{num_transactions} transactions")
            
        transaction_date = fake.date_between(start_date=start_date, end_date=end_date)
        tx_type = random.choice(transaction_types.get(account_type, transaction_types["Checking"]))
        
        # Amount based on transaction type
        if tx_type in ["Deposit", "Payment", "Interest", "Dividend"]:
            amount = round(random.uniform(10, 5000), 2)
        else:
            amount = round(-random.uniform(1, 2000), 2)
        
        # Merchant data
        merchant_name = None
        merchant_category = None
        merchant_location = None
        if tx_type in ["Purchase", "Payment"]:
            merchant_name = fake.company()
            merchant_category = random.choice(merchant_categories)
            merchant_location = f"{fake.city()}, {fake.state_abbr()}"
        
        # Risk data
        risk_score = random.randint(0, 100) if random.random() > 0.8 else None
        fraud_flag = "Y" if risk_score and risk_score > 85 else "N"
        
        # Technical data
        ip_address = fake.ipv4() if random.random() > 0.3 else None
        device_id = f"DEV-{uuid.uuid4().hex[:8]}" if random.random() > 0.4 else None
        session_id = uuid.uuid4().hex if random.random() > 0.4 else None
        user_agent = fake.user_agent() if random.random() > 0.6 else None
        channel = random.choice(["Web", "Mobile", "Branch", "ATM", "Phone"])
        
        transaction = {
            "transaction_id": f"TX-{uuid.uuid4().hex[:10]}",
            "account_id": account_id,
            "transaction_date": transaction_date.strftime("%Y-%m-%d"),
            "process_date": transaction_date.strftime("%Y%m%d"),
            "transaction_time": fake.time(),
            "transaction_type": tx_type,
            "amount": amount,
            "currency": "USD",
            "merchant_name": merchant_name,
            "merchant_category": merchant_category,
            "merchant_location": merchant_location,
            "transaction_status": "Completed" if random.random() > 0.05 else random.choice(["Pending", "Failed", "Disputed"]),
            "reference_number": f"REF-{random.randint(1000000, 9999999)}",
            "risk_score": risk_score,
            "fraud_flag": fraud_flag,
            "anti_money_laundering_check": random.choice(["Passed", "Failed", "Flagged", None]),
            "ip_address": ip_address,
            "device_id": device_id,
            "session_id": session_id,
            "user_agent": user_agent,
            "channel": channel,
            "created_by": "SYSTEM",
            "created_at": transaction_date.strftime("%Y-%m-%d %H:%M:%S"),
            "updated_by": None,
            "updated_at": None,
            "record_status": "Active"
        }
        
        transactions.append(transaction)
    
    print(f"  ✓ Completed generating {num_transactions} transactions for account {account_id}")
    return transactions

def generate_complete_dataset(num_customers=1000, months=6, output_dir="/content/sample_data"):
    """
    Generate a complete financial dataset with customers, accounts, and transactions
    
    Args:
        num_customers: Number of customers to generate
        months: Months of historical data to generate
        output_dir: Directory to save the generated files
    
    Returns:
        Dictionary containing all the generated dataframes
    """
    print("\n" + "="*80)
    print(f"GENERATING FINANCIAL DATASET")
    print(f"Customers: {num_customers} | Historical months: {months} | Output: {output_dir}")
    print("="*80 + "\n")
    
    start_time = datetime.now()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Creating output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Define date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30 * months)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Lists for data
    all_customers = []
    all_accounts = []
    all_transactions = []
    
    # Generate data
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] PHASE 1: Generating customer and account data")
    print(f"Generating data for {num_customers} customers...")
    
    customer_progress_step = max(1, num_customers // 20)  # Show progress at 5% intervals
    
    for i in range(num_customers):
        if i % customer_progress_step == 0 or i == num_customers - 1:
            progress = (i / num_customers) * 100
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Progress: {progress:.1f}% - Processing customer {i+1}/{num_customers}")
        
        # Generate customer
        customer = generate_customer_data()
        all_customers.append(customer)
        
        # Generate 1-3 accounts per customer
        num_accounts = random.randint(1, 3)
        for acc_idx in range(num_accounts):
            account = generate_account_data(customer["customer_id"])
            all_accounts.append(account)
            
            # Generate transactions only for active accounts
            if account["account_status"] != "Closed":
                # Number of transactions depends on account activity
                monthly_tx_count = random.randint(10, 100)
                total_tx_count = monthly_tx_count * months
                
                # For large datasets, we may want to cap this
                total_tx_count = min(total_tx_count, 500)  # Cap at 500 transactions per account
                
                transactions = generate_transaction_data(
                    account["account_id"], 
                    account["account_type"],
                    start_date, 
                    end_date,
                    total_tx_count
                )
                all_transactions.extend(transactions)
    
    # Convert to dataframes
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] PHASE 2: Converting to dataframes")
    print(f"Converting {len(all_customers)} customers, {len(all_accounts)} accounts, and {len(all_transactions)} transactions to dataframes")
    
    df_customers = pd.DataFrame(all_customers)
    df_accounts = pd.DataFrame(all_accounts)
    df_transactions = pd.DataFrame(all_transactions)
    
    # Save to CSV files
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] PHASE 3: Saving individual tables to CSV")
    
    customer_file = os.path.join(output_dir, "customers.csv")
    account_file = os.path.join(output_dir, "accounts.csv")
    transaction_file = os.path.join(output_dir, "transactions.csv")
    
    print(f"Saving customers to {customer_file}")
    df_customers.to_csv(customer_file, index=False)
    
    print(f"Saving accounts to {account_file}")
    df_accounts.to_csv(account_file, index=False)
    
    print(f"Saving transactions to {transaction_file}")
    df_transactions.to_csv(transaction_file, index=False)
    
    # Create denormalized table
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] PHASE 4: Creating denormalized table")
    print("Merging accounts with customers...")
    
    # Merge accounts with customers
    df_combined = pd.merge(
        df_accounts,
        df_customers,
        on="customer_id",
        how="left",
        suffixes=("", "_customer")
    )
    
    print("Merging transactions with combined data...")
    # Merge transactions
    df_final = pd.merge(
        df_transactions,
        df_combined,
        on="account_id",
        how="left",
        suffixes=("", "_account")
    )
    
    # Save the combined data
    combined_file = os.path.join(output_dir, "smartdq_table.csv")
    print(f"Saving denormalized data to {combined_file}")
    df_final.to_csv(combined_file, index=False)
    
    # Calculate execution time
    end_time = datetime.now()
    execution_time = end_time - start_time
    hours, remainder = divmod(execution_time.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Print summary statistics
    print("\n" + "="*80)
    print(f"DATASET GENERATION COMPLETE")
    print(f"Execution time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print("="*80)
    
    print(f"\nSUMMARY:")
    print(f"✓ Total customers: {len(df_customers)}")
    print(f"✓ Total accounts: {len(df_accounts)}")
    print(f"✓ Total transactions: {len(df_transactions)}")
    
    # Calculate file sizes
    customer_size_mb = os.path.getsize(customer_file) / (1024 * 1024)
    account_size_mb = os.path.getsize(account_file) / (1024 * 1024)
    transaction_size_mb = os.path.getsize(transaction_file) / (1024 * 1024)
    combined_size_mb = os.path.getsize(combined_file) / (1024 * 1024)
    total_size_mb = customer_size_mb + account_size_mb + transaction_size_mb + combined_size_mb
    
    print(f"\nFILE SIZES:")
    print(f"✓ Customer data: {customer_size_mb:.2f} MB")
    print(f"✓ Account data: {account_size_mb:.2f} MB")
    print(f"✓ Transaction data: {transaction_size_mb:.2f} MB")
    print(f"✓ Denormalized data: {combined_size_mb:.2f} MB")
    print(f"✓ Total size: {total_size_mb:.2f} MB ({total_size_mb/1024:.2f} GB)")
    
    print(f"\nFILES CREATED:")
    print(f"✓ Customer data: {customer_file}")
    print(f"✓ Account data: {account_file}")
    print(f"✓ Transaction data: {transaction_file}")
    print(f"✓ Denormalized data: {combined_file}")
    
    return {
        "customers": df_customers,
        "accounts": df_accounts,
        "transactions": df_transactions,
        "combined": df_final
    }

def generate_sample_dataset(output_dir="/content/sample_data"):
    """Generate a small sample dataset for testing"""
    print("Generating sample dataset (100 customers, 3 months of data)")
    return generate_complete_dataset(num_customers=100, months=3, output_dir=output_dir)