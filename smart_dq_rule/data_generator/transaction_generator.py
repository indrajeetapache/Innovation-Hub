import random
import uuid
from faker import Faker
from datetime import datetime

# Initialize Faker
fake = Faker()

def generate_transaction_data(account_id, account_type, start_date, end_date, num_transactions):
    print(f" Generate transaction data for an account ")
    """Generate transaction data for an account"""
    transactions = []    
    transaction_types = {
        "Checking": ["Deposit", "Withdrawal", "Transfer", "Fee", "Payment"],
        "Savings": ["Deposit", "Withdrawal", "Transfer", "Interest", "Fee"],
        "Credit": ["Purchase", "Payment", "Fee", "Interest", "Cash Advance"],
        "Investment": ["Deposit", "Withdrawal", "Dividend", "Fee", "Trade"]
    }
    print(f"transaction_types : {transaction_types}")
    merchant_categories = ["Retail", "Food", "Travel", "Transportation", "Utilities", 
                           "Entertainment", "Healthcare", "Education", "Technology"]
    print(f"merchant_categories : {merchant_categories}")
    # Generate transactions
    for _ in range(num_transactions):
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
        print(f"transaction : {transaction}")
        transactions.append(transaction)
    
    return transactions