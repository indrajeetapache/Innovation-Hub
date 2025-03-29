import random
import uuid
from faker import Faker

# Initialize Faker
fake = Faker()

def generate_customer_data():
    """Generate a single synthetic customer record"""
    print("generating single synthetic customer record started")
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
    print("generating single synthetic customer record completed")
    return customer

def generate_account_data(customer_id):
    """Generate an account for a given customer"""
    print(f"Generate an account for a given customer started")
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
    print(f"Generate an account for a given customer completed")
    return account