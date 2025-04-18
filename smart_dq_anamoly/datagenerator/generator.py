import pandas as pd
import numpy as np
import time
from datetime import timedelta

def generate_wealth_management_data(start_date, end_date, num_customers=200, 
                                   num_accounts_per_customer=3, daily_records=1000):
    """
    Generate synthetic wealth management data with realistic patterns
    
    Parameters:
    -----------
    start_date : str
        Start date in format 'YYYY-MM-DD'
    end_date : str
        End date in format 'YYYY-MM-DD'
    num_customers : int
        Number of unique customers to generate
    num_accounts_per_customer : int
        Average number of accounts per customer
    daily_records : int
        Number of records to generate per day
        
    Returns:
    --------
    DataFrame with synthetic wealth management data
    """
    start_time = time.time()
    print("Step 1: Initializing data generation process...")
    
    # Convert dates to datetime objects
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Generate date range
    dates = pd.date_range(start=start, end=end, freq='D')
    print(f"Generating data for {len(dates)} days from {start_date} to {end_date}")
    
    print("\nStep 2: Creating customer profiles...")
    # Create customer IDs
    customer_ids = [f'CUST{i:06d}' for i in range(1, num_customers + 1)]
    
    print("\nStep 3: Creating account structures...")
    # Create account IDs for each customer (varying number of accounts per customer)
    accounts = []
    for cust_id in customer_ids:
        # Randomly vary the number of accounts per customer
        num_accounts = max(1, int(np.random.normal(num_accounts_per_customer, 1)))
        for j in range(num_accounts):
            accounts.append({
                'customer_id': cust_id,
                'account_id': f'{cust_id}_ACC{j:03d}',
                'account_type': np.random.choice(['SAVINGS', 'CHECKING', 'INVESTMENT', 'RETIREMENT']),
                'open_date': start - timedelta(days=np.random.randint(1, 1000)),
                'risk_profile': np.random.choice(['LOW', 'MEDIUM', 'HIGH']),
                'relationship_manager': f'RM{np.random.randint(1, 50):03d}'
            })
    
    account_df = pd.DataFrame(accounts)
    account_ids = account_df['account_id'].tolist()
    
    print(f"Generated {len(customer_ids)} customers with {len(account_ids)} accounts")

    # Asset classes and sectors for investment details
    asset_classes = ['EQUITY', 'FIXED_INCOME', 'CASH', 'ALTERNATIVES', 'REAL_ESTATE']
    sectors = ['TECHNOLOGY', 'HEALTHCARE', 'FINANCE', 'CONSUMER', 'INDUSTRIAL', 'ENERGY', 'UTILITIES', 'MATERIALS']
    
    print("\nStep 4: Generating daily transaction and account data...")
    # Initialize empty list to store records
    data = []
    
    # Track how many records have been generated
    total_records = 0
    target_records = len(dates) * daily_records
    
    # Generate daily records
    for process_date in dates:
        # Select random accounts for this day
        daily_account_sample = np.random.choice(account_ids, size=daily_records, replace=True)
        
        # Market conditions vary by date (simulating market fluctuations)
        market_condition = np.random.normal(0, 1)  # Random market factor for the day
        
        for account_id in daily_account_sample:
            # Get account info
            account_info = account_df[account_df['account_id'] == account_id].iloc[0]
            
            # Base values that are consistent for this account
            base_balance = np.random.lognormal(11, 1)  # Generates values mostly between 10K-1M
            base_contribution = base_balance * np.random.uniform(0.001, 0.05)
            base_withdrawal = base_balance * np.random.uniform(0, 0.03)
            
            # Adjustments based on time (seasonal patterns and trends)
            time_factor = 1.0
            # Add yearly growth trend
            days_since_start = (process_date - start).days
            trend_factor = 1 + (days_since_start / 365) * 0.05  # 5% annual growth
            
            # Add quarterly pattern (higher activity at quarter end)
            month = process_date.month
            day = process_date.day
            if month in [3, 6, 9, 12] and day > 20:
                time_factor *= 1.2  # 20% more activity at quarter end
            
            # Add year-end effect
            if month == 12 and day > 15:
                time_factor *= 1.3  # 30% more activity at year end
            
            # Adjustments based on account type
            type_factor = {
                'SAVINGS': 0.8,
                'CHECKING': 1.0,
                'INVESTMENT': 1.2,
                'RETIREMENT': 0.9
            }[account_info['account_type']]
            
            # Adjustments based on risk profile
            risk_factor = {
                'LOW': 0.7,
                'MEDIUM': 1.0,
                'HIGH': 1.3
            }[account_info['risk_profile']]
            
            # Calculate final values with some randomness
            balance = base_balance * trend_factor * type_factor * (1 + 0.1 * market_condition * risk_factor) * np.random.uniform(0.95, 1.05)
            
            # Sometimes introduce outliers (about 1% of records)
            if np.random.random() < 0.01:
                if np.random.random() < 0.5:  # 50% of outliers are high values
                    balance *= np.random.uniform(3, 10)
                else:  # 50% of outliers are low values
                    balance *= np.random.uniform(0.1, 0.4)
            
            # Calculate other metrics based on balance
            contributions = base_contribution * time_factor * np.random.uniform(0.7, 1.3) if np.random.random() < 0.3 else 0
            withdrawals = base_withdrawal * time_factor * np.random.uniform(0.7, 1.3) if np.random.random() < 0.2 else 0
            
            # Market values fluctuate with the market condition
            market_value = balance * (1 + 0.05 * market_condition) * np.random.uniform(0.98, 1.02)
            
            # Portfolio allocation - varies by risk profile
            if account_info['account_type'] in ['INVESTMENT', 'RETIREMENT']:
                if account_info['risk_profile'] == 'LOW':
                    equity_pct = np.random.uniform(20, 40)
                    fixed_income_pct = np.random.uniform(40, 60)
                elif account_info['risk_profile'] == 'MEDIUM':
                    equity_pct = np.random.uniform(40, 60)
                    fixed_income_pct = np.random.uniform(20, 40)
                else:  # HIGH risk
                    equity_pct = np.random.uniform(60, 80)
                    fixed_income_pct = np.random.uniform(10, 30)
                
                alternatives_pct = np.random.uniform(5, 15)
                real_estate_pct = np.random.uniform(0, 10)
                cash_pct = 100 - equity_pct - fixed_income_pct - alternatives_pct - real_estate_pct
            else:
                equity_pct = 0
                fixed_income_pct = 0
                alternatives_pct = 0
                real_estate_pct = 0
                cash_pct = 100
            
            # Transaction counts
            buy_transactions = np.random.poisson(3 * risk_factor * time_factor)
            sell_transactions = np.random.poisson(2 * risk_factor * time_factor)
            
            # Performance metrics
            ytd_return = (np.random.normal(0.07, 0.12) * risk_factor) * (process_date.month / 12)
            mtd_return = np.random.normal(0.005, 0.02) * risk_factor * market_condition
            
            # Fee structure
            management_fee_rate = np.random.uniform(0.005, 0.015)
            transaction_fee_rate = np.random.uniform(0.001, 0.005)
            management_fee = balance * management_fee_rate / 12  # Monthly fee
            transaction_fees = (buy_transactions + sell_transactions) * transaction_fee_rate * balance * 0.01
            
            # Credit scores and risk metrics (for some account types)
            if account_info['account_type'] in ['SAVINGS', 'CHECKING']:
                credit_score = np.random.randint(300, 850)
                delinquency_flag = np.random.random() < 0.05  # 5% chance of delinquency
                default_probability = np.random.beta(1, 20) if credit_score < 650 else np.random.beta(1, 100)
            else:
                credit_score = None
                delinquency_flag = False
                default_probability = 0
            
            # Digital engagement metrics
            login_count = np.random.poisson(5)
            mobile_app_use = np.random.random() < 0.7  # 70% chance of mobile app use
            notification_count = np.random.poisson(3) if mobile_app_use else 0
            
            # Security and compliance
            kyc_status = np.random.choice(['COMPLETE', 'PENDING', 'REVIEW'], p=[0.94, 0.05, 0.01])
            aml_flag = np.random.random() < 0.02  # 2% chance of AML flag
            fraud_flag = np.random.random() < 0.01  # 1% chance of fraud flag
            
            # Create record
            record = {
                'process_date': process_date,
                'customer_id': account_info['customer_id'],
                'account_id': account_id,
                'account_type': account_info['account_type'],
                'risk_profile': account_info['risk_profile'],
                'relationship_manager': account_info['relationship_manager'],
                'balance': round(balance, 2),
                'market_value': round(market_value, 2),
                'cash_value': round(market_value * cash_pct / 100, 2),
                'available_credit': round(balance * 0.2, 2) if account_info['account_type'] == 'CHECKING' else 0,
                'contributions': round(contributions, 2),
                'withdrawals': round(withdrawals, 2),
                'equity_pct': round(equity_pct, 2),
                'fixed_income_pct': round(fixed_income_pct, 2),
                'cash_pct': round(cash_pct, 2),
                'alternatives_pct': round(alternatives_pct, 2),
                'real_estate_pct': round(real_estate_pct, 2),
                'buy_transactions': buy_transactions,
                'sell_transactions': sell_transactions,
                'ytd_return': round(ytd_return, 4),
                'mtd_return': round(mtd_return, 4),
                'management_fee': round(management_fee, 2),
                'transaction_fees': round(transaction_fees, 2),
                'credit_score': credit_score,
                'delinquency_flag': delinquency_flag,
                'default_probability': round(default_probability, 4),
                'login_count': login_count,
                'mobile_app_use': mobile_app_use,
                'notification_count': notification_count,
                'kyc_status': kyc_status,
                'aml_flag': aml_flag,
                'fraud_flag': fraud_flag,
                'primary_asset_class': np.random.choice(asset_classes),
                'primary_sector': np.random.choice(sectors),
                'last_advisor_contact_days': np.random.randint(1, 90),
                'customer_satisfaction': np.random.randint(1, 11),
                'upsell_opportunity_score': np.random.uniform(0, 1),
                'churn_risk_score': np.random.uniform(0, 1),
                'lifetime_value': round(balance * np.random.uniform(1.2, 3.5), 2),
                'household_id': f'HH{int(account_info["customer_id"][4:]) % 100:03d}',
                'household_income_band': np.random.choice(['LOW', 'MEDIUM', 'HIGH', 'AFFLUENT', 'WEALTH']),
                'relationship_tenure_years': np.random.randint(1, 20),
                'age_band': np.random.choice(['18-25', '26-35', '36-45', '46-55', '56-65', '65+']),
                'occupation': np.random.choice(['PROFESSIONAL', 'BUSINESS_OWNER', 'RETIRED', 'STUDENT', 'OTHER']),
                'region': np.random.choice(['NORTH', 'SOUTH', 'EAST', 'WEST', 'CENTRAL']),
                'urban_rural': np.random.choice(['URBAN', 'SUBURBAN', 'RURAL']),
                'education_level': np.random.choice(['HIGH_SCHOOL', 'COLLEGE', 'GRADUATE', 'PHD']),
                'foreign_national': np.random.random() < 0.1,
                'politically_exposed': np.random.random() < 0.05,
                'tax_status': np.random.choice(['STANDARD', 'EXEMPT', 'FOREIGN']),
                'channel_preference': np.random.choice(['DIGITAL', 'BRANCH', 'PHONE', 'ADVISOR']),
                'social_media_engagement': np.random.choice([0, 1, 2, 3, 4, 5]),
                'next_best_product': np.random.choice(['CREDIT_CARD', 'MORTGAGE', 'INVESTMENT', 'INSURANCE', 'NONE']),
                'days_since_last_transaction': np.random.randint(0, 30)
            }
            
            data.append(record)
            total_records += 1
            
            # Print progress every 100,000 records
            if total_records % 100000 == 0:
                elapsed_time = time.time() - start_time
                records_per_second = total_records / elapsed_time
                print(f"Generated {total_records:,} records ({round(total_records/target_records*100, 1)}% complete)")
                print(f"Processing speed: {records_per_second:.2f} records/second")
                print(f"Estimated time remaining: {((target_records - total_records) / records_per_second) / 60:.2f} minutes")
    
    print("\nStep 5: Finalizing dataset...")
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Calculate and print statistics
    elapsed_time = time.time() - start_time
    print(f"Data generation complete in {elapsed_time:.2f} seconds")
    print(f"Total records: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
    
    return df
