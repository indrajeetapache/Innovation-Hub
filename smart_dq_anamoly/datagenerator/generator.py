import pandas as pd
import numpy as np
import time
from datetime import timedelta

def generate_wealth_management_data(start_date, end_date, num_customers=200, 
                                  num_accounts_per_customer=3, daily_records=1000,
                                  randomness_factor=0.1):
   start_time = time.time()
   print("✅ Step 1: Initializing data generation process...")

   start = pd.to_datetime(start_date)
   end = pd.to_datetime(end_date)
   dates = pd.date_range(start=start, end=end, freq='D')
   print(f"🗓️  Generating data for {len(dates)} days → {start_date} to {end_date}")

   print("\n✅ Step 2: Creating customer profiles...")
   customer_ids = [f'CUST{i:06d}' for i in range(1, num_customers + 1)]
   print(f"👤 Total customers: {len(customer_ids)}")

   print("\n✅ Step 3: Creating account structures...")
   accounts = []
   for cust_id in customer_ids:
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
   print(f"🏦 Created {len(account_ids)} accounts for {len(customer_ids)} customers")

   print("\n✅ Step 4: Generating daily transaction and account data...")
   data = []
   total_records = 0
   target_records = len(dates) * daily_records
   print(f"📊 Expected total records: {target_records:,}")

   for idx, process_date in enumerate(dates, start=1):
       if idx % 5 == 0 or idx == 1:
           print(f"🔄 Day {idx}/{len(dates)} → {process_date.date()}")

       # Base market condition for the day
       base_market_condition = np.random.normal(0, 1)
       
       daily_account_sample = np.random.choice(account_ids, size=daily_records, replace=True)

       for account_id in daily_account_sample:
           try:
               account_info = account_df[account_df['account_id'] == account_id].iloc[0]

               # Add individual randomness to each transaction's market condition
               market_condition = base_market_condition + np.random.normal(0, randomness_factor)
               
               base_balance = np.random.lognormal(11, 1)
               
               # Add randomness to trend factor
               base_trend_factor = 1 + ((process_date - start).days / 365) * 0.05
               trend_factor = base_trend_factor * (1 + np.random.normal(0, randomness_factor/3))
               
               # Add randomness to type factor
               base_type_factor = {'SAVINGS': 0.8, 'CHECKING': 1.0, 'INVESTMENT': 1.2, 'RETIREMENT': 0.9}[account_info['account_type']]
               type_factor = base_type_factor * (1 + np.random.normal(0, randomness_factor/2))
               
               # Add randomness to risk factor
               base_risk_factor = {'LOW': 0.7, 'MEDIUM': 1.0, 'HIGH': 1.3}[account_info['risk_profile']]
               risk_factor = base_risk_factor * (1 + np.random.normal(0, randomness_factor/2))
               
               balance = base_balance * trend_factor * type_factor * (1 + 0.1 * market_condition * risk_factor) * np.random.uniform(0.95, 1.05)

               record = {
                   'process_date': process_date,
                   'customer_id': account_info['customer_id'],
                   'account_id': account_id,
                   'account_type': account_info['account_type'],
                   'risk_profile': account_info['risk_profile'],
                   'relationship_manager': account_info['relationship_manager'],
                   'balance': round(balance, 2),
                   'market_condition': round(market_condition, 4),
                   'trend_factor': round(trend_factor, 4),
                   'risk_adjusted_factor': round(risk_factor, 2),
                   'account_type_factor': round(type_factor, 2),
                   'base_balance': round(base_balance, 2)
               }

               data.append(record)
               total_records += 1

               if total_records % 100000 == 0:
                   elapsed = time.time() - start_time
                   rate = total_records / elapsed
                   print(f"⚡ {total_records:,} records generated ({(total_records / target_records) * 100:.1f}%)")
                   print(f"🚀 Speed: {rate:.2f} rec/sec | ⏱️ Est time left: {((target_records - total_records) / rate) / 60:.2f} min")

           except Exception as e:
               print(f"❌ Skipped record due to error: {e} | Account ID: {account_id}")

   print("\n✅ Step 5: Finalizing and converting to DataFrame...")
   df = pd.DataFrame(data)

   print("\n📦 Summary:")
   print(f"🧾 Total records: {len(df):,}")
   print(f"📐 Total columns: {len(df.columns)}")
   print(f"💾 Memory used: {df.memory_usage(deep=True).sum() / (1024 ** 2):.2f} MB")
   print(f"⏳ Total time: {time.time() - start_time:.2f} sec")
   print("Data load completed")
   return df