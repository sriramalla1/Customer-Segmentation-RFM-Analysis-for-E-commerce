
# First, let's create a comprehensive synthetic e-commerce dataset
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Configuration
NUM_CUSTOMERS = 25000
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2025, 11, 15)

print("Generating synthetic e-commerce transaction data...")
print(f"Number of customers: {NUM_CUSTOMERS}")
print(f"Date range: {START_DATE.date()} to {END_DATE.date()}")

# Product categories with different price ranges
PRODUCT_CATEGORIES = {
    'Electronics': (50, 2000),
    'Fashion': (20, 300),
    'Home & Garden': (15, 500),
    'Beauty & Health': (10, 150),
    'Sports & Outdoors': (25, 400),
    'Books & Media': (5, 100),
    'Toys & Games': (10, 200),
    'Food & Beverages': (5, 80)
}

# Customer segments with different behaviors
CUSTOMER_SEGMENTS = {
    'Champions': {'transaction_prob': 0.15, 'frequency_range': (15, 40), 'value_mult': (1.5, 3.0), 'recency_days': (1, 30)},
    'Loyal': {'transaction_prob': 0.12, 'frequency_range': (8, 20), 'value_mult': (1.2, 2.0), 'recency_days': (1, 45)},
    'Potential Loyalists': {'transaction_prob': 0.08, 'frequency_range': (4, 10), 'value_mult': (0.8, 1.5), 'recency_days': (1, 60)},
    'New Customers': {'transaction_prob': 0.06, 'frequency_range': (1, 3), 'value_mult': (0.7, 1.2), 'recency_days': (1, 30)},
    'Promising': {'transaction_prob': 0.07, 'frequency_range': (2, 5), 'value_mult': (0.9, 1.4), 'recency_days': (1, 90)},
    'Need Attention': {'transaction_prob': 0.05, 'frequency_range': (3, 8), 'value_mult': (0.8, 1.3), 'recency_days': (60, 150)},
    'At Risk': {'transaction_prob': 0.03, 'frequency_range': (5, 15), 'value_mult': (1.0, 2.0), 'recency_days': (120, 250)},
    'Cant Lose Them': {'transaction_prob': 0.02, 'frequency_range': (10, 30), 'value_mult': (1.5, 2.5), 'recency_days': (90, 180)},
    'Hibernating': {'transaction_prob': 0.02, 'frequency_range': (1, 5), 'value_mult': (0.6, 1.0), 'recency_days': (180, 400)},
    'Lost': {'transaction_prob': 0.01, 'frequency_range': (1, 8), 'value_mult': (0.7, 1.2), 'recency_days': (300, 600)}
}

# Assign customers to segments
segment_weights = [v['transaction_prob'] for v in CUSTOMER_SEGMENTS.values()]
customer_segments = random.choices(list(CUSTOMER_SEGMENTS.keys()), 
                                  weights=segment_weights, 
                                  k=NUM_CUSTOMERS)

print("\nCustomer distribution by segment:")
for segment in CUSTOMER_SEGMENTS.keys():
    count = customer_segments.count(segment)
    print(f"  {segment}: {count} ({count/NUM_CUSTOMERS*100:.1f}%)")

# Generate transactions
transactions = []
transaction_id = 1

for customer_id in range(1, NUM_CUSTOMERS + 1):
    segment = customer_segments[customer_id - 1]
    segment_config = CUSTOMER_SEGMENTS[segment]
    
    # Determine number of transactions for this customer
    num_transactions = random.randint(*segment_config['frequency_range'])
    
    # Determine recency (when was last purchase)
    recency_days = random.randint(*segment_config['recency_days'])
    last_purchase_date = END_DATE - timedelta(days=recency_days)
    
    # Generate transaction dates (working backwards from last purchase)
    if num_transactions == 1:
        transaction_dates = [last_purchase_date]
    else:
        # Spread transactions over time
        days_between = (last_purchase_date - START_DATE).days
        if days_between <= 0:
            transaction_dates = [last_purchase_date]
        else:
            # Generate dates with some clustering (more realistic)
            transaction_dates = sorted([
                START_DATE + timedelta(days=random.randint(0, days_between))
                for _ in range(num_transactions - 1)
            ])
            transaction_dates.append(last_purchase_date)
    
    # Generate transactions
    for trans_date in transaction_dates:
        # Select product category
        category = random.choice(list(PRODUCT_CATEGORIES.keys()))
        price_range = PRODUCT_CATEGORIES[category]
        
        # Calculate purchase amount with segment multiplier
        base_price = random.uniform(*price_range)
        value_mult = random.uniform(*segment_config['value_mult'])
        purchase_amount = round(base_price * value_mult, 2)
        
        # Quantity (occasional bulk purchases)
        quantity = random.choices([1, 2, 3, 4, 5], weights=[70, 15, 8, 5, 2])[0]
        total_amount = round(purchase_amount * quantity, 2)
        
        transactions.append({
            'transaction_id': transaction_id,
            'customer_id': f'CUST{customer_id:05d}',
            'transaction_date': trans_date,
            'product_category': category,
            'quantity': quantity,
            'unit_price': purchase_amount,
            'total_amount': total_amount,
            'true_segment': segment  # Hidden label for validation
        })
        
        transaction_id += 1

# Create DataFrame
df_transactions = pd.DataFrame(transactions)

print(f"\nGenerated {len(df_transactions)} transactions")
print(f"\nDataset shape: {df_transactions.shape}")
print(f"\nFirst few transactions:")
print(df_transactions.head(10))

# Save to CSV
df_transactions.to_csv('ecommerce_transactions.csv', index=False)
print("\n✓ Saved to 'ecommerce_transactions.csv'")

# Create summary statistics
print("\n" + "="*60)
print("DATASET SUMMARY STATISTICS")
print("="*60)
print(f"\nTotal Customers: {df_transactions['customer_id'].nunique():,}")
print(f"Total Transactions: {len(df_transactions):,}")
print(f"Date Range: {df_transactions['transaction_date'].min().date()} to {df_transactions['transaction_date'].max().date()}")
print(f"Total Revenue: ${df_transactions['total_amount'].sum():,.2f}")
print(f"\nAverage Order Value: ${df_transactions['total_amount'].mean():.2f}")
print(f"Median Order Value: ${df_transactions['total_amount'].median():.2f}")
print(f"\nTransactions per Customer:")
print(f"  Mean: {len(df_transactions) / df_transactions['customer_id'].nunique():.1f}")
print(f"  Min: {df_transactions.groupby('customer_id').size().min()}")
print(f"  Max: {df_transactions.groupby('customer_id').size().max()}")

print("\n✓ Synthetic dataset generation complete!")
