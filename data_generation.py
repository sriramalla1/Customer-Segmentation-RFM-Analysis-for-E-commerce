"""
E-Commerce Synthetic Data Generator
Generates realistic transaction data for RFM analysis and customer segmentation

Author: Data Analytics Team
Date: November 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import argparse

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Configuration
NUM_CUSTOMERS = 25000
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2025, 11, 15)

# Product categories with realistic price ranges
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

# Customer segments with behavioral characteristics
CUSTOMER_SEGMENTS = {
    'Champions': {
        'transaction_prob': 0.15,
        'frequency_range': (15, 40),
        'value_mult': (1.5, 3.0),
        'recency_days': (1, 30)
    },
    'Loyal': {
        'transaction_prob': 0.12,
        'frequency_range': (8, 20),
        'value_mult': (1.2, 2.0),
        'recency_days': (1, 45)
    },
    'Potential Loyalists': {
        'transaction_prob': 0.08,
        'frequency_range': (4, 10),
        'value_mult': (0.8, 1.5),
        'recency_days': (1, 60)
    },
    'New Customers': {
        'transaction_prob': 0.06,
        'frequency_range': (1, 3),
        'value_mult': (0.7, 1.2),
        'recency_days': (1, 30)
    },
    'Promising': {
        'transaction_prob': 0.07,
        'frequency_range': (2, 5),
        'value_mult': (0.9, 1.4),
        'recency_days': (1, 90)
    },
    'Need Attention': {
        'transaction_prob': 0.05,
        'frequency_range': (3, 8),
        'value_mult': (0.8, 1.3),
        'recency_days': (60, 150)
    },
    'At Risk': {
        'transaction_prob': 0.03,
        'frequency_range': (5, 15),
        'value_mult': (1.0, 2.0),
        'recency_days': (120, 250)
    },
    'Cant Lose Them': {
        'transaction_prob': 0.02,
        'frequency_range': (10, 30),
        'value_mult': (1.5, 2.5),
        'recency_days': (90, 180)
    },
    'Hibernating': {
        'transaction_prob': 0.02,
        'frequency_range': (1, 5),
        'value_mult': (0.6, 1.0),
        'recency_days': (180, 400)
    },
    'Lost': {
        'transaction_prob': 0.01,
        'frequency_range': (1, 8),
        'value_mult': (0.7, 1.2),
        'recency_days': (300, 600)
    }
}


def generate_transactions(num_customers=NUM_CUSTOMERS, 
                          start_date=START_DATE, 
                          end_date=END_DATE,
                          output_file='ecommerce_transactions.csv',
                          verbose=True):
    """
    Generate synthetic e-commerce transaction data
    
    Args:
        num_customers (int): Number of unique customers to generate
        start_date (datetime): Earliest transaction date
        end_date (datetime): Latest transaction date
        output_file (str): Output CSV filename
        verbose (bool): Print progress information
        
    Returns:
        pd.DataFrame: Generated transaction data
    """
    
    if verbose:
        print("="*70)
        print("E-COMMERCE SYNTHETIC DATA GENERATOR")
        print("="*70)
        print(f"\nConfiguration:")
        print(f"  Customers: {num_customers:,}")
        print(f"  Date Range: {start_date.date()} to {end_date.date()}")
        print(f"  Product Categories: {len(PRODUCT_CATEGORIES)}")
        print(f"  Customer Segments: {len(CUSTOMER_SEGMENTS)}")
    
    # Assign customers to segments
    segment_weights = [v['transaction_prob'] for v in CUSTOMER_SEGMENTS.values()]
    customer_segments = random.choices(
        list(CUSTOMER_SEGMENTS.keys()), 
        weights=segment_weights, 
        k=num_customers
    )
    
    if verbose:
        print("\nCustomer Distribution by Segment:")
        for segment in CUSTOMER_SEGMENTS.keys():
            count = customer_segments.count(segment)
            pct = count / num_customers * 100
            print(f"  {segment:20s}: {count:5d} ({pct:5.1f}%)")
    
    # Generate transactions
    transactions = []
    transaction_id = 1
    
    if verbose:
        print(f"\nGenerating transactions...")
    
    for customer_id in range(1, num_customers + 1):
        segment = customer_segments[customer_id - 1]
        segment_config = CUSTOMER_SEGMENTS[segment]
        
        # Determine number of transactions
        num_transactions = random.randint(*segment_config['frequency_range'])
        
        # Determine recency (last purchase date)
        recency_days = random.randint(*segment_config['recency_days'])
        last_purchase_date = end_date - timedelta(days=recency_days)
        
        # Generate transaction dates
        if num_transactions == 1:
            transaction_dates = [last_purchase_date]
        else:
            days_between = (last_purchase_date - start_date).days
            if days_between <= 0:
                transaction_dates = [last_purchase_date]
            else:
                # Generate dates with realistic clustering
                transaction_dates = sorted([
                    start_date + timedelta(days=random.randint(0, days_between))
                    for _ in range(num_transactions - 1)
                ])
                transaction_dates.append(last_purchase_date)
        
        # Generate transactions
        for trans_date in transaction_dates:
            # Select product category
            category = random.choice(list(PRODUCT_CATEGORIES.keys()))
            price_range = PRODUCT_CATEGORIES[category]
            
            # Calculate purchase amount
            base_price = random.uniform(*price_range)
            value_mult = random.uniform(*segment_config['value_mult'])
            purchase_amount = round(base_price * value_mult, 2)
            
            # Quantity (occasional bulk purchases)
            quantity = random.choices(
                [1, 2, 3, 4, 5], 
                weights=[70, 15, 8, 5, 2]
            )[0]
            
            total_amount = round(purchase_amount * quantity, 2)
            
            transactions.append({
                'transaction_id': transaction_id,
                'customer_id': f'CUST{customer_id:05d}',
                'transaction_date': trans_date,
                'product_category': category,
                'quantity': quantity,
                'unit_price': purchase_amount,
                'total_amount': total_amount,
                'true_segment': segment
            })
            
            transaction_id += 1
        
        # Progress indicator
        if verbose and customer_id % 5000 == 0:
            print(f"  Processed {customer_id:,} customers...")
    
    # Create DataFrame
    df = pd.DataFrame(transactions)
    
    if verbose:
        print(f"\n✓ Generated {len(df):,} transactions")
        print(f"\n" + "="*70)
        print("DATASET SUMMARY")
        print("="*70)
        print(f"\nTotal Customers: {df['customer_id'].nunique():,}")
        print(f"Total Transactions: {len(df):,}")
        print(f"Date Range: {df['transaction_date'].min().date()} to {df['transaction_date'].max().date()}")
        print(f"Total Revenue: ${df['total_amount'].sum():,.2f}")
        print(f"\nAverage Order Value: ${df['total_amount'].mean():.2f}")
        print(f"Median Order Value: ${df['total_amount'].median():.2f}")
        print(f"\nTransactions per Customer:")
        trans_per_cust = df.groupby('customer_id').size()
        print(f"  Mean: {trans_per_cust.mean():.1f}")
        print(f"  Min: {trans_per_cust.min()}")
        print(f"  Max: {trans_per_cust.max()}")
        
        print(f"\nProduct Category Distribution:")
        category_dist = df['product_category'].value_counts()
        for cat, count in category_dist.items():
            pct = count / len(df) * 100
            print(f"  {cat:20s}: {count:6d} ({pct:5.1f}%)")
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    if verbose:
        print(f"\n✓ Data saved to '{output_file}'")
        print("="*70)
    
    return df


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Generate synthetic e-commerce transaction data for RFM analysis'
    )
    parser.add_argument(
        '--customers',
        type=int,
        default=NUM_CUSTOMERS,
        help=f'Number of customers to generate (default: {NUM_CUSTOMERS})'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='ecommerce_transactions.csv',
        help='Output CSV filename (default: ecommerce_transactions.csv)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output'
    )
    
    args = parser.parse_args()
    
    # Generate data
    df = generate_transactions(
        num_customers=args.customers,
        output_file=args.output,
        verbose=not args.quiet
    )
    
    return df


if __name__ == "__main__":
    main()
