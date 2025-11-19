
# Reload and perform RFM analysis
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

print("Loading transaction data...")
df = pd.read_csv('ecommerce_transactions.csv')
df['transaction_date'] = pd.to_datetime(df['transaction_date'])

# Set analysis date
ANALYSIS_DATE = df['transaction_date'].max() + timedelta(days=1)

# Calculate RFM Metrics
rfm_df = df.groupby('customer_id').agg({
    'transaction_date': lambda x: (ANALYSIS_DATE - x.max()).days,
    'transaction_id': 'count',
    'total_amount': 'sum'
}).reset_index()

rfm_df.columns = ['customer_id', 'recency', 'frequency', 'monetary']

# Calculate RFM Scores
rfm_df['r_score'] = pd.qcut(rfm_df['recency'], q=5, labels=[5, 4, 3, 2, 1], duplicates='drop')
rfm_df['f_score'] = pd.qcut(rfm_df['frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
rfm_df['m_score'] = pd.qcut(rfm_df['monetary'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')

rfm_df['r_score'] = rfm_df['r_score'].astype(int)
rfm_df['f_score'] = rfm_df['f_score'].astype(int)
rfm_df['m_score'] = rfm_df['m_score'].astype(int)

rfm_df['rfm_score'] = rfm_df['r_score'].astype(str) + rfm_df['f_score'].astype(str) + rfm_df['m_score'].astype(str)
rfm_df['rfm_score_numeric'] = rfm_df['r_score'] + rfm_df['f_score'] + rfm_df['m_score']

# K-Means Clustering
X = rfm_df[['recency', 'frequency', 'monetary']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform clustering with 8 clusters
kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
rfm_df['cluster'] = kmeans.fit_predict(X_scaled)

# Assign segment names
def assign_segment_name(row):
    r, f, m = row['avg_recency'], row['avg_frequency'], row['avg_monetary']
    r_mean, f_mean, m_mean = rfm_df['recency'].mean(), rfm_df['frequency'].mean(), rfm_df['monetary'].mean()
    
    if r < r_mean * 0.5 and f > f_mean * 1.5 and m > m_mean * 1.5:
        return 'Champions'
    elif r < r_mean and f > f_mean and m > m_mean:
        return 'Loyal Customers'
    elif r < r_mean * 0.7 and f < f_mean * 0.5:
        return 'New Customers'
    elif r < r_mean and f > f_mean * 0.8:
        return 'Potential Loyalists'
    elif r > r_mean * 1.5 and f > f_mean and m > m_mean:
        return 'At Risk'
    elif r > r_mean * 2 and f > f_mean * 1.2:
        return 'Cant Lose Them'
    elif r > r_mean * 1.5 and m < m_mean:
        return 'Hibernating'
    else:
        return 'Need Attention'

cluster_summary = rfm_df.groupby('cluster').agg({
    'recency': 'mean',
    'frequency': 'mean',
    'monetary': 'mean',
    'customer_id': 'count'
}).round(2)
cluster_summary.columns = ['avg_recency', 'avg_frequency', 'avg_monetary', 'customer_count']

segment_mapping = {}
for cluster_id in cluster_summary.index:
    segment_name = assign_segment_name(cluster_summary.loc[cluster_id])
    segment_mapping[cluster_id] = segment_name

rfm_df['segment'] = rfm_df['cluster'].map(segment_mapping)

# Save results
rfm_df.to_csv('rfm_analysis_results.csv', index=False)

segment_summary = rfm_df.groupby('segment').agg({
    'recency': 'mean',
    'frequency': 'mean',
    'monetary': 'mean',
    'customer_id': 'count'
}).round(2)
segment_summary.columns = ['avg_recency_days', 'avg_frequency', 'avg_monetary_value', 'customer_count']
segment_summary['pct_customers'] = (segment_summary['customer_count'] / len(rfm_df) * 100).round(2)
segment_summary['total_revenue'] = rfm_df.groupby('segment')['monetary'].sum().round(2)
segment_summary['pct_revenue'] = (segment_summary['total_revenue'] / rfm_df['monetary'].sum() * 100).round(2)
segment_summary = segment_summary.sort_values('pct_revenue', ascending=False)

segment_summary.to_csv('segment_summary.csv')

print("✓ Analysis complete!")
print(f"\nSegments identified:")
for seg in segment_summary.index:
    print(f"  - {seg}: {segment_summary.loc[seg, 'pct_customers']:.1f}% customers, {segment_summary.loc[seg, 'pct_revenue']:.1f}% revenue")
