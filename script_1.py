
# Now let's create the complete RFM Analysis
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("RFM ANALYSIS & CUSTOMER SEGMENTATION")
print("="*70)

# Load the data
df = pd.read_csv('ecommerce_transactions.csv')
df['transaction_date'] = pd.to_datetime(df['transaction_date'])

# Set analysis date (day after last transaction)
ANALYSIS_DATE = df['transaction_date'].max() + timedelta(days=1)
print(f"\nAnalysis Date: {ANALYSIS_DATE.date()}")

# Calculate RFM Metrics
print("\n" + "="*70)
print("STEP 1: CALCULATING RFM METRICS")
print("="*70)

rfm_df = df.groupby('customer_id').agg({
    'transaction_date': lambda x: (ANALYSIS_DATE - x.max()).days,  # Recency
    'transaction_id': 'count',  # Frequency
    'total_amount': 'sum'  # Monetary
}).reset_index()

rfm_df.columns = ['customer_id', 'recency', 'frequency', 'monetary']

print(f"\nRFM Metrics calculated for {len(rfm_df)} customers")
print("\nRFM Statistics:")
print(rfm_df[['recency', 'frequency', 'monetary']].describe())

# Calculate RFM Scores (1-5 scale using quintiles)
print("\n" + "="*70)
print("STEP 2: CALCULATING RFM SCORES (1-5 Scale)")
print("="*70)

# For Recency: Lower is better, so reverse the score
rfm_df['r_score'] = pd.qcut(rfm_df['recency'], q=5, labels=[5, 4, 3, 2, 1], duplicates='drop')
# For Frequency and Monetary: Higher is better
rfm_df['f_score'] = pd.qcut(rfm_df['frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
rfm_df['m_score'] = pd.qcut(rfm_df['monetary'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')

# Convert to numeric
rfm_df['r_score'] = rfm_df['r_score'].astype(int)
rfm_df['f_score'] = rfm_df['f_score'].astype(int)
rfm_df['m_score'] = rfm_df['m_score'].astype(int)

# Calculate RFM Score (combined)
rfm_df['rfm_score'] = rfm_df['r_score'].astype(str) + rfm_df['f_score'].astype(str) + rfm_df['m_score'].astype(str)
rfm_df['rfm_score_numeric'] = rfm_df['r_score'] + rfm_df['f_score'] + rfm_df['m_score']

print("\nRFM Scores calculated!")
print("\nSample RFM Scores:")
print(rfm_df[['customer_id', 'recency', 'frequency', 'monetary', 'r_score', 'f_score', 'm_score', 'rfm_score']].head(10))

# K-Means Clustering
print("\n" + "="*70)
print("STEP 3: K-MEANS CLUSTERING")
print("="*70)

# Prepare data for clustering
X = rfm_df[['recency', 'frequency', 'monetary']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find optimal number of clusters using elbow method
print("\nFinding optimal number of clusters...")
inertias = []
silhouette_scores = []
K_range = range(3, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

print("\nCluster Evaluation Metrics:")
for k, inertia, sil_score in zip(K_range, inertias, silhouette_scores):
    print(f"  K={k}: Inertia={inertia:.2f}, Silhouette Score={sil_score:.4f}")

# Use 8 clusters (based on research and silhouette score)
OPTIMAL_CLUSTERS = 8
print(f"\n✓ Selected {OPTIMAL_CLUSTERS} clusters for segmentation")

# Perform final clustering
kmeans_final = KMeans(n_clusters=OPTIMAL_CLUSTERS, random_state=42, n_init=10)
rfm_df['cluster'] = kmeans_final.fit_predict(X_scaled)

print(f"\nClustering complete!")
print(f"Silhouette Score: {silhouette_score(X_scaled, rfm_df['cluster']):.4f}")

# Analyze clusters
print("\n" + "="*70)
print("STEP 4: CLUSTER ANALYSIS")
print("="*70)

cluster_summary = rfm_df.groupby('cluster').agg({
    'recency': 'mean',
    'frequency': 'mean',
    'monetary': 'mean',
    'customer_id': 'count'
}).round(2)

cluster_summary.columns = ['avg_recency', 'avg_frequency', 'avg_monetary', 'customer_count']
cluster_summary['pct_customers'] = (cluster_summary['customer_count'] / len(rfm_df) * 100).round(2)
cluster_summary['total_revenue'] = rfm_df.groupby('cluster')['monetary'].sum().round(2)
cluster_summary['pct_revenue'] = (cluster_summary['total_revenue'] / rfm_df['monetary'].sum() * 100).round(2)

print("\nCluster Summary:")
print(cluster_summary)

# Assign segment names based on RFM characteristics
def assign_segment_name(row):
    """Assign strategic segment names based on cluster characteristics"""
    r, f, m = row['avg_recency'], row['avg_frequency'], row['avg_monetary']
    
    # Calculate relative scores (compared to overall means)
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

# Create segment mapping
segment_mapping = {}
for cluster_id in cluster_summary.index:
    cluster_data = cluster_summary.loc[cluster_id]
    segment_name = assign_segment_name(cluster_data)
    segment_mapping[cluster_id] = segment_name

rfm_df['segment'] = rfm_df['cluster'].map(segment_mapping)

print("\n" + "="*70)
print("STEP 5: SEGMENT NAMING & PROFILING")
print("="*70)

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

print("\nSegment Profiles:")
print(segment_summary)

# Calculate key insights
top_segments = segment_summary.head(3)
top_segment_customers = top_segments['pct_customers'].sum()
top_segment_revenue = top_segments['pct_revenue'].sum()

print("\n" + "="*70)
print("KEY INSIGHTS")
print("="*70)
print(f"\n📊 Top 3 segments represent {top_segment_customers:.1f}% of customers")
print(f"💰 These segments contribute {top_segment_revenue:.1f}% of total revenue")
print(f"\n🏆 Highest value segment: {segment_summary.index[0]}")
print(f"   - {segment_summary.loc[segment_summary.index[0], 'customer_count']} customers ({segment_summary.loc[segment_summary.index[0], 'pct_customers']:.1f}%)")
print(f"   - ${segment_summary.loc[segment_summary.index[0], 'total_revenue']:,.0f} revenue ({segment_summary.loc[segment_summary.index[0], 'pct_revenue']:.1f}%)")

# Save results
rfm_df.to_csv('rfm_analysis_results.csv', index=False)
segment_summary.to_csv('segment_summary.csv')
cluster_summary.to_csv('cluster_summary.csv')

print("\n" + "="*70)
print("FILES SAVED")
print("="*70)
print("✓ rfm_analysis_results.csv - Complete RFM analysis with segments")
print("✓ segment_summary.csv - Segment-level statistics")
print("✓ cluster_summary.csv - Cluster-level statistics")
print("\n✓ RFM Analysis Complete!")
