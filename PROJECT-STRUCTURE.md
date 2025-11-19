# Project Structure & File Guide

## Complete File Listing

```
customer-segmentation-rfm/
│
├── README.md                          # Main project documentation
├── requirements.txt                   # Python dependencies
├── LICENSE                           # MIT License (add if needed)
│
├── data/
│   ├── ecommerce_transactions.csv    # Generated synthetic dataset (320K+ records)
│   ├── rfm_analysis_results.csv      # RFM scores per customer with segments
│   ├── segment_summary.csv           # Segment-level aggregated statistics
│   └── cluster_summary.csv           # Cluster-level analysis
│
├── src/
│   ├── data_generation.py            # Synthetic data generator script
│   └── rfm_analysis.py               # RFM analysis script (optional)
│
├── notebooks/
│   └── rfm_analysis.ipynb            # Complete analysis Jupyter notebook
│
├── docs/
│   ├── METHODOLOGY.md                # Technical methodology documentation
│   ├── marketing-strategies.md       # Detailed marketing strategies per segment
│   └── executive-summary.md          # Executive-level findings & recommendations
│
├── visualizations/
│   ├── segment_distribution.png      # Customer count by segment
│   ├── revenue_by_segment.png        # Revenue contribution chart
│   ├── rfm_heatmap.png              # RFM metrics heatmap
│   ├── elbow_method.png             # K-means elbow curve
│   ├── silhouette_analysis.png      # Silhouette scores
│   └── 3d_segmentation.png          # 3D scatter plot
│
└── presentation/
    ├── slides.pdf                    # Presentation deck (if created)
    └── demo_video.mp4               # Demo walkthrough (optional)
```

---

## File Descriptions

### Core Documentation

#### README.md
- **Purpose**: Main entry point for the project
- **Content**:
  - Project overview and objectives
  - Key findings (5 segments, revenue concentration)
  - Technologies used
  - Installation instructions
  - Quick start guide
  - Methodology summary
  - Marketing strategies overview
  - Skills demonstrated
  - Contact information
- **Audience**: Recruiters, hiring managers, technical reviewers

#### METHODOLOGY.md
- **Purpose**: Technical deep-dive into analysis approach
- **Content**:
  - Data generation process
  - RFM metric definitions and calculations
  - K-means clustering implementation
  - Feature standardization techniques
  - Cluster validation methods (elbow, silhouette)
  - Segment naming algorithm
  - Model evaluation metrics
  - Limitations and assumptions
  - Future enhancements
- **Audience**: Data scientists, technical interviewers

#### marketing-strategies.md
- **Purpose**: Business-focused marketing playbook
- **Content**:
  - Detailed strategy for each segment
  - Tactics and channel mix
  - Budget allocation recommendations
  - Expected KPIs and ROI
  - Campaign calendars
  - Implementation roadmap
- **Audience**: Marketing managers, business stakeholders

#### executive-summary.md
- **Purpose**: High-level business case
- **Content**:
  - Key findings at a glance
  - Revenue concentration analysis
  - Segment profiles
  - Strategic recommendations
  - Risk assessment
  - Success metrics
  - Next steps
- **Audience**: Executives, non-technical stakeholders

---

### Code Files

#### data_generation.py
- **Purpose**: Create synthetic e-commerce transaction data
- **Features**:
  - Configurable customer count
  - Realistic behavioral patterns (10 pre-defined segments)
  - Multiple product categories with price ranges
  - Time-based transaction distribution
  - Command-line interface
- **Usage**:
  ```bash
  python data_generation.py
  python data_generation.py --customers 50000 --output custom_data.csv
  ```
- **Output**: `ecommerce_transactions.csv`

#### rfm_analysis.py (Optional)
- **Purpose**: Standalone script version of notebook analysis
- **Features**:
  - RFM metric calculation
  - K-means clustering
  - Segment assignment
  - Summary statistics export
- **Usage**:
  ```bash
  python rfm_analysis.py --input ecommerce_transactions.csv
  ```

---

### Data Files

#### ecommerce_transactions.csv
- **Size**: ~320,000 rows
- **Columns**:
  - `transaction_id`: Unique transaction identifier
  - `customer_id`: Unique customer identifier (CUST00001 format)
  - `transaction_date`: Date of purchase (YYYY-MM-DD)
  - `product_category`: Product category (8 categories)
  - `quantity`: Items purchased (1-5)
  - `unit_price`: Price per unit
  - `total_amount`: Total transaction value
  - `true_segment`: Hidden ground truth segment (for validation)
- **Format**: CSV with headers
- **Sample**:
  ```
  transaction_id,customer_id,transaction_date,product_category,quantity,unit_price,total_amount,true_segment
  1,CUST00001,2025-10-27,Sports & Outdoors,1,331.56,331.56,New Customers
  ```

#### rfm_analysis_results.csv
- **Size**: ~25,000 rows (one per customer)
- **Columns**:
  - `customer_id`: Customer identifier
  - `recency`: Days since last purchase
  - `frequency`: Total number of purchases
  - `monetary`: Total revenue from customer
  - `r_score`: Recency score (1-5)
  - `f_score`: Frequency score (1-5)
  - `m_score`: Monetary score (1-5)
  - `rfm_score`: Combined score string (e.g., "555")
  - `rfm_score_numeric`: Numeric sum (3-15)
  - `cluster`: K-means cluster assignment (0-7)
  - `segment`: Named segment (Champions, Loyal, etc.)
- **Use Cases**:
  - Customer-level targeting
  - Export to CRM/marketing automation
  - Personalization engine input

#### segment_summary.csv
- **Size**: 5-8 rows (one per segment)
- **Columns**:
  - `segment`: Segment name
  - `customer_count`: Number of customers
  - `pct_customers`: Percentage of customer base
  - `total_revenue`: Total revenue from segment
  - `pct_revenue`: Percentage of total revenue
  - `avg_recency_days`: Average days since last purchase
  - `avg_frequency`: Average purchase count
  - `avg_monetary_value`: Average customer lifetime value
- **Use Cases**:
  - Executive reporting
  - Budget allocation decisions
  - Strategy prioritization

#### cluster_summary.csv
- **Size**: 8 rows (one per cluster)
- **Columns**:
  - `cluster`: Cluster number (0-7)
  - `avg_recency`: Mean recency
  - `avg_frequency`: Mean frequency
  - `avg_monetary`: Mean monetary
  - `customer_count`: Customers in cluster
  - `pct_customers`: Percentage of base
  - `total_revenue`: Cluster revenue
  - `pct_revenue`: Revenue percentage
- **Use Cases**:
  - Model validation
  - Technical analysis
  - Cluster quality assessment

---

### Notebook

#### rfm_analysis.ipynb
- **Purpose**: Interactive analysis workflow
- **Sections**:
  1. Data loading & exploration
  2. RFM metrics calculation
  3. RFM scoring (quintiles)
  4. K-means clustering (with elbow/silhouette)
  5. Segment analysis & profiling
  6. Visualization (10+ charts)
  7. Marketing recommendations
  8. Export results
- **Outputs**:
  - All CSV files
  - Inline visualizations
  - Summary statistics
- **Platform**: Jupyter Notebook / JupyterLab / Google Colab
- **Runtime**: ~2-3 minutes on standard hardware

---

### Visualizations

All visualizations should be saved as high-resolution PNG (300 DPI) for portfolio use.

#### segment_distribution.png
- **Type**: Horizontal bar chart
- **Shows**: Customer count per segment with percentages
- **Insight**: Shows segment sizes, highlights Champions are only 14%

#### revenue_by_segment.png
- **Type**: Horizontal bar chart
- **Shows**: Total revenue per segment with percentages
- **Insight**: Champions drive 46% of revenue

#### rfm_heatmap.png
- **Type**: Heatmap
- **Shows**: Average R, F, M values per segment (normalized)
- **Insight**: Visual segment differentiation

#### elbow_method.png
- **Type**: Line plot
- **Shows**: Inertia vs. K (3-10 clusters)
- **Insight**: Justifies K=8 selection

#### silhouette_analysis.png
- **Type**: Line plot
- **Shows**: Silhouette score vs. K
- **Insight**: K=8 has good silhouette score (0.46)

#### 3d_segmentation.png
- **Type**: 3D scatter plot
- **Shows**: Customers in R-F-M space, colored by segment
- **Insight**: Visual cluster separation

---

## Setup Instructions

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/customer-segmentation-rfm.git
cd customer-segmentation-rfm
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Generate Data
```bash
python data_generation.py
```

### 5. Run Analysis
```bash
jupyter notebook rfm_analysis.ipynb
```

---

## Usage Examples

### Generate Custom Dataset
```bash
# Generate 50,000 customers
python data_generation.py --customers 50000 --output large_dataset.csv

# Quiet mode (no progress output)
python data_generation.py --quiet
```

### Run Analysis Programmatically
```python
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv('ecommerce_transactions.csv')

# Calculate RFM
rfm = df.groupby('customer_id').agg({
    'transaction_date': lambda x: (pd.Timestamp('2025-11-16') - x.max()).days,
    'transaction_id': 'count',
    'total_amount': 'sum'
})

# Cluster
scaler = StandardScaler()
X_scaled = scaler.fit_transform(rfm.values)
kmeans = KMeans(n_clusters=8, random_state=42)
rfm['cluster'] = kmeans.fit_predict(X_scaled)

print(rfm.groupby('cluster').size())
```

---

## Portfolio Presentation Tips

### GitHub Repository
- ✅ Clear, detailed README with badges
- ✅ Well-organized file structure
- ✅ Comprehensive documentation
- ✅ Code comments and docstrings
- ✅ Requirements.txt with exact versions
- ✅ .gitignore for Python projects
- ✅ Professional commit messages

### Portfolio Website
Create a project page with:
1. **Hero Section**: Project title, tagline, key metrics
2. **Problem Statement**: Business context
3. **Methodology**: RFM + K-Means overview
4. **Key Findings**: 3-5 bullet points with visuals
5. **Visualizations**: 4-6 key charts embedded
6. **Impact**: ROI projections, business value
7. **Technical Skills**: Technologies used
8. **Links**: GitHub repo, notebook viewer, documentation

### Resume Bullet Points
- "Developed customer segmentation model using RFM analysis and K-Means clustering, identifying 5 distinct segments contributing to $223M revenue"
- "Designed data-driven retention marketing strategies with projected 36:1 ROI across customer segments"
- "Generated synthetic e-commerce dataset (320K+ transactions) and built reproducible analytics pipeline in Python"
- "Created executive-level reporting and technical documentation for cross-functional stakeholders"

### Interview Talking Points
- **Technical**: Explain K-means algorithm, why K=8, how silhouette score works
- **Business**: Discuss marketing strategies, ROI calculations, segment prioritization
- **Impact**: "Top 3 segments = 44% customers but 86% revenue"
- **Process**: Data → RFM → Clustering → Segmentation → Strategy
- **Challenges**: Handling outliers, choosing K, segment naming logic
- **Extensions**: Predictive churn model, real-time segmentation, A/B testing

---

## Maintenance & Updates

### Regular Updates
- **Data Refresh**: Re-run with new date ranges quarterly
- **Segment Validation**: Check if segments still make business sense
- **Metric Updates**: Adjust RFM thresholds based on business changes
- **Documentation**: Keep strategies aligned with current marketing practices

### Version Control
```bash
# Tag releases
git tag -a v1.0 -m "Initial release"
git push origin v1.0

# Branch for experiments
git checkout -b feature/hierarchical-clustering
```

---

## Additional Resources

### Learning Materials
- RFM Analysis Guide: [expressanalytics.com/rfm](https://www.expressanalytics.com/blog/rfm-analysis-customer-segmentation)
- K-Means Tutorial: [scikit-learn.org/kmeans](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- Customer Analytics: [dataquest.io/segmentation](https://www.dataquest.io/blog/customer-segmentation-using-k-means-clustering/)

### Tools & Platforms
- **Jupyter**: Interactive notebooks
- **Google Colab**: Cloud notebooks (free GPU)
- **Deepnote**: Collaborative notebooks
- **Observable**: Interactive visualizations
- **Streamlit**: Deploy as web app

---

## Troubleshooting

### Common Issues

**Issue**: "ModuleNotFoundError: No module named 'sklearn'"
**Solution**: 
```bash
pip install scikit-learn
```

**Issue**: "Memory error when clustering"
**Solution**: Sample data or use MiniBatchKMeans
```python
from sklearn.cluster import MiniBatchKMeans
kmeans = MiniBatchKMeans(n_clusters=8, batch_size=1000)
```

**Issue**: "CSV file not found"
**Solution**: Ensure you're in correct directory
```bash
pwd  # Check current directory
python data_generation.py  # Generate data first
```

---

## License

MIT License - See LICENSE file for details.

---

## Contact & Support

- **GitHub Issues**: [github.com/yourusername/customer-segmentation-rfm/issues](https://github.com/yourusername/customer-segmentation-rfm/issues)
- **Email**: your.email@example.com
- **LinkedIn**: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)

---

*Last Updated: November 19, 2025*
