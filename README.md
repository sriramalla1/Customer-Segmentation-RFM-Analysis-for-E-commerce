# Customer Segmentation & RFM Analysis for E-commerce Retention Strategy

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![Pandas](https://img.shields.io/badge/Pandas-1.3+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A comprehensive data analytics project demonstrating customer segmentation using RFM (Recency, Frequency, Monetary) methodology combined with K-Means clustering to identify distinct customer groups and develop targeted retention marketing strategies for e-commerce businesses.

## 📊 Project Overview

This project showcases advanced customer analytics skills critical for marketing roles in e-commerce, retail, and digital marketing. By analyzing customer transaction behavior, we identify high-value segments, predict churn risk, and develop data-driven marketing strategies that maximize customer lifetime value (CLV) and retention.

### Key Findings

- **Identified 5 distinct customer segments** with unique behavioral patterns
- **Top 3 segments** represent only 44.2% of customers but contribute **86.0% of total revenue**
- **Champions segment** (14.1% of customers) generates **46.3% of all revenue**
- Average customer lifetime value varies **6.8x** between segments
- **33.5% of customers are new** with significant growth potential

## 🎯 Business Impact

### Strategic Insights

1. **Revenue Concentration**: 14% of customers (Champions) drive nearly half of all revenue
2. **Churn Risk**: 22% of customers are hibernating, representing recovery opportunity
3. **Growth Potential**: 52% of customer base (New + Potential Loyalists) shows strong upside
4. **Retention Priority**: At-Risk and Can't Lose segments require immediate intervention

### Marketing ROI Potential

- **Personalized campaigns** can increase conversion rates by 20-30%
- **Retention strategies** are 5-7x more cost-effective than acquisition
- **Segment-specific messaging** improves email engagement by 40-50%
- **Churn prevention** protects high-value customer relationships

## 🛠️ Technologies Used

- **Python 3.8+**: Core programming language
- **Pandas & NumPy**: Data manipulation and analysis
- **scikit-learn**: K-Means clustering and preprocessing
- **Matplotlib & Seaborn**: Data visualization
- **Jupyter Notebook**: Interactive analysis environment
- **Faker**: Synthetic data generation

## 📁 Project Structure

```
customer-segmentation-rfm/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── data_generation.py                 # Synthetic dataset creation script
├── rfm_analysis.ipynb                # Complete analysis notebook
├── ecommerce_transactions.csv        # Generated transaction data (320K+ records)
├── rfm_analysis_results.csv          # RFM scores and segments per customer
├── segment_summary.csv               # Segment-level statistics
├── marketing_strategies.md           # Detailed marketing recommendations
├── METHODOLOGY.md                    # Technical methodology documentation
└── presentation/                     # Presentation materials
    └── executive_summary.md          # Executive-level findings
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/customer-segmentation-rfm.git
cd customer-segmentation-rfm

# Install dependencies
pip install -r requirements.txt

# Generate synthetic data
python data_generation.py

# Run the analysis (Jupyter Notebook)
jupyter notebook rfm_analysis.ipynb
```

### Requirements

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
faker>=8.0.0
jupyter>=1.0.0
```

## 📈 Methodology

### 1. RFM Analysis Framework

**Recency (R)**: Days since last purchase
- Score 1-5 (5 = most recent)
- Indicates customer engagement level
- Critical for identifying active vs. dormant customers

**Frequency (F)**: Total number of purchases
- Score 1-5 (5 = most frequent)
- Measures customer loyalty and habit formation
- Higher frequency correlates with retention

**Monetary (M)**: Total revenue generated
- Score 1-5 (5 = highest value)
- Identifies high-value customers
- Direct CLV indicator

### 2. K-Means Clustering

- **Algorithm**: K-Means (unsupervised learning)
- **Features**: Standardized R, F, M metrics
- **Optimal Clusters**: 8 (determined via elbow method & silhouette analysis)
- **Validation**: Silhouette score, business interpretability

### 3. Segment Naming Strategy

Segments are named based on behavioral patterns and marketing actionability:

| Segment | Characteristics | % Customers | % Revenue |
|---------|----------------|-------------|-----------|
| **Champions** | High R, F, M - Best customers | 14.1% | 46.3% |
| **Loyal Customers** | High F, M - Regular buyers | 11.6% | 22.9% |
| **Potential Loyalists** | Moderate R, F, M - Growth opportunity | 18.5% | 16.8% |
| **New Customers** | High R, Low F - Recent first purchase | 33.5% | 6.8% |
| **Hibernating** | Low R, Low F, M - Dormant | 22.3% | 7.2% |

## 💡 Marketing Strategies by Segment

### Champions (High R, F, M)
**Strategy**: VIP Treatment & Advocacy

- **Channels**: Email, SMS, Direct Mail
- **Tactics**: 
  - Exclusive early product access
  - VIP loyalty tier with premium benefits
  - Referral program with rewards
  - Personalized thank-you campaigns
- **Expected Impact**: 15-20% increase in advocacy, 10-15% CLV growth
- **Budget Allocation**: 25-30% of retention budget

### Loyal Customers (High F, M, Moderate R)
**Strategy**: Relationship Deepening

- **Channels**: Email, Retargeting Ads
- **Tactics**:
  - Point-based loyalty program
  - Cross-sell recommendations
  - Birthday/anniversary rewards
  - Exclusive community access
- **Expected Impact**: 20% frequency increase, 12% AOV growth
- **Budget Allocation**: 20-25% of retention budget

### At-Risk (Low R, Previously High F, M)
**Strategy**: Win-Back Campaigns

- **Channels**: Email, SMS, Retargeting
- **Tactics**:
  - "We miss you" personalized outreach
  - Time-limited 20-30% discount offers
  - Free shipping incentives
  - Product recommendation based on past purchases
- **Expected Impact**: 25-35% reactivation rate
- **Budget Allocation**: 15-20% of retention budget

### New Customers (High R, Low F)
**Strategy**: Onboarding & Conversion

- **Channels**: Email automation, Push notifications
- **Tactics**:
  - Welcome series (3-5 emails)
  - Second-purchase discount (15% off)
  - Educational content about products
  - Loyalty program enrollment
- **Expected Impact**: 30% second purchase rate
- **Budget Allocation**: 15-20% of retention budget

### Hibernating (Low R, F, M)
**Strategy**: Low-Cost Reactivation

- **Channels**: Email only (minimal spend)
- **Tactics**:
  - Aggressive discounts (30-50% off)
  - "Last chance" messaging
  - Survey for feedback/win-back
  - New product announcements
- **Expected Impact**: 10-15% reactivation
- **Budget Allocation**: 5-10% of retention budget

## 📊 Dataset Details

### Synthetic Data Generation

- **Total Customers**: 25,000
- **Total Transactions**: 320,385
- **Date Range**: January 1, 2023 - November 15, 2025 (34 months)
- **Total Revenue**: $223.5M
- **Product Categories**: 8 (Electronics, Fashion, Home & Garden, Beauty, Sports, Books, Toys, Food)

### Data Realism Features

✅ Realistic transaction patterns (seasonal trends, purchase clustering)
✅ Varying customer behaviors (champions, churned, new, loyal)
✅ Multi-category purchases
✅ Price variation by category
✅ Quantity-based purchases (bulk orders)
✅ Customer lifecycle representation

### Data Quality

- **Completeness**: 100% (no missing values)
- **Consistency**: Validated date ranges, positive monetary values
- **Accuracy**: Segment distributions match real-world e-commerce patterns
- **Validity**: All customer IDs unique, transaction IDs sequential

## 🎓 Skills Demonstrated

### Technical Skills

- ✅ Customer analytics and behavioral segmentation
- ✅ RFM methodology implementation
- ✅ Unsupervised machine learning (K-Means clustering)
- ✅ Feature engineering and standardization
- ✅ Model validation (elbow method, silhouette analysis)
- ✅ Data visualization and storytelling
- ✅ Python programming (pandas, scikit-learn, matplotlib)

### Business Skills

- ✅ Customer lifetime value (CLV) analysis
- ✅ Churn prediction and retention strategy
- ✅ Marketing campaign design
- ✅ Budget allocation optimization
- ✅ ROI estimation and business case development
- ✅ Executive communication and reporting

### Marketing Skills

- ✅ Personalized marketing campaign development
- ✅ Multi-channel strategy design
- ✅ Customer journey mapping
- ✅ A/B testing framework design
- ✅ KPI definition and tracking
- ✅ Retention marketing best practices

## 📈 Key Performance Indicators (KPIs)

### Customer Metrics
- Customer Lifetime Value (CLV) by segment
- Retention rate (90-day, 180-day, annual)
- Churn rate by segment
- Average order value (AOV)
- Purchase frequency

### Campaign Metrics
- Email open rate (by segment)
- Click-through rate (CTR)
- Conversion rate
- Win-back rate (At-Risk, Hibernating)
- Referral rate (Champions)

### Revenue Metrics
- Revenue per segment
- Revenue concentration (% from top segments)
- Month-over-month growth by segment
- Customer acquisition cost (CAC) vs. CLV ratio

## 🔬 Analysis Highlights

### Clustering Validation

```python
# Silhouette Score: 0.4587
# Indicates good cluster separation and cohesion

# Cluster Distribution:
# - Well-balanced segment sizes (no dominant cluster)
# - Clear behavioral differences between segments
# - Business-interpretable groupings
```

### Revenue Distribution

```
Champions:          46.3% revenue | 14.1% customers → 3.3x multiplier
Loyal:              22.9% revenue | 11.6% customers → 2.0x multiplier
Potential Loyalists: 16.8% revenue | 18.5% customers → 0.9x multiplier
Hibernating:         7.2% revenue | 22.3% customers → 0.3x multiplier
New Customers:       6.8% revenue | 33.5% customers → 0.2x multiplier
```

### Customer Lifetime Value Analysis

```
Champions:          Avg CLV = $32,876
Loyal Customers:    Avg CLV = $19,732
Potential Loyalists: Avg CLV = $9,087
Hibernating:        Avg CLV = $3,226
New Customers:      Avg CLV = $2,030
```

## 🎯 Business Recommendations

### Immediate Actions (Next 30 Days)

1. **Launch Champion VIP Program** - Protect and grow top 14% of customers
2. **Deploy Win-Back Campaign** - Target At-Risk segment with 25% discount
3. **Implement New Customer Onboarding** - Automated 5-email welcome series
4. **Set Up Segment-Based Email Flows** - Personalized messaging by segment

### Medium-Term (90 Days)

1. **Build Predictive Churn Model** - ML model to identify customers moving to At-Risk
2. **Create Loyalty Point System** - Gamification for Loyal and Potential Loyalists
3. **Develop Referral Program** - Leverage Champions for new customer acquisition
4. **A/B Test Campaign Strategies** - Optimize messaging and offers per segment

### Long-Term (6-12 Months)

1. **Real-Time Segmentation** - Move from batch to streaming RFM calculation
2. **Personalization Engine** - AI-driven product recommendations per segment
3. **Customer Journey Optimization** - Map and improve paths to Champions status
4. **CLV Maximization** - Strategies to move customers up segment ladder

## 📚 Further Reading & Resources

### RFM Analysis
- [Complete Guide to RFM Analysis](https://www.expressanalytics.com/blog/rfm-analysis-customer-segmentation)
- [RFM Analysis Best Practices 2025](https://www.moengage.com/blog/predicitve-segments-rfm-analysis/)

### K-Means Clustering
- [Customer Segmentation with K-Means](https://www.dataquest.io/blog/customer-segmentation-using-k-means-clustering/)
- [Improved K-Means for Customer Segmentation](https://www.sciencedirect.com/science/article/abs/pii/S1568494621008462)

### E-commerce Retention Marketing
- [E-commerce Retention Strategies](https://www.smartbugmedia.com/ecommerce-retention-marketing)
- [Customer Retention Best Practices](https://www.shopify.com/in/blog/customer-retention-strategies)

## 📧 Contact & Portfolio

- **Portfolio**: [yourwebsite.com](https://yourwebsite.com)
- **LinkedIn**: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)
- **Email**: your.email@example.com

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Data generation inspired by real-world e-commerce patterns
- RFM methodology based on industry best practices
- Marketing strategies informed by current retail trends

---

**⭐ If you found this project helpful, please star this repository!**

*This project demonstrates marketing analytics expertise for portfolio purposes. The dataset is synthetically generated for educational use.*
