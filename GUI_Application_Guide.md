# 🤖 AI-Powered Stock Investment Analyzer - GUI Application

## Overview
A comprehensive web-based GUI application that transforms the basic stock analysis script into a sophisticated investment platform with AI/ML capabilities, email notifications, and country-specific recommendations.

## ✨ Key Features

### 🌍 Multi-Country Support
- **United States**: Top 20 S&P 500 stocks
- **Canada**: TSX top 15 stocks
- **India**: NIFTY top 15 stocks

### 🤖 AI/ML Integration
- **Gradient Boosting Regression** for return prediction
- **Feature importance analysis** showing which metrics matter most
- **Synthetic data training** when historical data is insufficient
- **12 key financial features** for ML model training

### 📧 Automated Email Notifications
- **Daily, Weekly, Monthly** notification options
- **HTML-formatted emails** with professional styling
- **Gmail integration** with App Password support
- **Automated scheduling** with background service

### 📊 Interactive Dashboard
- **Real-time analysis** with progress indicators
- **Portfolio visualization** with pie charts and tables
- **ML insights** with feature importance and scatter plots
- **Risk-based recommendations** (Conservative/Moderate/Aggressive)

### 💰 Investment Optimization
- **Personalized allocation** based on user preferences
- **Risk-adjusted positioning** per investment amount
- **Share calculations** with actual investment amounts
- **Expected return projections** using target prices

## 🚀 Installation & Setup

### 1. Install Dependencies
```bash
pip install -r requirements_gui.txt
```

**Required packages:**
- streamlit (Web GUI framework)
- yfinance (Stock data)
- pandas, numpy (Data processing)
- plotly (Interactive charts)
- scikit-learn (Machine learning)
- schedule (Task scheduling)
- beautifulsoup4 (Web scraping)

### 2. Run the Application
```bash
# Easy launcher (installs dependencies automatically)
python3 run_app.py

# Or run directly
streamlit run stock_investment_app.py
```

### 3. Access the Web Interface
- **Local URL**: http://localhost:8501
- **Network URL**: Available to other devices on your network

## 📋 User Guide

### Step 1: Configure Profile
**In the sidebar, enter:**
- Full Name
- Email Address
- Country/Market selection
- Investment Amount ($100 - $1,000,000)
- Desired Annual Return (5% - 30%)
- Risk Tolerance (Conservative/Moderate/Aggressive)
- Email Notification Frequency

### Step 2: Email Setup (Optional)
**For notifications:**
1. Enable 2-Factor Authentication on Gmail
2. Generate App Password: Google Account → Security → App passwords
3. Enter Gmail address and App Password in sidebar

### Step 3: Run Analysis
1. Click **"🔍 Analyze Stocks"** button
2. Wait for AI/ML analysis to complete
3. Review top 5 recommendations
4. Optionally send email report

### Step 4: Explore Results
**Four main tabs:**

#### 📊 Analysis Tab
- Top 5 stock recommendations
- Investment scores (0-100)
- ML predicted returns
- Detailed financial metrics
- Investment allocation per stock

#### 📈 Portfolio Tab
- Portfolio allocation pie chart
- Investment distribution table
- Total portfolio value
- Recommended share quantities

#### 🤖 ML Insights Tab
- Feature importance chart
- ML prediction vs score scatter plot
- Model performance indicators

#### ⚙️ Settings Tab
- Email setup instructions
- ML model information
- Supported markets overview

## 🎯 How the AI/ML Works

### ML Model Architecture
**Gradient Boosting Regression** trained on:
- **Financial Ratios**: PE, PB, ROE, Profit Margins
- **Growth Metrics**: Revenue Growth, Earnings Growth
- **Technical Indicators**: Price Momentum, Volatility
- **Market Data**: Volume Ratios, Beta

### Scoring Algorithm
**Investment Score (0-100) based on:**
- **Fundamental Analysis (70%)**: Traditional financial metrics
- **ML Prediction (30%)**: AI-predicted returns

**Score Ranges:**
- **80-100**: STRONG BUY (Low Risk)
- **65-79**: BUY (Medium-Low Risk)
- **50-64**: HOLD (Medium Risk)
- **35-49**: WEAK HOLD (Medium-High Risk)
- **0-34**: AVOID (High Risk)

### Risk Adjustment
**Investment allocation considers:**
- User risk tolerance
- Stock volatility
- Financial health metrics
- ML confidence levels

## 📧 Email Notification System

### Setup Process
1. **Enable 2FA** on your Gmail account
2. **Generate App Password**:
   - Go to Google Account settings
   - Security → App passwords
   - Generate password for "Mail"
3. **Use App Password** (not regular password)

### Email Content
**Professional HTML emails include:**
- Top 5 recommendations
- Investment scores and ML predictions
- Current prices and key metrics
- Sector information
- Analysis timestamp

### Automated Scheduling
**Daily scheduler runs:**
- **8:00 AM**: Daily recommendations
- **Sunday 9:00 AM**: Weekly recommendations
- **1st of month 10:00 AM**: Monthly recommendations
- **Monday 2:00 AM**: Database cleanup

## 🗄️ Database Schema

### Users Table
- User profile information
- Investment preferences
- Notification settings

### Analysis History
- Historical analysis results
- ML predictions
- Stock scores and recommendations

### Email Log
- Email sending status
- Delivery confirmations
- Error tracking

## 📈 Investment Allocation Logic

### Risk-Based Allocation
```python
# Risk multipliers
Conservative: 0.6x allocation
Moderate: 0.8x allocation  
Aggressive: 1.0x allocation
```

### Portfolio Distribution
- **Maximum 20%** per individual stock
- **Score-based weighting** (higher scores = larger allocation)
- **Share-based calculations** (whole shares only)
- **Cash remainder tracking**

### Expected Returns
- **Target price analysis** from analyst recommendations
- **ML-predicted returns** from trained models
- **Risk-adjusted projections**

## 🔧 Advanced Configuration

### Environment Variables
**For automated scheduling:**
```bash
export SENDER_EMAIL="your_email@gmail.com"
export SENDER_PASSWORD="your_app_password"
```

### Custom Stock Lists
**Modify country_stocks dictionary in EnhancedStockAnalyzer:**
```python
self.country_stocks = {
    'US': ['AAPL', 'MSFT', ...],
    'CANADA': ['SHOP.TO', 'RY.TO', ...],
    'INDIA': ['RELIANCE.NS', 'TCS.NS', ...]
}
```

### ML Model Tuning
**Adjust model parameters:**
```python
self.model = GradientBoostingRegressor(
    n_estimators=200,      # Number of trees
    learning_rate=0.1,     # Learning rate
    max_depth=6,           # Tree depth
    random_state=42
)
```

## 🛠️ Troubleshooting

### Common Issues

**1. Email Not Sending**
- ✅ Check Gmail App Password (not regular password)
- ✅ Verify 2FA is enabled
- ✅ Check internet connection
- ✅ Ensure correct email format

**2. Stock Data Not Loading**
- ✅ Check internet connection
- ✅ Verify stock symbols are correct
- ✅ yfinance API may have rate limits
- ✅ Try different country/market

**3. ML Model Errors**
- ✅ Insufficient data - model uses synthetic data
- ✅ Check scikit-learn installation
- ✅ Verify numpy compatibility

**4. GUI Not Loading**
- ✅ Check Streamlit installation
- ✅ Port 8501 may be in use
- ✅ Try different port: `--server.port 8502`

## 🎮 Usage Examples

### Example 1: Conservative US Investor
```
Name: John Smith
Email: john@email.com
Country: US
Investment: $50,000
Return Target: 8%
Risk: Conservative
```
**Result:** Portfolio focused on dividend-paying blue chips

### Example 2: Aggressive Indian Investor
```
Name: Priya Sharma
Email: priya@email.com
Country: INDIA
Investment: ₹10,00,000
Return Target: 20%
Risk: Aggressive
```
**Result:** Growth-focused tech and banking stocks

### Example 3: Moderate Canadian Investor
```
Name: Alex Brown
Email: alex@email.com
Country: CANADA
Investment: CAD $25,000
Return Target: 12%
Risk: Moderate
```
**Result:** Balanced portfolio with growth and value stocks

## 🚀 Running as Background Service

### Start Daily Scheduler
```bash
# Set environment variables
export SENDER_EMAIL="your_email@gmail.com"
export SENDER_PASSWORD="your_app_password"

# Run scheduler
python3 daily_scheduler.py

# Or run in background
nohup python3 daily_scheduler.py > scheduler.log 2>&1 &
```

### Monitor Scheduler
```bash
# Check logs
tail -f scheduler.log

# Check if running
ps aux | grep daily_scheduler
```

## 🔮 Future Enhancements

### Planned Features
1. **Sentiment Analysis** of scraped news
2. **Technical Analysis** indicators (RSI, MACD, Moving Averages)
3. **Sector Comparison** and rotation strategies
4. **Backtesting** historical performance
5. **Real-time Alerts** for significant changes
6. **ESG Scoring** integration
7. **Options Analysis** for advanced strategies
8. **Mobile App** version

### API Integrations
- **Alpha Vantage** for additional data
- **Yahoo Finance** enhanced endpoints
- **SEC EDGAR** for fundamental data
- **News APIs** for sentiment analysis

## 📞 Support & Maintenance

### Log Files
- **scheduler.log**: Background service logs
- **investment_app.db**: SQLite database
- **data/**: Analysis export files

### Performance Monitoring
- Track email delivery rates
- Monitor ML model accuracy
- Analyze user engagement
- Database performance metrics

## 🎯 Best Practices

### For Users
1. **Diversify** across multiple recommendations
2. **Regular review** of portfolio allocation
3. **Risk assessment** based on market conditions
4. **Long-term perspective** for investment decisions

### For Administrators
1. **Regular backups** of user database
2. **Monitor server resources** for Streamlit app
3. **Update stock lists** quarterly
4. **Retrain ML models** with new data

---

**Disclaimer**: This application provides automated investment analysis for educational and research purposes. Always conduct your own research and consider consulting with financial advisors before making investment decisions. Past performance does not guarantee future results.