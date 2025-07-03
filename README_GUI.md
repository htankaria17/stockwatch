# 🤖 AI-Powered Stock Investment Analyzer - GUI Application

## Quick Start Guide

### 🚀 Launch the Application
```bash
python3 run_app.py
```
This will automatically:
- Install all required dependencies
- Launch the web interface at http://localhost:8501

### 📋 Basic Usage
1. **Configure Profile** in sidebar (name, email, country, investment amount)
2. **Select Risk Tolerance** (Conservative/Moderate/Aggressive) 
3. **Click "Analyze Stocks"** to get AI-powered recommendations
4. **Explore Results** in the four main tabs:
   - 📊 Analysis: Top recommendations with scores
   - 📈 Portfolio: Allocation charts and tables
   - 🤖 ML Insights: Feature importance and predictions
   - ⚙️ Settings: Configuration and help

### 📧 Email Notifications (Optional)
1. Enable 2FA on your Gmail account
2. Generate App Password: Google Account → Security → App passwords
3. Enter Gmail address and App Password in sidebar
4. Select notification frequency (Daily/Weekly/Monthly)

### 🌍 Supported Markets
- **US**: Top S&P 500 stocks (AAPL, MSFT, GOOGL, etc.)
- **Canada**: TSX stocks (SHOP.TO, RY.TO, TD.TO, etc.)
- **India**: NIFTY stocks (RELIANCE.NS, TCS.NS, INFY.NS, etc.)

### 🧠 AI/ML Features
- **Gradient Boosting Regression** for return prediction
- **12 financial features** analyzed per stock
- **Risk-adjusted portfolio allocation**
- **Feature importance analysis**

### 📊 Investment Analysis
- **Comprehensive scoring** (0-100 scale)
- **Risk categorization** (STRONG BUY, BUY, HOLD, etc.)
- **Portfolio optimization** based on user preferences
- **Share calculations** with exact investment amounts

### 🔧 Troubleshooting
- **Port in use?** Try: `streamlit run stock_investment_app.py --server.port 8502`
- **Email not working?** Use Gmail App Password, not regular password
- **Slow analysis?** Normal for first run, internet connection required

### 📚 Documentation
- **`GUI_Application_Guide.md`** - Comprehensive user guide
- **`GUI_Development_Summary.md`** - Technical overview
- **`Stock_Analysis_Enhancement_Summary.md`** - Original features

### 🎯 Key Features
✅ Multi-country stock analysis  
✅ AI/ML predictions  
✅ Interactive web dashboard  
✅ Email notifications  
✅ Portfolio optimization  
✅ Risk-based recommendations  

**Get started:** `python3 run_app.py` 🚀