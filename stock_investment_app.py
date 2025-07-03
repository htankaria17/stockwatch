import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sqlite3
import smtplib

# Email imports with fallback
try:
    from email.mime.text import MIMEText as MimeText
    from email.mime.multipart import MIMEMultipart as MimeMultipart
except ImportError:
    try:
        from email.MIMEText import MIMEText as MimeText
        from email.MIMEMultipart import MIMEMultipart as MimeMultipart
    except ImportError:
        # Fallback for older Python versions
        from email.mime.text import MIMEText as MimeText
        from email.mime.multipart import MIMEMultipart as MimeMultipart

# Application Version Information
APP_VERSION = "2.1.0"
APP_BUILD_DATE = "2024-12-21"
APP_NAME = "AI-Powered Stock Investment Analyzer"
VERSION_NOTES = {
    "2.1.0": [
        "Added SENSEX stocks to Indian market analysis (35 total stocks)",
        "Fixed Python 3.13 email import compatibility",
        "Enhanced ML prediction reporting and documentation",
        "Added comprehensive version tracking system",
        "Improved error handling and user experience"
    ],
    "2.0.0": [
        "Complete GUI transformation with Streamlit",
        "Integrated AI/ML prediction system",
        "Multi-country support (US, Canada, India)",
        "Email automation and scheduling",
        "Portfolio optimization algorithms"
    ],
    "1.0.0": [
        "Basic stock analysis script",
        "Single stock analysis capability",
        "Fundamental metrics calculation"
    ]
}
import requests
from bs4 import BeautifulSoup
import json
import time
import warnings
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import schedule
import threading
from typing import Dict, List, Tuple, Optional
import os
from dataclasses import dataclass
import hashlib

warnings.filterwarnings('ignore')

@dataclass
class UserProfile:
    name: str
    email: str
    country: str
    investment_amount: float
    desired_return: float
    risk_tolerance: str
    notification_frequency: str

class DatabaseManager:
    def __init__(self, db_name="investment_app.db"):
        self.db_name = db_name
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                country TEXT NOT NULL,
                investment_amount REAL NOT NULL,
                desired_return REAL NOT NULL,
                risk_tolerance TEXT NOT NULL,
                notification_frequency TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Stock analysis history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                symbol TEXT NOT NULL,
                analysis_date DATE NOT NULL,
                score REAL NOT NULL,
                recommendation TEXT NOT NULL,
                current_price REAL,
                predicted_price REAL,
                analysis_data TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Email notifications log
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS email_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                subject TEXT,
                status TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_user(self, user: UserProfile) -> int:
        """Save user profile to database"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO users 
            (name, email, country, investment_amount, desired_return, risk_tolerance, notification_frequency)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (user.name, user.email, user.country, user.investment_amount, 
              user.desired_return, user.risk_tolerance, user.notification_frequency))
        
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return user_id
    
    def get_user_by_email(self, email: str) -> Optional[UserProfile]:
        """Get user by email"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return UserProfile(
                name=row[1], email=row[2], country=row[3],
                investment_amount=row[4], desired_return=row[5],
                risk_tolerance=row[6], notification_frequency=row[7]
            )
        return None
    
    def save_analysis(self, user_id: int, symbol: str, score: float, 
                     recommendation: str, current_price: float, 
                     predicted_price: float, analysis_data: dict):
        """Save analysis results"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO analysis_history 
            (user_id, symbol, analysis_date, score, recommendation, current_price, predicted_price, analysis_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, symbol, datetime.now().date(), score, recommendation, 
              current_price, predicted_price, json.dumps(analysis_data)))
        
        conn.commit()
        conn.close()

class MLPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'pe_ratio', 'pb_ratio', 'roe', 'profit_margin', 'revenue_growth',
            'debt_to_equity', 'dividend_yield', 'beta', 'volume_ratio',
            'price_momentum_1m', 'price_momentum_3m', 'volatility'
        ]
    
    def prepare_features(self, stock_data: Dict) -> np.array:
        """Prepare features for ML model"""
        features = []
        for feature in self.feature_names:
            value = stock_data.get(feature, 0)
            # Handle None values and outliers
            if value is None or pd.isna(value):
                value = 0
            elif feature == 'pe_ratio' and value > 100:
                value = 100  # Cap PE ratio
            elif feature == 'pb_ratio' and value > 20:
                value = 20   # Cap PB ratio
            features.append(float(value))
        
        return np.array(features).reshape(1, -1)
    
    def train_model(self, historical_data: List[Dict]):
        """Train ML model with historical data"""
        if len(historical_data) < 50:  # Need minimum data
            # Use a simple ensemble model with default parameters
            self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            # Create synthetic training data based on financial principles
            X, y = self._create_synthetic_data()
        else:
            X, y = self._prepare_training_data(historical_data)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        self.model.fit(X_scaled, y)
        
        return self.model
    
    def _create_synthetic_data(self) -> Tuple[np.array, np.array]:
        """Create synthetic training data based on financial principles"""
        np.random.seed(42)
        n_samples = 1000
        
        # Generate synthetic features
        pe_ratios = np.random.gamma(2, 10, n_samples)  # PE ratios
        pb_ratios = np.random.gamma(1.5, 2, n_samples)  # PB ratios
        roe = np.random.beta(2, 3, n_samples)  # ROE
        profit_margin = np.random.beta(2, 8, n_samples)  # Profit margins
        revenue_growth = np.random.normal(0.08, 0.15, n_samples)  # Revenue growth
        debt_to_equity = np.random.gamma(1, 0.5, n_samples)  # Debt to equity
        dividend_yield = np.random.beta(1, 4, n_samples) * 0.1  # Dividend yield
        beta = np.random.normal(1, 0.3, n_samples)  # Beta
        volume_ratio = np.random.gamma(2, 0.5, n_samples)  # Volume ratio
        momentum_1m = np.random.normal(0, 0.1, n_samples)  # 1M momentum
        momentum_3m = np.random.normal(0, 0.15, n_samples)  # 3M momentum
        volatility = np.random.gamma(2, 0.1, n_samples)  # Volatility
        
        X = np.column_stack([
            pe_ratios, pb_ratios, roe, profit_margin, revenue_growth,
            debt_to_equity, dividend_yield, beta, volume_ratio,
            momentum_1m, momentum_3m, volatility
        ])
        
        # Generate target based on financial logic
        # Lower PE/PB, higher ROE/profit margin = higher returns
        y = (
            -0.3 * np.log(pe_ratios + 1) +  # Lower PE is better
            -0.2 * np.log(pb_ratios + 1) +  # Lower PB is better
            0.4 * roe +                     # Higher ROE is better
            0.3 * profit_margin +           # Higher margin is better
            0.2 * revenue_growth +          # Growth is good
            -0.1 * debt_to_equity +         # Less debt is better
            0.1 * dividend_yield +          # Dividends are good
            0.1 * momentum_3m +             # Momentum is good
            np.random.normal(0, 0.05, n_samples)  # Add noise
        )
        
        return X, y
    
    def predict_return(self, stock_data: Dict) -> float:
        """Predict expected return for a stock"""
        if self.model is None:
            self.train_model([])  # Train with synthetic data
        
        features = self.prepare_features(stock_data)
        features_scaled = self.scaler.transform(features)
        
        prediction = self.model.predict(features_scaled)[0]
        
        # Convert to percentage and cap at reasonable bounds
        return max(-50, min(100, prediction * 100))
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model"""
        if self.model is None:
            return {}
        
        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance))

class EnhancedStockAnalyzer:
    def __init__(self):
        self.ml_predictor = MLPredictor()
        self.country_stocks = {
            'US': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V',
                   'PG', 'UNH', 'HD', 'MA', 'DIS', 'PYPL', 'ADBE', 'NFLX', 'CRM', 'NVDA'],
            'CANADA': ['SHOP.TO', 'RY.TO', 'TD.TO', 'BNS.TO', 'BMO.TO', 'CNR.TO', 'CP.TO', 
                      'SU.TO', 'CNQ.TO', 'ABX.TO', 'CCO.TO', 'ENB.TO', 'TRP.TO', 'FNV.TO', 'WPM.TO'],
            'INDIA': [
                     # NIFTY 50 Top Stocks
                     'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 
                     'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS',
                     'LT.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'AXISBANK.NS', 'SUNPHARMA.NS',
                     'NESTLEIND.NS', 'BAJFINANCE.NS', 'HCLTECH.NS', 'WIPRO.NS', 'ULTRACEMCO.NS',
                     
                     # SENSEX Top Stocks (Additional)
                     'POWERGRID.NS', 'NTPC.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS', 'M&M.NS',
                     'TECHM.NS', 'BAJAJFINSV.NS', 'DRREDDY.NS', 'CIPLA.NS', 'DIVISLAB.NS',
                     'ADANIPORTS.NS', 'JSWSTEEL.NS', 'TITAN.NS', 'GRASIM.NS', 'HEROMOTOCO.NS'
                     ]
        }
        
    def get_stock_data(self, symbol: str) -> Dict:
        """Get comprehensive stock data"""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            history = stock.history(period="1y")
            
            if len(history) == 0:
                return {}
            
            current_price = history['Close'].iloc[-1]
            
            # Calculate technical indicators
            volume_ratio = history['Volume'].iloc[-20:].mean() / history['Volume'].mean() if len(history) > 20 else 1
            
            fundamentals = {
                'symbol': symbol,
                'company_name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'current_price': current_price,
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'pb_ratio': info.get('priceToBook', 0),
                'roe': info.get('returnOnEquity', 0),
                'profit_margin': info.get('profitMargins', 0),
                'revenue_growth': info.get('revenueGrowth', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', 1),
                'volume_ratio': volume_ratio,
                'ebitda': info.get('ebitda', 0),
                'free_cashflow': info.get('freeCashflow', 0),
                'target_price': info.get('targetMeanPrice', current_price),
                'recommendation': info.get('recommendationMean', 3),
            }
            
            # Calculate momentum and volatility
            if len(history) >= 252:
                fundamentals['price_momentum_1y'] = (current_price / history['Close'].iloc[-252]) - 1
            if len(history) >= 63:
                fundamentals['price_momentum_3m'] = (current_price / history['Close'].iloc[-63]) - 1
            if len(history) >= 21:
                fundamentals['price_momentum_1m'] = (current_price / history['Close'].iloc[-21]) - 1
            
            fundamentals['volatility'] = history['Close'].pct_change().std() * np.sqrt(252)
            
            return fundamentals
            
        except Exception as e:
            st.error(f"Error getting data for {symbol}: {e}")
            return {}
    
    def calculate_investment_score(self, fundamentals: Dict, ml_prediction: float) -> Tuple[float, Dict]:
        """Enhanced scoring with ML prediction"""
        score = 0
        criteria = {}
        
        # Traditional fundamental analysis (70% weight)
        fundamental_score = 0
        
        # Valuation (25%)
        pe_ratio = fundamentals.get('pe_ratio', 0)
        if 0 < pe_ratio < 15:
            fundamental_score += 25
            criteria['pe_ratio'] = 'Excellent'
        elif 15 <= pe_ratio < 25:
            fundamental_score += 18
            criteria['pe_ratio'] = 'Good'
        elif 25 <= pe_ratio < 35:
            fundamental_score += 10
            criteria['pe_ratio'] = 'Fair'
        else:
            criteria['pe_ratio'] = 'Poor'
        
        # Profitability (20%)
        roe = fundamentals.get('roe', 0)
        if roe > 0.15:
            fundamental_score += 20
            criteria['roe'] = 'Excellent'
        elif roe > 0.10:
            fundamental_score += 15
            criteria['roe'] = 'Good'
        elif roe > 0.05:
            fundamental_score += 8
            criteria['roe'] = 'Fair'
        else:
            criteria['roe'] = 'Poor'
        
        # Growth (15%)
        revenue_growth = fundamentals.get('revenue_growth', 0)
        if revenue_growth > 0.15:
            fundamental_score += 15
            criteria['revenue_growth'] = 'Excellent'
        elif revenue_growth > 0.08:
            fundamental_score += 12
            criteria['revenue_growth'] = 'Good'
        elif revenue_growth > 0.03:
            fundamental_score += 6
            criteria['revenue_growth'] = 'Fair'
        else:
            criteria['revenue_growth'] = 'Poor'
        
        # Financial Health (10%)
        debt_to_equity = fundamentals.get('debt_to_equity', 0)
        if debt_to_equity < 0.3:
            fundamental_score += 10
            criteria['debt_to_equity'] = 'Excellent'
        elif debt_to_equity < 0.6:
            fundamental_score += 7
            criteria['debt_to_equity'] = 'Good'
        elif debt_to_equity < 1.0:
            fundamental_score += 4
            criteria['debt_to_equity'] = 'Fair'
        else:
            criteria['debt_to_equity'] = 'Poor'
        
        # ML Prediction component (30% weight)
        ml_score = max(0, min(30, (ml_prediction + 10) * 1.5))  # Scale prediction to 0-30
        
        total_score = fundamental_score + ml_score
        criteria['ml_prediction'] = f"{ml_prediction:.1f}% predicted return"
        
        return total_score, criteria
    
    def analyze_stocks_for_country(self, country: str, user_profile: UserProfile) -> List[Dict]:
        """Analyze stocks for specific country with user preferences"""
        stocks = self.country_stocks.get(country, [])
        analyses = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, symbol in enumerate(stocks):
            status_text.text(f'Analyzing {symbol}...')
            progress_bar.progress((i + 1) / len(stocks))
            
            fundamentals = self.get_stock_data(symbol)
            if not fundamentals:
                continue
            
            # Get ML prediction
            ml_prediction = self.ml_predictor.predict_return(fundamentals)
            
            # Calculate enhanced score
            score, criteria = self.calculate_investment_score(fundamentals, ml_prediction)
            
            # Calculate investment recommendation based on user preferences
            investment_recommendation = self.calculate_investment_allocation(
                fundamentals, user_profile, score
            )
            
            analysis = {
                'symbol': symbol,
                'fundamentals': fundamentals,
                'score': score,
                'ml_prediction': ml_prediction,
                'criteria': criteria,
                'investment_recommendation': investment_recommendation,
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            analyses.append(analysis)
            time.sleep(0.5)  # Rate limiting
        
        progress_bar.empty()
        status_text.empty()
        
        # Sort by score
        analyses.sort(key=lambda x: x['score'], reverse=True)
        return analyses[:20]  # Return top 20
    
    def calculate_investment_allocation(self, fundamentals: Dict, user_profile: UserProfile, score: float) -> Dict:
        """Calculate investment allocation based on user preferences"""
        current_price = fundamentals.get('current_price', 0)
        if current_price == 0:
            return {'shares': 0, 'amount': 0, 'percentage': 0}
        
        # Risk-adjusted allocation based on score and user risk tolerance
        risk_multipliers = {'Conservative': 0.6, 'Moderate': 0.8, 'Aggressive': 1.0}
        risk_multiplier = risk_multipliers.get(user_profile.risk_tolerance, 0.8)
        
        # Base allocation percentage (0-20% per stock)
        base_allocation = min(20, (score / 100) * 20) * risk_multiplier
        
        # Amount to invest in this stock
        investment_amount = (base_allocation / 100) * user_profile.investment_amount
        shares = int(investment_amount / current_price)
        actual_amount = shares * current_price
        
        return {
            'shares': shares,
            'amount': actual_amount,
            'percentage': base_allocation,
            'expected_return': fundamentals.get('target_price', current_price) - current_price
        }

class EmailNotifier:
    def __init__(self):
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
    
    def send_daily_recommendations(self, user_email: str, recommendations: List[Dict], 
                                 sender_email: str, sender_password: str):
        """Send daily recommendations via email"""
        try:
            # Create email content
            subject = f"Daily Investment Recommendations - {datetime.now().strftime('%Y-%m-%d')}"
            
            html_content = self._create_email_html(recommendations)
            
            # Create message
            msg = MimeMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = sender_email
            msg['To'] = user_email
            
            # Attach HTML content
            html_part = MimeText(html_content, 'html')
            msg.attach(html_part)
            
            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
            server.quit()
            
            return True
            
        except Exception as e:
            st.error(f"Failed to send email: {e}")
            return False
    
    def _create_email_html(self, recommendations: List[Dict]) -> str:
        """Create HTML email content"""
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .header {{ background-color: #2E8B57; color: white; padding: 20px; text-align: center; }}
                .recommendation {{ border: 1px solid #ddd; margin: 10px; padding: 15px; border-radius: 5px; }}
                .strong-buy {{ border-left: 5px solid #28a745; }}
                .buy {{ border-left: 5px solid #007bff; }}
                .hold {{ border-left: 5px solid #ffc107; }}
                .metric {{ display: inline-block; margin: 5px 10px; }}
                .footer {{ background-color: #f8f9fa; padding: 15px; text-align: center; font-size: 12px; color: #666; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Your Daily Investment Recommendations</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p style="font-size: 14px; opacity: 0.9;">{APP_NAME} v{APP_VERSION}</p>
            </div>
            
            <div style="padding: 20px;">
                <h2>Top 5 Recommendations</h2>
        """
        
        for i, rec in enumerate(recommendations[:5], 1):
            fund = rec['fundamentals']
            recommendation_class = rec['criteria'].get('recommendation', 'hold').lower().replace(' ', '-')
            
            html += f"""
                <div class="recommendation {recommendation_class}">
                    <h3>{i}. {fund.get('company_name', 'N/A')} ({rec['symbol']})</h3>
                    <p><strong>Score:</strong> {rec['score']:.1f}/100 | 
                       <strong>ML Prediction:</strong> {rec['ml_prediction']:.1f}% return</p>
                    <p><strong>Current Price:</strong> ${fund.get('current_price', 0):.2f} | 
                       <strong>Sector:</strong> {fund.get('sector', 'N/A')}</p>
                    <div class="metric"><strong>PE Ratio:</strong> {fund.get('pe_ratio', 0):.2f}</div>
                    <div class="metric"><strong>ROE:</strong> {fund.get('roe', 0)*100:.1f}%</div>
                    <div class="metric"><strong>Revenue Growth:</strong> {fund.get('revenue_growth', 0)*100:.1f}%</div>
                </div>
            """
        
        html += f"""
            </div>
            <div class="footer">
                <p>This is an automated recommendation based on fundamental and ML analysis.<br>
                Please conduct your own research before making investment decisions.</p>
                <hr style="margin: 10px 0; border: none; border-top: 1px solid #ddd;">
                <p><strong>{APP_NAME}</strong> v{APP_VERSION} | Build Date: {APP_BUILD_DATE}</p>
                <p>AI-Powered Investment Analysis | Machine Learning Enhanced Predictions</p>
            </div>
        </body>
        </html>
        """
        
        return html

def main():
    st.set_page_config(
        page_title=f"{APP_NAME} v{APP_VERSION}",
        page_icon="📈",
        layout="wide"
    )
    
    # Initialize components
    db = DatabaseManager()
    analyzer = EnhancedStockAnalyzer()
    emailer = EmailNotifier()
    
    # Sidebar for user input
    st.sidebar.title("🚀 Investment Preferences")
    
    # Version info in sidebar
    with st.sidebar.expander("ℹ️ App Information", expanded=False):
        st.markdown(f"**Version:** {APP_VERSION}")
        st.markdown(f"**Build Date:** {APP_BUILD_DATE}")
        st.markdown(f"**Latest Updates:**")
        for update in VERSION_NOTES[APP_VERSION]:
            st.markdown(f"• {update}")
        st.markdown("---")
        st.markdown("**Previous Versions:**")
        for version in sorted(VERSION_NOTES.keys(), reverse=True)[1:]:
            with st.expander(f"Version {version}"):
                for note in VERSION_NOTES[version]:
                    st.markdown(f"• {note}")
    
    # User profile inputs
    user_name = st.sidebar.text_input("Full Name", value="")
    user_email = st.sidebar.text_input("Email Address", value="")
    
    country = st.sidebar.selectbox(
        "Select Country/Market", 
        options=["US", "CANADA", "INDIA"],
        index=0
    )
    
    investment_amount = st.sidebar.number_input(
        "Investment Amount ($)", 
        min_value=100, 
        max_value=1000000, 
        value=10000, 
        step=100
    )
    
    desired_return = st.sidebar.slider(
        "Desired Annual Return (%)", 
        min_value=5, 
        max_value=30, 
        value=12, 
        step=1
    )
    
    risk_tolerance = st.sidebar.selectbox(
        "Risk Tolerance", 
        options=["Conservative", "Moderate", "Aggressive"],
        index=1
    )
    
    notification_frequency = st.sidebar.selectbox(
        "Email Notifications", 
        options=["Daily", "Weekly", "Monthly", "None"],
        index=0
    )
    
    # Email settings (for notifications)
    if notification_frequency != "None":
        st.sidebar.subheader("📧 Email Settings")
        sender_email = st.sidebar.text_input("Your Gmail Address", type="default")
        sender_password = st.sidebar.text_input("Gmail App Password", type="password", 
                                               help="Use Gmail App Password, not regular password")
    
    # Main interface
    st.title(f"🤖 {APP_NAME}")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### Intelligent Investment Recommendations with Machine Learning")
    with col2:
        st.markdown(f"**Version {APP_VERSION}** | {APP_BUILD_DATE}")
    
    # Version badge
    st.markdown(f"""
    <div style="text-align: right; margin-bottom: 20px;">
        <span style="background-color: #28a745; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px;">
            v{APP_VERSION} - Latest
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Analysis", "📈 Portfolio", "🤖 ML Insights", "⚙️ Settings"])
    
    with tab1:
        st.subheader(f"Stock Analysis for {country}")
        
        if st.button("🔍 Analyze Stocks", type="primary"):
            if not user_name or not user_email:
                st.error("Please fill in your name and email address.")
            else:
                # Create user profile
                user_profile = UserProfile(
                    name=user_name,
                    email=user_email,
                    country=country,
                    investment_amount=investment_amount,
                    desired_return=desired_return,
                    risk_tolerance=risk_tolerance,
                    notification_frequency=notification_frequency
                )
                
                # Save user profile
                db.save_user(user_profile)
                
                # Analyze stocks
                with st.spinner("Analyzing stocks with AI/ML models..."):
                    analyses = analyzer.analyze_stocks_for_country(country, user_profile)
                
                if analyses:
                    st.success(f"Analysis complete! Found {len(analyses)} stocks.")
                    
                    # Display top 5 recommendations
                    st.subheader("🏆 Top 5 Investment Recommendations")
                    
                    for i, analysis in enumerate(analyses[:5], 1):
                        fund = analysis['fundamentals']
                        
                        with st.expander(f"{i}. {fund.get('company_name', 'N/A')} ({analysis['symbol']}) - Score: {analysis['score']:.1f}/100"):
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Current Price", f"${fund.get('current_price', 0):.2f}")
                                st.metric("PE Ratio", f"{fund.get('pe_ratio', 0):.2f}")
                                st.metric("ROE", f"{fund.get('roe', 0)*100:.1f}%")
                            
                            with col2:
                                st.metric("ML Prediction", f"{analysis['ml_prediction']:.1f}%")
                                st.metric("Revenue Growth", f"{fund.get('revenue_growth', 0)*100:.1f}%")
                                st.metric("Debt/Equity", f"{fund.get('debt_to_equity', 0):.2f}")
                            
                            with col3:
                                inv_rec = analysis['investment_recommendation']
                                st.metric("Recommended Shares", f"{inv_rec['shares']}")
                                st.metric("Investment Amount", f"${inv_rec['amount']:.2f}")
                                st.metric("Portfolio %", f"{inv_rec['percentage']:.1f}%")
                    
                    # Send email notification if configured
                    if notification_frequency != "None" and sender_email and sender_password:
                        if st.button("📧 Send Email Report"):
                            with st.spinner("Sending email..."):
                                success = emailer.send_daily_recommendations(
                                    user_email, analyses, sender_email, sender_password
                                )
                                if success:
                                    st.success("Email sent successfully!")
                                else:
                                    st.error("Failed to send email. Check your credentials.")
                else:
                    st.error("No data available for analysis.")
    
    with tab2:
        st.subheader("📈 Portfolio Optimization")
        
        if 'analyses' in locals() and analyses:
            # Portfolio allocation chart
            portfolio_data = []
            for analysis in analyses[:10]:  # Top 10
                fund = analysis['fundamentals']
                inv_rec = analysis['investment_recommendation']
                if inv_rec['amount'] > 0:
                    portfolio_data.append({
                        'Symbol': analysis['symbol'],
                        'Company': fund.get('company_name', 'N/A')[:20],
                        'Amount': inv_rec['amount'],
                        'Percentage': inv_rec['percentage'],
                        'Shares': inv_rec['shares']
                    })
            
            if portfolio_data:
                df_portfolio = pd.DataFrame(portfolio_data)
                
                # Pie chart
                fig_pie = px.pie(df_portfolio, values='Amount', names='Symbol', 
                               title="Portfolio Allocation")
                st.plotly_chart(fig_pie, use_container_width=True)
                
                # Portfolio table
                st.subheader("Portfolio Details")
                st.dataframe(df_portfolio, use_container_width=True)
                
                # Portfolio summary
                total_investment = df_portfolio['Amount'].sum()
                st.metric("Total Portfolio Value", f"${total_investment:.2f}")
        else:
            st.info("Run analysis first to see portfolio recommendations.")
    
    with tab3:
        st.subheader("🤖 Machine Learning Insights")
        
        if 'analyses' in locals() and analyses:
            # Feature importance
            feature_importance = analyzer.ml_predictor.get_feature_importance()
            if feature_importance:
                importance_df = pd.DataFrame(
                    list(feature_importance.items()), 
                    columns=['Feature', 'Importance']
                ).sort_values('Importance', ascending=False)
                
                fig_importance = px.bar(
                    importance_df, 
                    x='Importance', 
                    y='Feature', 
                    orientation='h',
                    title="ML Model Feature Importance"
                )
                st.plotly_chart(fig_importance, use_container_width=True)
            
            # Prediction vs Score scatter plot
            ml_predictions = [a['ml_prediction'] for a in analyses]
            scores = [a['score'] for a in analyses]
            symbols = [a['symbol'] for a in analyses]
            
            fig_scatter = go.Figure(data=go.Scatter(
                x=ml_predictions,
                y=scores,
                mode='markers+text',
                text=symbols,
                textposition="top center",
                marker=dict(size=10, opacity=0.6)
            ))
            fig_scatter.update_layout(
                title="ML Prediction vs Investment Score",
                xaxis_title="ML Predicted Return (%)",
                yaxis_title="Investment Score"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("Run analysis first to see ML insights.")
    
    with tab4:
        st.subheader("⚙️ Application Settings")
        
        st.markdown("### 📧 Email Notification Setup")
        st.markdown("""
        To receive daily email notifications:
        1. Enable 2-factor authentication on your Gmail account
        2. Generate an App Password: Google Account → Security → App passwords
        3. Use the App Password (not your regular password) in the sidebar
        """)
        
        st.markdown("### 🤖 ML Model Information")
        st.markdown("""
        The application uses Gradient Boosting Regression to predict stock returns based on:
        - Financial ratios (PE, PB, ROE, etc.)
        - Growth metrics (Revenue growth, earnings growth)
        - Technical indicators (Momentum, volatility)
        - Market data (Volume ratios, beta)
        """)
        
        st.markdown("### 📊 Supported Markets")
        st.markdown("""
        - **US**: Top 20 S&P 500 stocks
        - **Canada**: TSX top 15 stocks  
        - **India**: NIFTY top 15 stocks
        """)

if __name__ == "__main__":
    main()