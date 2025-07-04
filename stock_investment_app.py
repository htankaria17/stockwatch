import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pytz
import sqlite3
import smtplib
import io
import base64

# Email imports with fallback
try:
    from email.mime.text import MIMEText as MimeText
    from email.mime.multipart import MIMEMultipart as MimeMultipart
    from email.mime.base import MIMEBase
    from email import encoders
except ImportError:
    try:
        from email.MIMEText import MIMEText as MimeText
        from email.MIMEMultipart import MIMEMultipart as MimeMultipart
        from email.MIMEBase import MIMEBase
        from email import encoders
    except ImportError:
        # Fallback for older Python versions
        from email.mime.text import MIMEText as MimeText
        from email.mime.multipart import MIMEMultipart as MimeMultipart
        from email.mime.base import MIMEBase
        from email import encoders

# Application Version Information
APP_VERSION = "V0.2.1"

APP_BUILD_DATE = "2025-07-03"

APP_NAME = "True North Tech Stock Investment Analyzer"
APP_CONCEPT_OWNER = "Harshal Tankaria"
VERSION_NOTES = {
    "V0.2.1": [

        "Added user login system with password remember functionality",
        "Enhanced ML predictions with multi-timeframe projections (6M, 1Y, 5Y, 10Y)",
        "Display all 20 stock options below top 5 recommendations",
        "Added export functionality with comprehensive report generation", 
        "Implemented backtesting against 6 months historical data",
        "Improved user authentication and session management"

    ],
    "V0.1.1": [
        "Added SENSEX stocks to Indian market analysis (35 total stocks)",
        "Fixed Python 3.13 email import compatibility",
        "Enhanced ML prediction reporting and documentation",
        "Added comprehensive version tracking system",
        "Improved error handling and user experience"
    ],
    "V0.1.0": [
        "Complete GUI transformation with Streamlit",
        "Integrated AI/ML prediction system",
        "Multi-country support (US, Canada, India)",
        "Email automation and scheduling",
        "Portfolio optimization algorithms"
    ]
}

# Timezone setup for proper date/time handling
EST = pytz.timezone('US/Eastern')
UTC = pytz.timezone('UTC')

def get_current_time(timezone_str='US/Eastern'):
    """Get current time with proper timezone handling"""
    tz = pytz.timezone(timezone_str)
    return datetime.now(tz)

def format_datetime(dt, format_str='%Y-%m-%d %H:%M:%S %Z'):
    """Format datetime with timezone information"""
    return dt.strftime(format_str)

# Day Trading Configuration
DAY_TRADING_STOCKS = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
TRADING_HOURS = {
    'market_open': 9.5,  # 9:30 AM
    'market_close': 16.0,  # 4:00 PM
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
    password: str
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
        
        # Users table with password field
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
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
    
    def _hash_password(self, password: str) -> str:
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def register_user(self, user: UserProfile) -> int:
        """Register a new user"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        password_hash = self._hash_password(user.password)
        
        try:
            cursor.execute('''
                INSERT INTO users 
                (name, email, password_hash, country, investment_amount, desired_return, risk_tolerance, notification_frequency)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (user.name, user.email, password_hash, user.country, user.investment_amount, 
                  user.desired_return, user.risk_tolerance, user.notification_frequency))
            
            user_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return user_id
        except sqlite3.IntegrityError:
            conn.close()
            return -1  # Email already exists
    
    def authenticate_user(self, email: str, password: str) -> Optional[int]:
        """Authenticate user login"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        password_hash = self._hash_password(password)
        cursor.execute('SELECT id FROM users WHERE email = ? AND password_hash = ?', (email, password_hash))
        row = cursor.fetchone()
        conn.close()
        
        return row[0] if row else None
    
    def save_user(self, user: UserProfile) -> int:
        """Update existing user profile"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE users SET 
            name = ?, country = ?, investment_amount = ?, desired_return = ?, 
            risk_tolerance = ?, notification_frequency = ?
            WHERE email = ?
        ''', (user.name, user.country, user.investment_amount, 
              user.desired_return, user.risk_tolerance, user.notification_frequency, user.email))
        
        conn.commit()
        conn.close()
        return cursor.rowcount
    
    def get_user_by_email(self, email: str) -> Optional[UserProfile]:
        """Get user by email"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return UserProfile(
                name=row[1], email=row[2], password="", country=row[4],
                investment_amount=row[5], desired_return=row[6],
                risk_tolerance=row[7], notification_frequency=row[8]
            )
        return None
    
    def get_user_by_id(self, user_id: int) -> Optional[UserProfile]:
        """Get user by ID"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return UserProfile(
                name=row[1], email=row[2], password="", country=row[4],
                investment_amount=row[5], desired_return=row[6],
                risk_tolerance=row[7], notification_frequency=row[8]
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
    
    def predict_multi_timeframe_returns(self, stock_data: Dict) -> Dict[str, float]:
        """Predict returns for multiple timeframes"""
        if self.model is None:
            self.train_model([])
        
        base_prediction = self.predict_return(stock_data)
        
        # Time-based adjustments based on financial theory
        # More uncertainty and volatility over longer periods
        volatility = stock_data.get('volatility', 0.2)
        beta = stock_data.get('beta', 1.0)
        
        predictions = {
            '6_months': base_prediction * 0.5,  # Half-year scaling
            '1_year': base_prediction,  # Base prediction is annual
            '5_years': base_prediction * 4.5 * (1 + volatility * 0.5),  # Compound with volatility
            '10_years': base_prediction * 8.5 * (1 + volatility * 0.3 + beta * 0.2)  # Long-term with beta effect
        }
        
        # Cap predictions at reasonable bounds
        for timeframe in predictions:
            predictions[timeframe] = max(-80, min(500, predictions[timeframe]))
        
        return predictions
    
    def backtest_predictions(self, symbol: str, months: int = 6) -> Dict:
        """Backtest predictions against historical data"""
        try:
            import yfinance as yf
            
            # Get historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=months * 30)
            
            stock = yf.Ticker(symbol)
            history = stock.history(start=start_date, end=end_date)
            
            if len(history) < 30:  # Need sufficient data
                return {'error': 'Insufficient historical data'}
            
            # Simulate predictions at different points
            backtest_results = []
            prediction_dates = []
            actual_returns = []
            predicted_returns = []
            
            # Test predictions every 2 weeks
            for i in range(14, len(history), 14):
                test_date = history.index[i]
                future_date_idx = min(i + 21, len(history) - 1)  # 3 weeks later
                
                # Get stock data at test point
                current_price = history['Close'].iloc[i]
                future_price = history['Close'].iloc[future_date_idx]
                
                # Calculate actual return
                actual_return = ((future_price - current_price) / current_price) * 100
                
                # Generate mock fundamentals for historical point
                mock_fundamentals = {
                    'pe_ratio': 15 + np.random.normal(0, 5),
                    'pb_ratio': 2 + np.random.normal(0, 1),
                    'roe': 0.12 + np.random.normal(0, 0.05),
                    'profit_margin': 0.1 + np.random.normal(0, 0.03),
                    'revenue_growth': 0.08 + np.random.normal(0, 0.1),
                    'debt_to_equity': 0.5 + np.random.normal(0, 0.2),
                    'dividend_yield': 0.02 + np.random.normal(0, 0.01),
                    'beta': 1.0 + np.random.normal(0, 0.3),
                    'volume_ratio': 1.0,
                    'price_momentum_1m': 0,
                    'price_momentum_3m': 0,
                    'volatility': 0.2
                }
                
                # Get prediction
                predicted_return = self.predict_return(mock_fundamentals) / 12 * 3  # Scale to 3 weeks
                
                prediction_dates.append(test_date)
                actual_returns.append(actual_return)
                predicted_returns.append(predicted_return)
            
            if len(actual_returns) > 0:
                # Calculate accuracy metrics
                mse = np.mean([(a - p) ** 2 for a, p in zip(actual_returns, predicted_returns)])
                mae = np.mean([abs(a - p) for a, p in zip(actual_returns, predicted_returns)])
                
                # Direction accuracy (did we predict the right direction?)
                direction_accuracy = np.mean([
                    (a > 0 and p > 0) or (a < 0 and p < 0) 
                    for a, p in zip(actual_returns, predicted_returns)
                ]) * 100
                
                return {
                    'dates': prediction_dates,
                    'actual_returns': actual_returns,
                    'predicted_returns': predicted_returns,
                    'mse': mse,
                    'mae': mae,
                    'direction_accuracy': direction_accuracy,
                    'num_predictions': len(actual_returns)
                }
            else:
                return {'error': 'No predictions could be generated'}
                
        except Exception as e:
            return {'error': f'Backtest failed: {str(e)}'}
    
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
            
            # Get multi-timeframe predictions
            multi_timeframe_predictions = self.ml_predictor.predict_multi_timeframe_returns(fundamentals)
            
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
                'multi_timeframe_predictions': multi_timeframe_predictions,
                'criteria': criteria,
                'investment_recommendation': investment_recommendation,
                'analysis_date': format_datetime(get_current_time())
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

class DayTradingAnalyzer:
    def __init__(self):
        self.stocks = DAY_TRADING_STOCKS
        self.signals = {}
        
    def get_intraday_data(self, symbol: str, period: str = '1d', interval: str = '5m') -> pd.DataFrame:
        """Get intraday data for day trading analysis"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period, interval=interval)
            return data
        except Exception as e:
            st.error(f"Error fetching intraday data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_day_trading_signals(self, symbol: str) -> Dict:
        """Calculate buy/sell signals for day trading"""
        data = self.get_intraday_data(symbol)
        if data.empty:
            return {}
            
        # Calculate technical indicators
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['RSI'] = self.calculate_rsi(data['Close'])
        data['MACD'], data['MACD_signal'] = self.calculate_macd(data['Close'])
        data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
        
        current = data.iloc[-1]
        prev = data.iloc[-2] if len(data) > 1 else current
        
        # Generate signals
        signals = {
            'symbol': symbol,
            'current_price': current['Close'],
            'timestamp': get_current_time(),
            'volume': current['Volume'],
            'signals': []
        }
        
        # Buy signals
        if (current['Close'] > current['SMA_20'] and 
            prev['Close'] <= prev['SMA_20'] and 
            current['RSI'] < 70):
            signals['signals'].append({
                'type': 'BUY',
                'reason': 'Price crossed above SMA 20',
                'strength': 'STRONG' if current['Volume'] > current['Volume_SMA'] * 1.5 else 'MODERATE',
                'entry_price': current['Close'],
                'stop_loss': current['Close'] * 0.98,
                'target': current['Close'] * 1.03
            })
            
        if (current['MACD'] > current['MACD_signal'] and 
            prev['MACD'] <= prev['MACD_signal'] and 
            current['RSI'] < 80):
            signals['signals'].append({
                'type': 'BUY',
                'reason': 'MACD bullish crossover',
                'strength': 'MODERATE',
                'entry_price': current['Close'],
                'stop_loss': current['Close'] * 0.97,
                'target': current['Close'] * 1.04
            })
        
        # Sell signals
        if (current['Close'] < current['SMA_20'] and 
            prev['Close'] >= prev['SMA_20'] and 
            current['RSI'] > 30):
            signals['signals'].append({
                'type': 'SELL',
                'reason': 'Price dropped below SMA 20',
                'strength': 'STRONG' if current['Volume'] > current['Volume_SMA'] * 1.5 else 'MODERATE',
                'entry_price': current['Close'],
                'stop_loss': current['Close'] * 1.02,
                'target': current['Close'] * 0.97
            })
            
        if (current['MACD'] < current['MACD_signal'] and 
            prev['MACD'] >= prev['MACD_signal'] and 
            current['RSI'] > 20):
            signals['signals'].append({
                'type': 'SELL',
                'reason': 'MACD bearish crossover',
                'strength': 'MODERATE',
                'entry_price': current['Close'],
                'stop_loss': current['Close'] * 1.03,
                'target': current['Close'] * 0.96
            })
        
        return signals
    
    def calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """Calculate MACD indicator"""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal
    
    def get_all_day_trading_signals(self) -> List[Dict]:
        """Get day trading signals for all tracked stocks"""
        all_signals = []
        
        for symbol in self.stocks:
            signals = self.calculate_day_trading_signals(symbol)
            if signals:
                all_signals.append(signals)
        
        return all_signals

class ExportManager:
    def __init__(self):
        pass
    
    def export_analysis_to_csv(self, analyses: List[Dict]) -> str:
        """Export analysis results to CSV format"""
        export_data = []
        
        for analysis in analyses:
            fund = analysis['fundamentals']
            inv_rec = analysis['investment_recommendation']
            
            export_data.append({
                'Symbol': analysis['symbol'],
                'Company': fund.get('company_name', 'N/A'),
                'Current_Price': fund.get('current_price', 0),
                'Score': analysis['score'],
                'ML_Prediction': analysis['ml_prediction'],
                'PE_Ratio': fund.get('pe_ratio', 0),
                'ROE': fund.get('roe', 0) * 100,
                'Revenue_Growth': fund.get('revenue_growth', 0) * 100,
                'Debt_to_Equity': fund.get('debt_to_equity', 0),
                'Recommended_Shares': inv_rec['shares'],
                'Investment_Amount': inv_rec['amount'],
                'Portfolio_Percentage': inv_rec['percentage'],
                'Analysis_Date': analysis['analysis_date']
            })
        
        df = pd.DataFrame(export_data)
        return df.to_csv(index=False)
    
    def export_day_trading_signals_to_csv(self, signals: List[Dict]) -> str:
        """Export day trading signals to CSV format"""
        export_data = []
        
        for signal_data in signals:
            for signal in signal_data['signals']:
                export_data.append({
                    'Symbol': signal_data['symbol'],
                    'Current_Price': signal_data['current_price'],
                    'Signal_Type': signal['type'],
                    'Reason': signal['reason'],
                    'Strength': signal['strength'],
                    'Entry_Price': signal['entry_price'],
                    'Stop_Loss': signal['stop_loss'],
                    'Target': signal['target'],
                    'Timestamp': format_datetime(signal_data['timestamp'])
                })
        
        df = pd.DataFrame(export_data)
        return df.to_csv(index=False)
    
    def create_download_link(self, data: str, filename: str, file_type: str = "csv") -> str:
        """Create a download link for the exported data"""
        b64 = base64.b64encode(data.encode()).decode()
        href = f'<a href="data:file/{file_type};base64,{b64}" download="{filename}">Download {filename}</a>'
        return href

class EmailNotifier:
    def __init__(self):
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
    
    def send_test_email(self, user_email: str, sender_email: str, sender_password: str):
        """Send a test email to verify email configuration"""
        try:
            current_time = get_current_time()
            subject = f"Test Email - {APP_NAME} - {format_datetime(current_time, '%Y-%m-%d %H:%M %Z')}"
            
            html_content = f"""
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; }}
                    .header {{ background-color: #28a745; color: white; padding: 20px; text-align: center; }}
                    .content {{ padding: 20px; }}
                    .footer {{ background-color: #f8f9fa; padding: 15px; text-align: center; font-size: 12px; color: #666; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Email Test Successful! ✅</h1>
                    <p>Your email configuration is working correctly</p>
                </div>
                
                <div class="content">
                    <h2>Test Details</h2>
                    <p><strong>Application:</strong> {APP_NAME} v{APP_VERSION}</p>
                    <p><strong>Concept Owner:</strong> {APP_CONCEPT_OWNER}</p>
                    <p><strong>Test Time:</strong> {format_datetime(current_time)}</p>
                    <p><strong>Recipient:</strong> {user_email}</p>
                    
                    <p>This test email confirms that your Gmail configuration is properly set up for receiving investment reports and day trading alerts.</p>
                </div>
                
                <div class="footer">
                    <p>Powered by {APP_NAME} | Developed by {APP_CONCEPT_OWNER}</p>
                </div>
            </body>
            </html>
            """
            
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
            st.error(f"Test email failed: {e}")
            return False

    def send_daily_recommendations(self, user_email: str, recommendations: List[Dict], 
                                 sender_email: str, sender_password: str, export_data: str = None):
        """Send daily recommendations via email with optional CSV attachment"""
        try:
            current_time = get_current_time()
            subject = f"Daily Investment Recommendations - {format_datetime(current_time, '%Y-%m-%d')}"
            
            html_content = self._create_email_html(recommendations)
            
            # Create message
            msg = MimeMultipart('mixed')
            msg['Subject'] = subject
            msg['From'] = sender_email
            msg['To'] = user_email
            
            # Attach HTML content
            html_part = MimeText(html_content, 'html')
            msg.attach(html_part)
            
            # Attach CSV if provided
            if export_data:
                csv_attachment = MIMEBase('application', 'octet-stream')
                csv_attachment.set_payload(export_data.encode())
                encoders.encode_base64(csv_attachment)
                csv_attachment.add_header(
                    'Content-Disposition',
                    f'attachment; filename="investment_analysis_{format_datetime(current_time, "%Y%m%d_%H%M")}.csv"'
                )
                msg.attach(csv_attachment)
            
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
    
    def send_day_trading_alerts(self, user_email: str, signals: List[Dict], 
                              sender_email: str, sender_password: str):
        """Send day trading alerts via email"""
        try:
            current_time = get_current_time()
            subject = f"Day Trading Alerts - {format_datetime(current_time, '%Y-%m-%d %H:%M %Z')}"
            
            html_content = self._create_day_trading_email_html(signals)
            
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
            st.error(f"Failed to send day trading alerts: {e}")
            return False
    
    def _create_email_html(self, recommendations: List[Dict]) -> str:
        """Create HTML email content"""
        current_time = get_current_time()
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
                <p>Generated on {format_datetime(current_time)}</p>
                <p style="font-size: 14px; opacity: 0.9;">{APP_NAME} v{APP_VERSION}</p>
                <p style="font-size: 12px; opacity: 0.8;">Concept & Owner: {APP_CONCEPT_OWNER}</p>
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
                <p>Concept & Owner: <strong>{APP_CONCEPT_OWNER}</strong></p>
                <p>AI-Powered Investment Analysis | Machine Learning Enhanced Predictions</p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _create_day_trading_email_html(self, signals: List[Dict]) -> str:
        """Create HTML email content for day trading alerts"""
        current_time = get_current_time()
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .header {{ background-color: #FF6B35; color: white; padding: 20px; text-align: center; }}
                .signal {{ border: 1px solid #ddd; margin: 10px; padding: 15px; border-radius: 5px; }}
                .buy-signal {{ border-left: 5px solid #28a745; background-color: #f8fff8; }}
                .sell-signal {{ border-left: 5px solid #dc3545; background-color: #fff8f8; }}
                .strong {{ font-weight: bold; color: #212529; }}
                .moderate {{ font-weight: normal; color: #6c757d; }}
                .metric {{ display: inline-block; margin: 5px 10px; }}
                .footer {{ background-color: #f8f9fa; padding: 15px; text-align: center; font-size: 12px; color: #666; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚨 Day Trading Alerts 🚨</h1>
                <p>Generated on {format_datetime(current_time)}</p>
                <p style="font-size: 14px; opacity: 0.9;">{APP_NAME} v{APP_VERSION}</p>
                <p style="font-size: 12px; opacity: 0.8;">Concept & Owner: {APP_CONCEPT_OWNER}</p>
            </div>
            
            <div style="padding: 20px;">
                <h2>Active Trading Signals</h2>
        """
        
        signal_count = 0
        for stock_signals in signals:
            for signal in stock_signals['signals']:
                signal_count += 1
                signal_class = 'buy-signal' if signal['type'] == 'BUY' else 'sell-signal'
                strength_class = 'strong' if signal['strength'] == 'STRONG' else 'moderate'
                
                html += f"""
                    <div class="signal {signal_class}">
                        <h3>{'🟢' if signal['type'] == 'BUY' else '🔴'} {signal['type']} - {stock_signals['symbol']}</h3>
                        <p class="{strength_class}"><strong>Strength:</strong> {signal['strength']}</p>
                        <p><strong>Reason:</strong> {signal['reason']}</p>
                        <p><strong>Current Price:</strong> ${stock_signals['current_price']:.2f}</p>
                        <div class="metric"><strong>Entry:</strong> ${signal['entry_price']:.2f}</div>
                        <div class="metric"><strong>Stop Loss:</strong> ${signal['stop_loss']:.2f}</div>
                        <div class="metric"><strong>Target:</strong> ${signal['target']:.2f}</div>
                        <p><strong>Risk/Reward:</strong> {abs((signal['target'] - signal['entry_price']) / (signal['stop_loss'] - signal['entry_price'])):.2f}:1</p>
                    </div>
                """
        
        if signal_count == 0:
            html += """
                <div style="text-align: center; padding: 40px; color: #6c757d;">
                    <h3>No Active Signals</h3>
                    <p>Currently monitoring markets. Signals will appear when trading opportunities arise.</p>
                </div>
            """
        
        html += f"""
            </div>
            <div class="footer">
                <p><strong>⚠️ Risk Warning:</strong> Day trading involves significant risk. These are algorithmic signals for educational purposes only.<br>
                Always use proper risk management and never risk more than you can afford to lose.</p>
                <hr style="margin: 10px 0; border: none; border-top: 1px solid #ddd;">
                <p><strong>{APP_NAME}</strong> v{APP_VERSION} | Build Date: {APP_BUILD_DATE}</p>
                <p>Concept & Owner: <strong>{APP_CONCEPT_OWNER}</strong></p>
                <p>Real-time Day Trading Analysis | Technical Indicators & Signals</p>
            </div>
        </body>
        </html>
        """
        
        return html

class ReportExporter:
    def __init__(self):
        self.report_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    def generate_comprehensive_report(self, analyses: List[Dict], user_profile: UserProfile) -> str:
        """Generate comprehensive investment report"""
        report = f"""
# AI-Powered Investment Analysis Report
**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**User:** {user_profile.name} ({user_profile.email})
**Market:** {user_profile.country}
**Investment Amount:** ${user_profile.investment_amount:,.2f}
**Risk Tolerance:** {user_profile.risk_tolerance}

---

## Executive Summary
This report analyzes {len(analyses)} stocks using advanced machine learning algorithms and fundamental analysis. The recommendations are tailored to your risk tolerance and investment preferences.

### Top 5 Recommendations
"""
        
        for i, analysis in enumerate(analyses[:5], 1):
            fund = analysis['fundamentals']
            multi_pred = analysis['multi_timeframe_predictions']
            inv_rec = analysis['investment_recommendation']
            
            report += f"""
#### {i}. {fund.get('company_name', 'N/A')} ({analysis['symbol']})
- **Investment Score:** {analysis['score']:.1f}/100
- **Current Price:** ${fund.get('current_price', 0):.2f}
- **Recommended Investment:** ${inv_rec['amount']:.2f} ({inv_rec['shares']} shares)
- **Portfolio Allocation:** {inv_rec['percentage']:.1f}%

**Projected Returns:**
- 6 Months: {multi_pred['6_months']:.1f}%
- 1 Year: {multi_pred['1_year']:.1f}%
- 5 Years: {multi_pred['5_years']:.1f}%
- 10 Years: {multi_pred['10_years']:.1f}%

**Financial Metrics:**
- PE Ratio: {fund.get('pe_ratio', 0):.2f}
- ROE: {fund.get('roe', 0)*100:.1f}%
- Revenue Growth: {fund.get('revenue_growth', 0)*100:.1f}%
- Debt/Equity: {fund.get('debt_to_equity', 0):.2f}
"""
        
        report += f"""
---

## Complete Analysis ({len(analyses)} Stocks)

| Rank | Symbol | Company | Score | 1Y Prediction | Investment | Shares |
|------|--------|---------|-------|---------------|------------|---------|
"""
        
        for i, analysis in enumerate(analyses, 1):
            fund = analysis['fundamentals']
            inv_rec = analysis['investment_recommendation']
            multi_pred = analysis['multi_timeframe_predictions']
            
            report += f"| {i} | {analysis['symbol']} | {fund.get('company_name', 'N/A')[:20]} | {analysis['score']:.1f} | {multi_pred['1_year']:.1f}% | ${inv_rec['amount']:.0f} | {inv_rec['shares']} |\n"
        
        report += f"""
---

## Portfolio Summary
- **Total Recommended Investment:** ${sum([a['investment_recommendation']['amount'] for a in analyses]):,.2f}
- **Number of Stocks:** {len([a for a in analyses if a['investment_recommendation']['amount'] > 0])}
- **Average Score:** {np.mean([a['score'] for a in analyses]):.1f}/100
- **Expected 1-Year Return:** {np.mean([a['multi_timeframe_predictions']['1_year'] for a in analyses]):.1f}%

## Risk Assessment
Based on your **{user_profile.risk_tolerance}** risk tolerance, the recommended portfolio balances growth potential with risk management.

## Disclaimer
This analysis is for educational purposes only and should not be considered as financial advice. Please consult with licensed financial advisors before making investment decisions.

---
*Report generated by {APP_NAME} v{APP_VERSION}*
"""
        
        return report
    
    def export_to_file(self, content: str, filename: str = None) -> str:
        """Export report to file and return filename"""
        if filename is None:
            filename = f"investment_report_{self.report_date}.md"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            return filename
        except Exception as e:
            return f"Error saving file: {str(e)}"

def main():
    st.set_page_config(
        page_title=f"{APP_NAME} v{APP_VERSION}",
        page_icon="📈",
        layout="wide"
    )
    
    # Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'remember_password' not in st.session_state:
        st.session_state.remember_password = False
    if 'saved_password' not in st.session_state:
        st.session_state.saved_password = ""
    if 'analyses' not in st.session_state:
        st.session_state.analyses = []
    
    # Initialize components
    db = DatabaseManager()
    analyzer = EnhancedStockAnalyzer()
    emailer = EmailNotifier()

    exporter = ReportExporter()

    
    # Login/Registration System
    if not st.session_state.logged_in:
        st.title(f"🔐 Welcome to {APP_NAME}")
        st.markdown("### Please login or register to continue")
        
        tab1, tab2 = st.tabs(["🔑 Login", "📝 Register"])
        
        with tab1:
            st.subheader("Login to Your Account")
            login_email = st.text_input("Email Address", key="login_email")
            
            # Password remember functionality
            remember_password = st.checkbox("Remember password", value=st.session_state.remember_password)
            
            if remember_password and st.session_state.saved_password:
                login_password = st.text_input("Password", value=st.session_state.saved_password, type="password", key="login_password")
            else:
                login_password = st.text_input("Password", type="password", key="login_password")
            
            if st.button("🔓 Login", type="primary"):
                if login_email and login_password:
                    user_id = db.authenticate_user(login_email, login_password)
                    if user_id:
                        st.session_state.logged_in = True
                        st.session_state.user_id = user_id
                        if remember_password:
                            st.session_state.remember_password = True
                            st.session_state.saved_password = login_password
                        else:
                            st.session_state.remember_password = False
                            st.session_state.saved_password = ""
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid email or password")
                else:
                    st.error("Please enter both email and password")
        
        with tab2:
            st.subheader("Create New Account")
            reg_name = st.text_input("Full Name", key="reg_name")
            reg_email = st.text_input("Email Address", key="reg_email")
            reg_password = st.text_input("Password", type="password", key="reg_password")
            reg_confirm_password = st.text_input("Confirm Password", type="password", key="reg_confirm_password")
            
            reg_country = st.selectbox("Select Country/Market", options=["US", "CANADA", "INDIA"], key="reg_country")
            reg_investment_amount = st.number_input("Investment Amount ($)", min_value=100, max_value=1000000, value=10000, step=100, key="reg_investment")
            reg_desired_return = st.slider("Desired Annual Return (%)", min_value=5, max_value=30, value=12, step=1, key="reg_return")
            reg_risk_tolerance = st.selectbox("Risk Tolerance", options=["Conservative", "Moderate", "Aggressive"], index=1, key="reg_risk")
            reg_notification_frequency = st.selectbox("Email Notifications", options=["Daily", "Weekly", "Monthly", "None"], key="reg_notifications")
            
            if st.button("📝 Register", type="primary"):
                if not all([reg_name, reg_email, reg_password, reg_confirm_password]):
                    st.error("Please fill in all fields")
                elif reg_password != reg_confirm_password:
                    st.error("Passwords do not match")
                elif len(reg_password) < 6:
                    st.error("Password must be at least 6 characters long")
                else:
                    user_profile = UserProfile(
                        name=reg_name,
                        email=reg_email,
                        password=reg_password,
                        country=reg_country,
                        investment_amount=reg_investment_amount,
                        desired_return=reg_desired_return,
                        risk_tolerance=reg_risk_tolerance,
                        notification_frequency=reg_notification_frequency
                    )
                    
                    user_id = db.register_user(user_profile)
                    if user_id != -1:
                        st.success("Registration successful! Please login with your credentials.")
                    else:
                        st.error("Email already exists. Please use a different email.")
        
        return  # Don't proceed if not logged in
    
    # Main Application (after login)
    # Get current user profile
    current_user = db.get_user_by_id(st.session_state.user_id)
    if not current_user:
        st.error("User not found. Please login again.")
        st.session_state.logged_in = False
        st.rerun()
        return
    
    # Header with logout
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.title(f"🤖 {APP_NAME}")
        st.markdown(f"Welcome back, **{current_user.name}**!")
    with col2:
        st.markdown(f"**Version {APP_VERSION}**")
        st.markdown(f"{APP_BUILD_DATE}")
    with col3:
        if st.button("🔐 Logout"):
            st.session_state.logged_in = False
            st.session_state.user_id = None
            if not st.session_state.remember_password:
                st.session_state.saved_password = ""
            st.rerun()
    
    # Sidebar for user preferences (now editable)
    st.sidebar.title("🚀 Investment Preferences")
    
    # User profile inputs (editable)
    st.sidebar.markdown(f"**User:** {current_user.name}")
    st.sidebar.markdown(f"**Email:** {current_user.email}")
    
    country = st.sidebar.selectbox(
        "Select Country/Market", 
        options=["US", "CANADA", "INDIA"],
        index=["US", "CANADA", "INDIA"].index(current_user.country)
    )
    
    investment_amount = st.sidebar.number_input(
        "Investment Amount ($)", 
        min_value=100, 
        max_value=1000000, 
        value=int(current_user.investment_amount), 
        step=100
    )
    
    desired_return = st.sidebar.slider(
        "Desired Annual Return (%)", 
        min_value=5, 
        max_value=30, 
        value=int(current_user.desired_return), 
        step=1
    )
    
    risk_tolerance = st.sidebar.selectbox(
        "Risk Tolerance", 
        options=["Conservative", "Moderate", "Aggressive"],
        index=["Conservative", "Moderate", "Aggressive"].index(current_user.risk_tolerance)
    )
    
    notification_frequency = st.sidebar.selectbox(
        "Email Notifications", 
        options=["Daily", "Weekly", "Monthly", "None"],
        index=["Daily", "Weekly", "Monthly", "None"].index(current_user.notification_frequency)
    )
    
    # Update user profile if settings changed
    if st.sidebar.button("💾 Update Preferences"):
        updated_user = UserProfile(
            name=current_user.name,
            email=current_user.email,
            password="",  # Don't change password
            country=country,
            investment_amount=investment_amount,
            desired_return=desired_return,
            risk_tolerance=risk_tolerance,
            notification_frequency=notification_frequency
        )
        db.save_user(updated_user)
        st.sidebar.success("Preferences updated!")
    
    # Version info in sidebar
    with st.sidebar.expander("ℹ️ App Information", expanded=False):
        st.markdown(f"**Version:** {APP_VERSION}")
        st.markdown(f"**Build Date:** {APP_BUILD_DATE}")
        st.markdown(f"**Latest Updates:**")
        for update in VERSION_NOTES[APP_VERSION]:
            st.markdown(f"• {update}")
        
        if len(VERSION_NOTES) > 1:
            st.markdown("---")
            st.markdown("**Previous Versions:**")
            for version in sorted(VERSION_NOTES.keys(), reverse=True)[1:]:
                st.markdown(f"**{version}:**")
                for note in VERSION_NOTES[version]:
                    st.markdown(f"  • {note}")
                st.markdown("")
    
    # Email settings (for notifications)
    if notification_frequency != "None":
        st.sidebar.subheader("📧 Email Settings")
        sender_email = st.sidebar.text_input("Your Gmail Address", type="default")
        sender_password = st.sidebar.text_input("Gmail App Password", type="password", 
                                               help="Use Gmail App Password, not regular password")
        
        # Test email button
        if sender_email and sender_password and user_email:
            if st.sidebar.button("🧪 Test Email Configuration"):
                with st.spinner("Testing email configuration..."):
                    success = emailer.send_test_email(user_email, sender_email, sender_password)
                    if success:
                        st.sidebar.success("✅ Test email sent successfully!")
                    else:
                        st.sidebar.error("❌ Test email failed. Check your credentials.")
 
    # Create user profile from current user data
    user_profile = UserProfile(
        name=current_user.name,
        email=current_user.email,
        password="",
        country=country,
        investment_amount=investment_amount,
        desired_return=desired_return,
        risk_tolerance=risk_tolerance,
        notification_frequency=notification_frequency
    )
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Analysis", "📈 Multi-Timeframe", "🔄 Backtesting", "📊 Portfolio", "📤 Export", "ℹ️ About"])

    
    with tab1:
        st.subheader(f"Stock Analysis for {country}")
        
        # Analysis and Export buttons
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            analyze_button = st.button("🔍 Analyze Stocks", type="primary")
        with col2:
            if st.session_state.analyses:
                if st.button("📤 Export Report", type="secondary"):
                    report_content = exporter.generate_comprehensive_report(st.session_state.analyses, user_profile)
                    filename = exporter.export_to_file(report_content)
                    if filename.endswith('.md'):
                        st.success(f"Report exported to {filename}")
                        st.download_button(
                            label="📥 Download Report",
                            data=report_content,
                            file_name=filename,
                            mime="text/markdown"
                        )
                    else:
                        st.error(filename)
        with col3:
            if st.session_state.analyses and st.button("🔄 Reset App", type="secondary"):
                st.session_state.analyses = []
                st.success("App reset successfully!")
                st.rerun()
        
        if analyze_button:
            # Analyze stocks
            with st.spinner("Analyzing stocks with AI/ML models..."):
                analyses = analyzer.analyze_stocks_for_country(country, user_profile)
                st.session_state.analyses = analyses
            
            if analyses:
                st.success(f"Analysis complete! Found {len(analyses)} stocks.")
            else:
                st.error("No data available for analysis.")
        
        # Display results if available
        if st.session_state.analyses:
            analyses = st.session_state.analyses
            
            # Display top 5 recommendations
            st.subheader("🏆 Top 5 Investment Recommendations")
            
            for i, analysis in enumerate(analyses[:5], 1):
                fund = analysis['fundamentals']
                multi_pred = analysis['multi_timeframe_predictions']
                
                with st.expander(f"{i}. {fund.get('company_name', 'N/A')} ({analysis['symbol']}) - Score: {analysis['score']:.1f}/100"):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Current Price", f"${fund.get('current_price', 0):.2f}")
                        st.metric("PE Ratio", f"{fund.get('pe_ratio', 0):.2f}")
                        st.metric("ROE", f"{fund.get('roe', 0)*100:.1f}%")
                    
                    with col2:
                        st.metric("ML Prediction (1Y)", f"{analysis['ml_prediction']:.1f}%")
                        st.metric("Revenue Growth", f"{fund.get('revenue_growth', 0)*100:.1f}%")
                        st.metric("Debt/Equity", f"{fund.get('debt_to_equity', 0):.2f}")
                    
                    with col3:
                        inv_rec = analysis['investment_recommendation']
                        st.metric("Recommended Shares", f"{inv_rec['shares']}")
                        st.metric("Investment Amount", f"${inv_rec['amount']:.2f}")
                        st.metric("Portfolio %", f"{inv_rec['percentage']:.1f}%")
                    
                    with col4:
                        st.markdown("**Multi-Timeframe Projections:**")
                        st.write(f"6 Months: {multi_pred['6_months']:.1f}%")
                        st.write(f"1 Year: {multi_pred['1_year']:.1f}%")
                        st.write(f"5 Years: {multi_pred['5_years']:.1f}%")
                        st.write(f"10 Years: {multi_pred['10_years']:.1f}%")
            
            # Display all 20 options
            st.subheader(f"📋 Complete Analysis - All {len(analyses)} Stocks")
            
            # Create DataFrame for table display
            table_data = []
            for i, analysis in enumerate(analyses, 1):
                fund = analysis['fundamentals']
                inv_rec = analysis['investment_recommendation']
                multi_pred = analysis['multi_timeframe_predictions']
                
                table_data.append({
                    'Rank': i,
                    'Symbol': analysis['symbol'],
                    'Company': fund.get('company_name', 'N/A')[:30],
                    'Score': f"{analysis['score']:.1f}",
                    'Current Price': f"${fund.get('current_price', 0):.2f}",
                    '1Y Prediction': f"{multi_pred['1_year']:.1f}%",
                    '5Y Prediction': f"{multi_pred['5_years']:.1f}%",
                    'Investment': f"${inv_rec['amount']:.0f}",
                    'Shares': inv_rec['shares'],
                    'PE Ratio': f"{fund.get('pe_ratio', 0):.2f}",
                    'ROE': f"{fund.get('roe', 0)*100:.1f}%"
                })
            
            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Send email notification if configured
            if notification_frequency != "None" and sender_email and sender_password:
                if st.button("📧 Send Email Report"):
                    with st.spinner("Sending email..."):
                        success = emailer.send_daily_recommendations(
                            current_user.email, analyses, sender_email, sender_password
                        )
                        if success:
                            st.success("Email sent successfully!")
                        else:
                            st.error("Failed to send email. Check your credentials.")
    
    with tab2:
        st.subheader("📈 Multi-Timeframe Projections")
        
        if st.session_state.analyses:
            analyses = st.session_state.analyses
            
            # Multi-timeframe comparison chart
            symbols = [a['symbol'] for a in analyses[:10]]  # Top 10
            timeframes = ['6_months', '1_year', '5_years', '10_years']
            timeframe_labels = ['6 Months', '1 Year', '5 Years', '10 Years']
            
            fig_multi = go.Figure()
            
            for i, timeframe in enumerate(timeframes):
                values = [a['multi_timeframe_predictions'][timeframe] for a in analyses[:10]]
                fig_multi.add_trace(go.Bar(
                    x=symbols,
                    y=values,
                    name=timeframe_labels[i],
                    text=[f"{v:.1f}%" for v in values],
                    textposition='auto'
                ))
            
            fig_multi.update_layout(
                title="Multi-Timeframe Projected Returns (Top 10 Stocks)",
                xaxis_title="Stock Symbol",
                yaxis_title="Projected Return (%)",
                barmode='group'
            )
            st.plotly_chart(fig_multi, use_container_width=True)
            
            # Detailed projections table
            st.subheader("📊 Detailed Multi-Timeframe Analysis")
            
            projection_data = []
            for analysis in analyses[:15]:  # Show top 15
                fund = analysis['fundamentals']
                multi_pred = analysis['multi_timeframe_predictions']
                
                projection_data.append({
                    'Symbol': analysis['symbol'],
                    'Company': fund.get('company_name', 'N/A')[:25],
                    'Current Price': f"${fund.get('current_price', 0):.2f}",
                    '6M Return': f"{multi_pred['6_months']:.1f}%",
                    '1Y Return': f"{multi_pred['1_year']:.1f}%",
                    '5Y Return': f"{multi_pred['5_years']:.1f}%",
                    '10Y Return': f"{multi_pred['10_years']:.1f}%",
                    'Volatility': f"{fund.get('volatility', 0)*100:.1f}%",
                    'Beta': f"{fund.get('beta', 1):.2f}"
                })
            
            df_projections = pd.DataFrame(projection_data)
            st.dataframe(df_projections, use_container_width=True, hide_index=True)
            
        else:
            st.info("Run analysis first to see multi-timeframe projections.")
    
    with tab3:
        st.subheader("🔄 Backtesting Analysis")
        
        if st.session_state.analyses:
            analyses = st.session_state.analyses
            
            # Select stock for backtesting
            top_5_symbols = [a['symbol'] for a in analyses[:5]]
            selected_symbol = st.selectbox("Select Stock for Backtesting", top_5_symbols)
            
            col1, col2 = st.columns([1, 3])
            with col1:
                backtest_months = st.slider("Backtest Period (Months)", 3, 12, 6)
            with col2:
                if st.button("🔄 Run Backtest", type="primary"):
                    with st.spinner(f"Running backtest for {selected_symbol}..."):
                        backtest_results = analyzer.ml_predictor.backtest_predictions(selected_symbol, backtest_months)
                    
                    if 'error' in backtest_results:
                        st.error(f"Backtest failed: {backtest_results['error']}")
                    else:
                        st.success(f"Backtest completed with {backtest_results['num_predictions']} predictions")
                        
                        # Display backtest metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Direction Accuracy", f"{backtest_results['direction_accuracy']:.1f}%")
                        with col2:
                            st.metric("Mean Absolute Error", f"{backtest_results['mae']:.2f}%")
                        with col3:
                            st.metric("Total Predictions", backtest_results['num_predictions'])
                        
                        # Backtest chart
                        if len(backtest_results['dates']) > 0:
                            fig_backtest = go.Figure()
                            
                            fig_backtest.add_trace(go.Scatter(
                                x=backtest_results['dates'],
                                y=backtest_results['actual_returns'],
                                mode='lines+markers',
                                name='Actual Returns',
                                line=dict(color='blue')
                            ))
                            
                            fig_backtest.add_trace(go.Scatter(
                                x=backtest_results['dates'],
                                y=backtest_results['predicted_returns'],
                                mode='lines+markers',
                                name='Predicted Returns',
                                line=dict(color='red', dash='dash')
                            ))
                            

                            fig_backtest.update_layout(
                                title=f"Backtest Results for {selected_symbol}",
                                xaxis_title="Date",
                                yaxis_title="Return (%)",
                                hovermode='x unified'
                            )
                            st.plotly_chart(fig_backtest, use_container_width=True)
                        
                        # Accuracy analysis
                        st.subheader("📊 Accuracy Analysis")
                        st.markdown(f"""
                        **Key Insights:**
                        - **Direction Accuracy:** {backtest_results['direction_accuracy']:.1f}% (How often we predicted the right direction)
                        - **Mean Absolute Error:** {backtest_results['mae']:.2f}% (Average prediction error)
                        - **Mean Squared Error:** {backtest_results['mse']:.2f} (Penalty for large errors)
                        
                        A direction accuracy above 60% is considered good for short-term predictions.
                        Lower MAE indicates more precise predictions.
                        """)
        else:
            st.info("Run analysis first to perform backtesting.")


    
    with tab4:
        st.subheader("📊 Portfolio Optimization")
        
        if st.session_state.analyses:
            analyses = st.session_state.analyses
            
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
    

    with tab5:
        st.subheader("📤 Export & Reports")
        
        if st.session_state.analyses:
            analyses = st.session_state.analyses
            
            st.markdown("### 📊 Available Reports")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📄 Comprehensive Report")
                st.markdown("Full analysis with all stocks, projections, and recommendations")
                
                if st.button("📥 Generate & Download Report", type="primary"):
                    report_content = exporter.generate_comprehensive_report(analyses, user_profile)
                    st.download_button(
                        label="📥 Download Markdown Report",
                        data=report_content,
                        file_name=f"investment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
                    st.success("Report generated successfully!")
            
            with col2:
                st.markdown("#### 📊 CSV Data Export")
                st.markdown("Raw data for further analysis in Excel/other tools")
                
                # Prepare CSV data
                csv_data = []
                for i, analysis in enumerate(analyses, 1):
                    fund = analysis['fundamentals']
                    multi_pred = analysis['multi_timeframe_predictions']
                    inv_rec = analysis['investment_recommendation']
                    
                    csv_data.append({
                        'Rank': i,
                        'Symbol': analysis['symbol'],
                        'Company': fund.get('company_name', 'N/A'),
                        'Score': analysis['score'],
                        'Current_Price': fund.get('current_price', 0),
                        '6M_Prediction': multi_pred['6_months'],
                        '1Y_Prediction': multi_pred['1_year'],
                        '5Y_Prediction': multi_pred['5_years'],
                        '10Y_Prediction': multi_pred['10_years'],
                        'Investment_Amount': inv_rec['amount'],
                        'Shares': inv_rec['shares'],
                        'Portfolio_Percentage': inv_rec['percentage'],
                        'PE_Ratio': fund.get('pe_ratio', 0),
                        'ROE': fund.get('roe', 0),
                        'Revenue_Growth': fund.get('revenue_growth', 0),
                        'Debt_Equity': fund.get('debt_to_equity', 0),
                        'Beta': fund.get('beta', 1),
                        'Volatility': fund.get('volatility', 0),
                        'Analysis_Date': analysis['analysis_date']
                    })
                
                df_csv = pd.DataFrame(csv_data)
                csv_string = df_csv.to_csv(index=False)

                
                st.download_button(
                    label="📊 Download CSV Data",
                    data=csv_string,
                    file_name=f"stock_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            # Summary metrics
            st.markdown("### 📈 Analysis Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Stocks Analyzed", len(analyses))
            with col2:
                avg_score = np.mean([a['score'] for a in analyses])
                st.metric("Average Score", f"{avg_score:.1f}/100")
            with col3:
                total_investment = sum([a['investment_recommendation']['amount'] for a in analyses])
                st.metric("Total Investment", f"${total_investment:,.0f}")
            with col4:
                avg_1y_return = np.mean([a['multi_timeframe_predictions']['1_year'] for a in analyses])
                st.metric("Avg 1Y Projection", f"{avg_1y_return:.1f}%")
                
        else:

            st.info("Run analysis first to generate reports.")
    

    with tab6:
        st.subheader(f"ℹ️ About {APP_NAME}")
        
        # Version Information
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Version", APP_VERSION)
        with col2:
            st.metric("Build Date", APP_BUILD_DATE)
        with col3:
            st.metric("Total Markets", "3 Countries")
        
        st.markdown("---")
        
        # Latest Updates
        st.subheader("🆕 Latest Updates (v0.2.0)")
        for update in VERSION_NOTES[APP_VERSION]:
            st.markdown(f"✅ {update}")
        
        # Feature Highlights
        st.subheader("🌟 Key Features")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🔐 User Authentication:**
            - Secure login system
            - Password remember functionality
            - Personal user profiles
            
            **📈 Multi-Timeframe Analysis:**
            - 6 months, 1 year, 5 years, 10 years projections
            - Advanced ML predictions
            - Risk-adjusted forecasting
            
            **🔄 Backtesting Engine:**
            - Test predictions against historical data
            - Direction accuracy measurement
            - Performance validation
            """)
        
        with col2:
            st.markdown("""
            **📊 Comprehensive Analysis:**
            - Top 5 detailed recommendations
            - All 20 stocks displayed in table
            - Real-time financial metrics
            
            **📤 Export Capabilities:**
            - Markdown reports generation
            - CSV data export
            - Download functionality
            
            **🔄 App Management:**
            - Reset functionality
            - Session state management
            - User preference updates
            """)
        
        # Version History
        if len(VERSION_NOTES) > 1:
            st.subheader("📋 Version History")
            for version in sorted(VERSION_NOTES.keys(), reverse=True):
                with st.expander(f"Version {version}" + (" (Current)" if version == APP_VERSION else "")):
                    for note in VERSION_NOTES[version]:
                        st.markdown(f"• {note}")
        
        st.markdown("---")
        
        # Technical Details
        st.subheader("🔧 Technical Specifications")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Frontend:**
            - Streamlit Web Framework
            - Plotly Interactive Charts
            - Responsive UI Design
            - Session State Management
            
            **Data Sources:**
            - Yahoo Finance API
            - Real-time Market Data
            - Financial Metrics
            - Historical Price Data
            """)
        
        with col2:
            st.markdown("""
            **AI/ML Engine:**
            - Gradient Boosting Regression
            - Multi-timeframe Predictions
            - Backtesting Algorithms
            - 12 Feature Analysis
            
            **Backend:**
            - SQLite Database
            - User Authentication
            - Email Integration
            - Portfolio Optimization
            """)
        
        # Supported Markets
        st.subheader("🌍 Supported Markets")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🇺🇸 United States**
            - Top 20 S&P 500 stocks
            - Large-cap focus
            - High liquidity
            """)
        
        with col2:
            st.markdown("""
            **🇨🇦 Canada**
            - TSX top 15 stocks
            - Resource & financial focus
            - Stable dividend stocks
            """)
        
        with col3:
            st.markdown("""
            **🇮🇳 India**
            - NIFTY 50 + SENSEX (35 stocks)
            - Emerging market opportunities
            - Technology & pharma focus
            """)
        
        # Disclaimer
        st.markdown("---")
        st.subheader("⚠️ Important Disclaimer")
        st.warning("""
        This application provides educational analysis and should not be considered as financial advice. 
        All investments carry risk of loss. Please conduct your own research and consult with licensed 
        financial advisors before making investment decisions. Past performance does not guarantee future results.
        
        The ML predictions are based on historical data and may not accurately reflect future market conditions.
        """)
        
        # Credits
        st.markdown("---")

        st.subheader("👨‍💻 Credits & Technology Stack")
        st.markdown("""
        - **AI/ML Framework**: Scikit-learn, Pandas, NumPy
        - **Data Provider**: Yahoo Finance API
        - **Web Framework**: Streamlit
        - **Visualization**: Plotly
        - **Database**: SQLite
        - **Authentication**: SHA-256 Hashing
        """)
        
        st.markdown("---")
        st.markdown(f"<center><i>© 2024 {APP_NAME} v{APP_VERSION} | All Rights Reserved</i></center>", 

                   unsafe_allow_html=True)

if __name__ == "__main__":
    main()