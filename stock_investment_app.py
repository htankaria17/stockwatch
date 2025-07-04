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
APP_VERSION = "V0.2.0"
APP_BUILD_DATE = "2024-12-21"
APP_NAME = "AI-Powered Stock Investment Analyzer"
APP_CONCEPT_OWNER = "Harshal Tankaria"
VERSION_NOTES = {
    "V0.2.0": [
        "Added Day Trading Mode with buy/sell triggers for top 5 stocks",
        "Fixed email functionality with test button and export features", 
        "Improved date/time handling with proper timezone support",
        "Added concept owner information (Harshal Tankaria)",
        "Enhanced export functionality for reports and analysis",
        "Added real-time day trading signals and alerts"
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
    day_trader = DayTradingAnalyzer()
    exporter = ExportManager()
    
    # Sidebar for user input
    st.sidebar.title("🚀 Investment Preferences")
    
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
        
        # Test email button
        if sender_email and sender_password and user_email:
            if st.sidebar.button("🧪 Test Email Configuration"):
                with st.spinner("Testing email configuration..."):
                    success = emailer.send_test_email(user_email, sender_email, sender_password)
                    if success:
                        st.sidebar.success("✅ Test email sent successfully!")
                    else:
                        st.sidebar.error("❌ Test email failed. Check your credentials.")
    
    # Main interface
    st.title(f"🤖 {APP_NAME}")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### Intelligent Investment Recommendations with Machine Learning")
        st.markdown(f"**Concept & Owner:** {APP_CONCEPT_OWNER}")
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
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Analysis", "📈 Portfolio", "🚨 Day Trading", "🤖 ML Insights", "⚙️ Settings", "ℹ️ About"])
    
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
                    
                    # Export and email functionality
                    st.subheader("📤 Export & Share")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("📥 Export to CSV"):
                            csv_data = exporter.export_analysis_to_csv(analyses)
                            current_time = get_current_time()
                            filename = f"investment_analysis_{format_datetime(current_time, '%Y%m%d_%H%M')}.csv"
                            st.download_button(
                                label="Download CSV",
                                data=csv_data,
                                file_name=filename,
                                mime="text/csv"
                            )
                    
                    with col2:
                        # Send email notification if configured
                        if notification_frequency != "None" and sender_email and sender_password:
                            if st.button("📧 Send Email Report"):
                                with st.spinner("Sending email..."):
                                    csv_data = exporter.export_analysis_to_csv(analyses)
                                    success = emailer.send_daily_recommendations(
                                        user_email, analyses, sender_email, sender_password, csv_data
                                    )
                                    if success:
                                        st.success("Email sent successfully with CSV attachment!")
                                    else:
                                        st.error("Failed to send email. Check your credentials.")
                    
                    with col3:
                        if st.button("📊 Generate Summary Report"):
                            st.info("Summary report feature coming soon!")
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
        st.subheader("🚨 Day Trading Mode")
        
        # Day trading header
        current_time = get_current_time()
        market_status = "🟢 OPEN" if 9.5 <= current_time.hour + current_time.minute/60 <= 16.0 else "🔴 CLOSED"
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown(f"**Market Status:** {market_status}")
            st.markdown(f"**Current Time:** {format_datetime(current_time, '%H:%M:%S %Z')}")
        with col2:
            auto_refresh = st.checkbox("Auto Refresh (30s)", value=False)
        with col3:
            refresh_signals = st.button("🔄 Refresh Signals")
        
        # Get day trading signals
        if auto_refresh or refresh_signals or 'day_trading_signals' not in st.session_state:
            with st.spinner("Analyzing day trading opportunities..."):
                day_trading_signals = day_trader.get_all_day_trading_signals()
                st.session_state.day_trading_signals = day_trading_signals
        else:
            day_trading_signals = st.session_state.get('day_trading_signals', [])
        
        if day_trading_signals:
            # Summary metrics
            total_signals = sum(len(stock['signals']) for stock in day_trading_signals)
            buy_signals = sum(len([s for s in stock['signals'] if s['type'] == 'BUY']) for stock in day_trading_signals)
            sell_signals = total_signals - buy_signals
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Signals", total_signals)
            with col2:
                st.metric("Buy Signals", buy_signals, delta=f"+{buy_signals}")
            with col3:
                st.metric("Sell Signals", sell_signals, delta=f"-{sell_signals}")
            with col4:
                st.metric("Stocks Monitored", len(DAY_TRADING_STOCKS))
            
            # Display signals by stock
            for stock_data in day_trading_signals:
                if stock_data['signals']:
                    with st.expander(f"📈 {stock_data['symbol']} - ${stock_data['current_price']:.2f} ({len(stock_data['signals'])} signals)"):
                        for i, signal in enumerate(stock_data['signals']):
                            signal_color = "🟢" if signal['type'] == 'BUY' else "🔴"
                            strength_badge = "🔥" if signal['strength'] == 'STRONG' else "📍"
                            
                            st.markdown(f"""
                            ### {signal_color} {signal['type']} Signal {strength_badge}
                            **Strength:** {signal['strength']}  
                            **Reason:** {signal['reason']}  
                            **Entry Price:** ${signal['entry_price']:.2f}  
                            **Stop Loss:** ${signal['stop_loss']:.2f}  
                            **Target:** ${signal['target']:.2f}  
                            **Risk/Reward:** {abs((signal['target'] - signal['entry_price']) / (signal['stop_loss'] - signal['entry_price'])):.2f}:1
                            """)
            
            # Export day trading signals
            st.subheader("📤 Export Day Trading Signals")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📥 Export Signals to CSV"):
                    csv_data = exporter.export_day_trading_signals_to_csv(day_trading_signals)
                    current_time = get_current_time()
                    filename = f"day_trading_signals_{format_datetime(current_time, '%Y%m%d_%H%M')}.csv"
                    st.download_button(
                        label="Download Signals CSV",
                        data=csv_data,
                        file_name=filename,
                        mime="text/csv"
                    )
            
            with col2:
                # Send day trading alerts if configured
                if notification_frequency != "None" and sender_email and sender_password:
                    if st.button("📧 Send Day Trading Alerts"):
                        with st.spinner("Sending trading alerts..."):
                            success = emailer.send_day_trading_alerts(
                                user_email, day_trading_signals, sender_email, sender_password
                            )
                            if success:
                                st.success("Day trading alerts sent successfully!")
                            else:
                                st.error("Failed to send alerts. Check your credentials.")
        else:
            st.info("No day trading signals available at the moment. The algorithm monitors the market during trading hours.")
            
        # Trading disclaimer
        st.warning("""
        **⚠️ Day Trading Risk Warning:**  
        Day trading involves significant financial risk and is not suitable for all investors. These signals are for educational purposes only and should not be considered as financial advice. Always conduct your own research and consider consulting with a licensed financial advisor. Never risk more than you can afford to lose.
        """)
        
        # Auto refresh mechanism
        if auto_refresh:
            time.sleep(30)
            st.rerun()

    with tab4:
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
    
    with tab5:
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
        - **India**: NIFTY 50 + SENSEX stocks (35 total)
        """)
    
        st.markdown("### 🚨 Day Trading Features")
        st.markdown("""
        **New Day Trading Mode includes:**
        - Real-time technical analysis for top 5 stocks (AAPL, MSFT, GOOGL, TSLA, NVDA)
        - Buy/sell signals based on SMA, RSI, and MACD indicators
        - Risk/reward ratio calculations
        - Email alerts for trading opportunities
        - CSV export for trading signals
        - Market status and timing information
        """)
    
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
        st.subheader("🆕 Latest Updates")
        for update in VERSION_NOTES[APP_VERSION]:
            st.markdown(f"✅ {update}")
        
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
            
            **Data Sources:**
            - Yahoo Finance API
            - Real-time Market Data
            - Financial Metrics
            """)
        
        with col2:
            st.markdown("""
            **AI/ML Engine:**
            - Gradient Boosting Regression
            - 12 Feature Analysis
            - Scikit-learn Framework
            
            **Backend:**
            - SQLite Database
            - Email Integration
            - Portfolio Optimization
            """)
        
        # Disclaimer
        st.markdown("---")
        st.subheader("⚠️ Important Disclaimer")
        st.warning("""
        This application provides educational analysis and should not be considered as financial advice. 
        All investments carry risk of loss. Please conduct your own research and consult with licensed 
        financial advisors before making investment decisions. Past performance does not guarantee future results.
        """)
        
        # Credits
        st.markdown("---")
        st.subheader("👨‍💻 Credits & Ownership")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            **Concept & Owner:** {APP_CONCEPT_OWNER}  
            **Application:** {APP_NAME}  
            **Version:** {APP_VERSION}  
            **Build Date:** {APP_BUILD_DATE}
            """)
        
        with col2:
            st.markdown("""
            **Technical Stack:**
            - **ML Models**: Scikit-learn, Pandas, NumPy
            - **Data Provider**: Yahoo Finance  
            - **UI Framework**: Streamlit
            - **Visualization**: Plotly
            - **Email**: SMTP Integration
            """)
            
        st.markdown("---")
        st.markdown("### 🚀 New Features in v2.0.0")
        st.markdown("""
        ✅ **Day Trading Mode** - Real-time signals for top 5 stocks  
        ✅ **Enhanced Email System** - Test button & CSV attachments  
        ✅ **Export Functionality** - Download analysis as CSV  
        ✅ **Improved DateTime** - Proper timezone handling  
        ✅ **Better UI** - Enhanced user experience  
        ✅ **Owner Attribution** - Clear ownership and concept credits
        """)
        
        st.markdown("---")
        st.markdown(f"<center><i>© 2024 {APP_NAME} | Developed by {APP_CONCEPT_OWNER} | All Rights Reserved</i></center>", 
                   unsafe_allow_html=True)

if __name__ == "__main__":
    main()