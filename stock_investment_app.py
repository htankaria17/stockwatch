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
import hashlib
import requests
import time

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
APP_BUILD_DATE = "2024-12-28"
APP_NAME = "AI-Powered Stock Investment Analyzer"
APP_CONCEPT_OWNER = "Harshal Tankaria"
VERSION_NOTES = {
    "V0.2.0": [
        "Fixed email functionality with enhanced test button and export features", 
        "Added proper user login system with secure authentication",
        "Fixed date and time handling with internet synchronization",
        "Added enhanced buy/sell triggers for top 5 trading stocks",
        "Improved export functionality for reports and analysis",
        "Added concept owner attribution (Harshal Tankaria)",
        "Enhanced day trading signals with real-time alerts"
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

# Enhanced timezone and time handling
def get_internet_time():
    """Get accurate current time from internet"""
    try:
        # Use WorldTimeAPI for accurate time
        response = requests.get('http://worldtimeapi.org/api/timezone/UTC', timeout=5)
        if response.status_code == 200:
            time_data = response.json()
            utc_time = datetime.fromisoformat(time_data['datetime'].replace('Z', '+00:00'))
            return utc_time
        else:
            # Fallback to system time
            return datetime.now(pytz.UTC)
    except:
        # Fallback to system time if internet fails
        return datetime.now(pytz.UTC)

def get_current_time(timezone_str='US/Eastern'):
    """Get current time with proper timezone handling using internet time"""
    try:
        utc_time = get_internet_time()
        tz = pytz.timezone(timezone_str)
        return utc_time.astimezone(tz)
    except:
        # Fallback to system time
        tz = pytz.timezone(timezone_str)
        return datetime.now(tz)

def format_datetime(dt, format_str='%Y-%m-%d %H:%M:%S %Z'):
    """Format datetime with timezone information"""
    return dt.strftime(format_str)

# Enhanced Day Trading Configuration with better signals
DAY_TRADING_STOCKS = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
TRADING_HOURS = {
    'market_open': 9.5,  # 9:30 AM
    'market_close': 16.0,  # 4:00 PM
}

# Missing imports for complete functionality
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

# User Authentication System
class UserAuth:
    def __init__(self, db_name="investment_app.db"):
        self.db_name = db_name
        self.init_auth_tables()
    
    def init_auth_tables(self):
        """Initialize authentication tables"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        # User authentication table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_auth (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        ''')
        
        # User sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                session_token TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                is_active BOOLEAN DEFAULT 1,
                FOREIGN KEY (user_id) REFERENCES user_auth (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def hash_password(self, password: str) -> tuple:
        """Hash password with salt"""
        salt = hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]
        password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
        return password_hash, salt
    
    def verify_password(self, password: str, stored_hash: str, salt: str) -> bool:
        """Verify password against stored hash"""
        return hashlib.sha256((password + salt).encode()).hexdigest() == stored_hash
    
    def register_user(self, username: str, email: str, password: str) -> tuple:
        """Register new user"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        try:
            # Check if user exists
            cursor.execute('SELECT id FROM user_auth WHERE username = ? OR email = ?', (username, email))
            if cursor.fetchone():
                return False, "Username or email already exists"
            
            # Hash password
            password_hash, salt = self.hash_password(password)
            
            # Insert user
            cursor.execute('''
                INSERT INTO user_auth (username, email, password_hash, salt)
                VALUES (?, ?, ?, ?)
            ''', (username, email, password_hash, salt))
            
            conn.commit()
            return True, "User registered successfully"
        
        except Exception as e:
            return False, f"Registration failed: {e}"
        finally:
            conn.close()
    
    def login_user(self, username: str, password: str) -> tuple:
        """Login user and create session"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        try:
            # Get user data
            cursor.execute('''
                SELECT id, password_hash, salt, email FROM user_auth 
                WHERE username = ? AND is_active = 1
            ''', (username,))
            
            user_data = cursor.fetchone()
            if not user_data:
                return False, "Invalid username or password", None
            
            user_id, stored_hash, salt, email = user_data
            
            # Verify password
            if not self.verify_password(password, stored_hash, salt):
                return False, "Invalid username or password", None
            
            # Create session token
            session_token = hashlib.sha256(f"{user_id}{time.time()}".encode()).hexdigest()
            expires_at = datetime.now() + timedelta(hours=24)  # Session expires in 24 hours
            
            # Store session
            cursor.execute('''
                INSERT INTO user_sessions (user_id, session_token, expires_at)
                VALUES (?, ?, ?)
            ''', (user_id, session_token, expires_at))
            
            # Update last login
            cursor.execute('''
                UPDATE user_auth SET last_login = CURRENT_TIMESTAMP WHERE id = ?
            ''', (user_id,))
            
            conn.commit()
            
            return True, "Login successful", {
                'user_id': user_id,
                'username': username,
                'email': email,
                'session_token': session_token
            }
        
        except Exception as e:
            return False, f"Login failed: {e}", None
        finally:
            conn.close()
    
    def verify_session(self, session_token: str) -> tuple:
        """Verify user session"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT s.user_id, u.username, u.email FROM user_sessions s
                JOIN user_auth u ON s.user_id = u.id
                WHERE s.session_token = ? AND s.is_active = 1 AND s.expires_at > CURRENT_TIMESTAMP
            ''', (session_token,))
            
            result = cursor.fetchone()
            if result:
                return True, {
                    'user_id': result[0],
                    'username': result[1], 
                    'email': result[2]
                }
            else:
                return False, None
        
        except Exception as e:
            return False, None
        finally:
            conn.close()
    
    def logout_user(self, session_token: str):
        """Logout user by deactivating session"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE user_sessions SET is_active = 0 WHERE session_token = ?
        ''', (session_token,))
        
        conn.commit()
        conn.close()

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
        
    def get_intraday_data(self, symbol: str, period: str = '2d', interval: str = '5m') -> pd.DataFrame:
        """Get enhanced intraday data for day trading analysis"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period, interval=interval)
            return data
        except Exception as e:
            st.error(f"Error fetching intraday data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_day_trading_signals(self, symbol: str) -> Dict:
        """Enhanced buy/sell signals calculation with multiple timeframes and indicators"""
        data = self.get_intraday_data(symbol)
        if data.empty or len(data) < 50:
            return {}
            
        # Calculate enhanced technical indicators
        data['SMA_10'] = data['Close'].rolling(window=10).mean()
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['EMA_12'] = data['Close'].ewm(span=12).mean()
        data['EMA_26'] = data['Close'].ewm(span=26).mean()
        
        # RSI (Relative Strength Index)
        data['RSI'] = self.calculate_rsi(data['Close'])
        
        # MACD indicators
        data['MACD'], data['MACD_signal'] = self.calculate_macd(data['Close'])
        data['MACD_histogram'] = data['MACD'] - data['MACD_signal']
        
        # Bollinger Bands
        data['BB_middle'] = data['Close'].rolling(window=20).mean()
        bb_std = data['Close'].rolling(window=20).std()
        data['BB_upper'] = data['BB_middle'] + (bb_std * 2)
        data['BB_lower'] = data['BB_middle'] - (bb_std * 2)
        
        # Volume indicators
        data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
        data['Volume_ratio'] = data['Volume'] / data['Volume_SMA']
        
        # Stochastic Oscillator
        data['Stoch_K'], data['Stoch_D'] = self.calculate_stochastic(data)
        
        # Price momentum
        data['Price_change'] = data['Close'].pct_change()
        data['Price_momentum'] = data['Close'].rolling(window=10).apply(lambda x: (x.iloc[-1] / x.iloc[0]) - 1)
        
        current = data.iloc[-1]
        prev = data.iloc[-2] if len(data) > 1 else current
        prev2 = data.iloc[-3] if len(data) > 2 else prev
        
        # Generate enhanced signals
        signals = {
            'symbol': symbol,
            'current_price': current['Close'],
            'timestamp': get_current_time(),
            'volume': current['Volume'],
            'volume_ratio': current['Volume_ratio'],
            'rsi': current['RSI'],
            'macd': current['MACD'],
            'signals': []
        }
        
        # Enhanced BUY signals with multiple confirmations
        buy_signals = []
        
        # Signal 1: Golden Cross + Volume + RSI
        if (current['SMA_10'] > current['SMA_20'] and 
            prev['SMA_10'] <= prev['SMA_20'] and 
            current['RSI'] > 30 and current['RSI'] < 70 and
            current['Volume_ratio'] > 1.2):
            buy_signals.append({
                'type': 'BUY',
                'reason': 'Golden Cross (SMA 10/20) with volume confirmation',
                'strength': 'STRONG',
                'confidence': 85,
                'entry_price': current['Close'],
                'stop_loss': current['Close'] * 0.985,
                'target': current['Close'] * 1.025
            })
        
        # Signal 2: MACD Bullish Crossover + RSI oversold recovery
        if (current['MACD'] > current['MACD_signal'] and 
            prev['MACD'] <= prev['MACD_signal'] and 
            current['RSI'] > 35 and current['RSI'] < 65 and
            current['MACD_histogram'] > prev['MACD_histogram']):
            buy_signals.append({
                'type': 'BUY',
                'reason': 'MACD bullish crossover with RSI confirmation',
                'strength': 'MODERATE',
                'confidence': 75,
                'entry_price': current['Close'],
                'stop_loss': current['Close'] * 0.98,
                'target': current['Close'] * 1.03
            })
        
        # Signal 3: Bollinger Band Bounce + Volume
        if (current['Close'] > current['BB_lower'] and 
            prev['Close'] <= prev['BB_lower'] and 
            current['RSI'] < 40 and
            current['Volume_ratio'] > 1.1):
            buy_signals.append({
                'type': 'BUY',
                'reason': 'Bollinger Band lower bounce with oversold RSI',
                'strength': 'MODERATE',
                'confidence': 70,
                'entry_price': current['Close'],
                'stop_loss': current['BB_lower'] * 0.995,
                'target': current['BB_middle']
            })
        
        # Signal 4: Stochastic Oversold Recovery
        if (current['Stoch_K'] > current['Stoch_D'] and 
            prev['Stoch_K'] <= prev['Stoch_D'] and 
            current['Stoch_K'] < 30 and current['RSI'] < 35):
            buy_signals.append({
                'type': 'BUY',
                'reason': 'Stochastic oversold recovery',
                'strength': 'WEAK',
                'confidence': 60,
                'entry_price': current['Close'],
                'stop_loss': current['Close'] * 0.975,
                'target': current['Close'] * 1.02
            })
        
        # Enhanced SELL signals with multiple confirmations
        sell_signals = []
        
        # Signal 1: Death Cross + Volume + RSI
        if (current['SMA_10'] < current['SMA_20'] and 
            prev['SMA_10'] >= prev['SMA_20'] and 
            current['RSI'] > 50 and current['RSI'] < 80 and
            current['Volume_ratio'] > 1.2):
            sell_signals.append({
                'type': 'SELL',
                'reason': 'Death Cross (SMA 10/20) with volume confirmation',
                'strength': 'STRONG',
                'confidence': 85,
                'entry_price': current['Close'],
                'stop_loss': current['Close'] * 1.015,
                'target': current['Close'] * 0.975
            })
        
        # Signal 2: MACD Bearish Crossover + RSI overbought
        if (current['MACD'] < current['MACD_signal'] and 
            prev['MACD'] >= prev['MACD_signal'] and 
            current['RSI'] > 60 and current['RSI'] < 85 and
            current['MACD_histogram'] < prev['MACD_histogram']):
            sell_signals.append({
                'type': 'SELL',
                'reason': 'MACD bearish crossover with RSI overbought',
                'strength': 'MODERATE',
                'confidence': 75,
                'entry_price': current['Close'],
                'stop_loss': current['Close'] * 1.02,
                'target': current['Close'] * 0.97
            })
        
        # Signal 3: Bollinger Band Upper Touch + Volume
        if (current['Close'] < current['BB_upper'] and 
            prev['Close'] >= prev['BB_upper'] and 
            current['RSI'] > 65 and
            current['Volume_ratio'] > 1.1):
            sell_signals.append({
                'type': 'SELL',
                'reason': 'Bollinger Band upper rejection with overbought RSI',
                'strength': 'MODERATE',
                'confidence': 70,
                'entry_price': current['Close'],
                'stop_loss': current['BB_upper'] * 1.005,
                'target': current['BB_middle']
            })
        
        # Signal 4: Stochastic Overbought Reversal
        if (current['Stoch_K'] < current['Stoch_D'] and 
            prev['Stoch_K'] >= prev['Stoch_D'] and 
            current['Stoch_K'] > 70 and current['RSI'] > 65):
            sell_signals.append({
                'type': 'SELL',
                'reason': 'Stochastic overbought reversal',
                'strength': 'WEAK',
                'confidence': 60,
                'entry_price': current['Close'],
                'stop_loss': current['Close'] * 1.025,
                'target': current['Close'] * 0.98
            })
        
        # Combine and filter signals by confidence
        all_signals = buy_signals + sell_signals
        high_confidence_signals = [s for s in all_signals if s.get('confidence', 0) >= 70]
        
        # Add best signals to result
        signals['signals'] = sorted(high_confidence_signals, key=lambda x: x.get('confidence', 0), reverse=True)[:3]
        
        return signals
    
    def calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator with improved accuracy"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """Calculate MACD indicator with enhanced parameters"""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal
    
    def calculate_stochastic(self, data: pd.DataFrame, k_window: int = 14, d_window: int = 3):
        """Calculate Stochastic Oscillator"""
        high_max = data['High'].rolling(window=k_window).max()
        low_min = data['Low'].rolling(window=k_window).min()
        
        stoch_k = 100 * (data['Close'] - low_min) / (high_max - low_min)
        stoch_d = stoch_k.rolling(window=d_window).mean()
        
        return stoch_k, stoch_d
    
    def get_all_day_trading_signals(self) -> List[Dict]:
        """Get enhanced day trading signals for all tracked stocks"""
        all_signals = []
        
        for symbol in self.stocks:
            try:
                signals = self.calculate_day_trading_signals(symbol)
                if signals and signals.get('signals'):
                    all_signals.append(signals)
            except Exception as e:
                st.warning(f"Could not analyze {symbol}: {e}")
                continue
        
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
        """Send a test email to verify email configuration with enhanced error handling"""
        try:
            current_time = get_current_time()
            subject = f"✅ Email Test Successful - {APP_NAME} - {format_datetime(current_time, '%Y-%m-%d %H:%M %Z')}"
            
            html_content = f"""
            <html>
            <head>
                <style>
                    body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; }}
                    .header {{ background: linear-gradient(135deg, #28a745, #20c997); color: white; padding: 30px; text-align: center; }}
                    .content {{ padding: 30px; background: #f8f9fa; }}
                    .test-box {{ background: white; border-radius: 10px; padding: 20px; margin: 20px 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                    .success-badge {{ background: #28a745; color: white; padding: 5px 15px; border-radius: 20px; display: inline-block; }}
                    .footer {{ background-color: #343a40; color: white; padding: 20px; text-align: center; font-size: 12px; }}
                    .feature-list {{ margin: 15px 0; }}
                    .feature-item {{ margin: 8px 0; padding: 5px 0; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>🎉 Email Configuration Test Successful!</h1>
                    <p>Your email setup is working perfectly</p>
                    <span class="success-badge">✅ VERIFIED</span>
                </div>
                
                <div class="content">
                    <div class="test-box">
                        <h2>📊 Test Results</h2>
                        <p><strong>Application:</strong> {APP_NAME} v{APP_VERSION}</p>
                        <p><strong>Concept & Owner:</strong> {APP_CONCEPT_OWNER}</p>
                        <p><strong>Test Time:</strong> {format_datetime(current_time)}</p>
                        <p><strong>Recipient:</strong> {user_email}</p>
                        <p><strong>Sender:</strong> {sender_email}</p>
                        <p><strong>Status:</strong> <span style="color: #28a745; font-weight: bold;">✅ SUCCESS</span></p>
                    </div>
                    
                    <div class="test-box">
                        <h3>🚀 What You Can Expect</h3>
                        <div class="feature-list">
                            <div class="feature-item">📈 <strong>Daily Investment Reports</strong> - Top stock recommendations with ML predictions</div>
                            <div class="feature-item">⚡ <strong>Day Trading Alerts</strong> - Real-time buy/sell signals for top 5 stocks</div>
                            <div class="feature-item">📊 <strong>Portfolio Analysis</strong> - Detailed CSV exports with your data</div>
                            <div class="feature-item">🤖 <strong>AI Insights</strong> - Machine learning enhanced predictions</div>
                            <div class="feature-item">🌍 <strong>Multi-Market Support</strong> - US, Canada, and India markets</div>
                        </div>
                    </div>
                </div>
                
                <div class="footer">
                    <p><strong>{APP_NAME}</strong> v{APP_VERSION} | Developed by <strong>{APP_CONCEPT_OWNER}</strong></p>
                    <p>AI-Powered Investment Analysis Platform | Build Date: {APP_BUILD_DATE}</p>
                </div>
            </body>
            </html>
            """
            
            # Create message with better headers
            msg = MimeMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = f"{APP_CONCEPT_OWNER} <{sender_email}>"
            msg['To'] = user_email
            msg['Reply-To'] = sender_email
            
            # Attach HTML content
            html_part = MimeText(html_content, 'html')
            msg.attach(html_part)
            
            # Send email with enhanced error handling
            try:
                server = smtplib.SMTP(self.smtp_server, self.smtp_port)
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(msg)
                server.quit()
                return True, "Test email sent successfully!"
                
            except smtplib.SMTPAuthenticationError:
                return False, "Authentication failed. Please check your Gmail App Password."
            except smtplib.SMTPRecipientsRefused:
                return False, "Recipient email address was refused by the server."
            except smtplib.SMTPServerDisconnected:
                return False, "SMTP server disconnected. Please try again."
            except Exception as smtp_error:
                return False, f"SMTP Error: {str(smtp_error)}"
            
        except Exception as e:
            return False, f"Email configuration error: {str(e)}"

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
    auth = UserAuth()
    db = DatabaseManager()
    analyzer = EnhancedStockAnalyzer()
    emailer = EmailNotifier()
    day_trader = DayTradingAnalyzer()
    exporter = ExportManager()
    
    # Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.user_data = None
    
    # Check if user is already logged in via session token
    if not st.session_state.logged_in and 'session_token' in st.session_state:
        is_valid, user_data = auth.verify_session(st.session_state.session_token)
        if is_valid:
            st.session_state.logged_in = True
            st.session_state.user_data = user_data
    
    # Login/Registration Interface
    if not st.session_state.logged_in:
        st.title("🔐 User Login")
        st.markdown(f"**Welcome to {APP_NAME} v{APP_VERSION}**")
        st.markdown(f"**Concept & Owner:** {APP_CONCEPT_OWNER}")
        
        # Display current time from internet
        current_time = get_current_time()
        st.info(f"**Current Time (Accurate):** {format_datetime(current_time, '%A, %B %d, %Y at %I:%M:%S %p %Z')}")
        
        tab1, tab2 = st.tabs(["🔑 Login", "📝 Register"])
        
        with tab1:
            st.subheader("Login to Your Account")
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                submit_login = st.form_submit_button("🔑 Login", type="primary")
                
                if submit_login:
                    if username and password:
                        success, message, user_data = auth.login_user(username, password)
                        if success:
                            st.session_state.logged_in = True
                            st.session_state.user_data = user_data
                            st.session_state.session_token = user_data['session_token']
                            st.success(f"Welcome back, {user_data['username']}!")
                            st.rerun()
                        else:
                            st.error(message)
                    else:
                        st.error("Please fill in all fields")
        
        with tab2:
            st.subheader("Create New Account")
            with st.form("register_form"):
                new_username = st.text_input("Choose Username", placeholder="Enter a unique username")
                new_email = st.text_input("Email Address", placeholder="Enter your email address")
                new_password = st.text_input("Password", type="password", placeholder="Choose a strong password")
                confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
                submit_register = st.form_submit_button("📝 Register", type="primary")
                
                if submit_register:
                    if new_username and new_email and new_password and confirm_password:
                        if new_password == confirm_password:
                            if len(new_password) >= 6:
                                success, message = auth.register_user(new_username, new_email, new_password)
                                if success:
                                    st.success(f"{message} You can now login!")
                                else:
                                    st.error(message)
                            else:
                                st.error("Password must be at least 6 characters long")
                        else:
                            st.error("Passwords do not match")
                    else:
                        st.error("Please fill in all fields")
        
        # Information about the app
        st.markdown("---")
        st.subheader("🤖 About This Application")
        st.markdown(f"""
        **{APP_NAME}** is an advanced AI-powered investment analysis platform that provides:
        
        ✅ **Smart Stock Analysis** - AI/ML-driven recommendations  
        ✅ **Day Trading Signals** - Real-time buy/sell triggers for top 5 stocks  
        ✅ **Email Reports** - Automated daily/weekly/monthly reports  
        ✅ **Portfolio Optimization** - Risk-adjusted investment allocations  
        ✅ **Multi-Market Support** - US, Canadian, and Indian markets  
        ✅ **Export Features** - Download analysis as CSV files  
        
        **Latest Updates in v{APP_VERSION}:**
        """)
        for update in VERSION_NOTES[APP_VERSION]:
            st.markdown(f"• {update}")
        
        st.markdown(f"""
        ---
        **Concept & Owner:** {APP_CONCEPT_OWNER}  
        **Built on:** {APP_BUILD_DATE}
        """)
        return
    
    # Main Application (for logged-in users)
    # Sidebar for user input
    st.sidebar.title("🚀 Investment Preferences")
    
    # User info and logout button
    with st.sidebar:
        st.markdown("---")
        st.markdown(f"**👤 Logged in as:** {st.session_state.user_data['username']}")
        if st.button("🚪 Logout"):
            auth.logout_user(st.session_state.session_token)
            st.session_state.logged_in = False
            st.session_state.user_data = None
            if 'session_token' in st.session_state:
                del st.session_state.session_token
            st.rerun()
    
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
    
    # User profile inputs (auto-filled from logged-in user)
    user_name = st.sidebar.text_input("Full Name", value=st.session_state.user_data['username'])
    user_email = st.sidebar.text_input("Email Address", value=st.session_state.user_data['email'])
    
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
        
        # Enhanced test email button
        if sender_email and sender_password and user_email:
            col1, col2 = st.sidebar.columns(2)
            with col1:
                if st.button("🧪 Test Email", help="Send a test email to verify your configuration"):
                    with st.spinner("Sending test email..."):
                        success, message = emailer.send_test_email(user_email, sender_email, sender_password)
                        if success:
                            st.success(f"✅ {message}")
                        else:
                            st.error(f"❌ {message}")
            with col2:
                if st.button("📧 Quick Test", help="Basic connectivity test"):
                    with st.spinner("Testing connection..."):
                        try:
                            # Quick SMTP test without sending email
                            server = smtplib.SMTP('smtp.gmail.com', 587)
                            server.starttls()
                            server.login(sender_email, sender_password)
                            server.quit()
                            st.success("✅ SMTP connection successful!")
                        except Exception as e:
                            st.error(f"❌ Connection failed: {str(e)}")
        else:
            st.sidebar.info("💡 Enter email credentials above to enable testing")
    
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
            
            # Display enhanced signals by stock
            for stock_data in day_trading_signals:
                if stock_data['signals']:
                    # Enhanced stock header with more data
                    rsi_color = "🟢" if 30 < stock_data.get('rsi', 50) < 70 else ("🔴" if stock_data.get('rsi', 50) > 70 else "🟡")
                    volume_indicator = "🔥" if stock_data.get('volume_ratio', 1) > 1.5 else "📊"
                    
                    with st.expander(f"📈 {stock_data['symbol']} - ${stock_data['current_price']:.2f} | {rsi_color} RSI: {stock_data.get('rsi', 0):.1f} | {volume_indicator} Vol: {stock_data.get('volume_ratio', 1):.1f}x ({len(stock_data['signals'])} signals)"):
                        
                        # Technical indicators summary
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Current Price", f"${stock_data['current_price']:.2f}")
                        with col2:
                            rsi_val = stock_data.get('rsi', 50)
                            rsi_status = "Overbought" if rsi_val > 70 else ("Oversold" if rsi_val < 30 else "Neutral")
                            st.metric("RSI", f"{rsi_val:.1f}", rsi_status)
                        with col3:
                            vol_ratio = stock_data.get('volume_ratio', 1)
                            vol_status = "High" if vol_ratio > 1.5 else ("Low" if vol_ratio < 0.7 else "Normal")
                            st.metric("Volume Ratio", f"{vol_ratio:.1f}x", vol_status)
                        with col4:
                            macd_val = stock_data.get('macd', 0)
                            macd_trend = "Bullish" if macd_val > 0 else "Bearish"
                            st.metric("MACD", f"{macd_val:.3f}", macd_trend)
                        
                        st.markdown("---")
                        
                        # Display enhanced signals
                        for i, signal in enumerate(stock_data['signals']):
                            signal_color = "🟢" if signal['type'] == 'BUY' else "🔴"
                            strength_badge = {
                                'STRONG': "🔥",
                                'MODERATE': "📍", 
                                'WEAK': "⚪"
                            }.get(signal['strength'], "📍")
                            
                            confidence = signal.get('confidence', 50)
                            confidence_color = "🟢" if confidence >= 80 else ("🟡" if confidence >= 70 else "�")
                            
                            risk_reward = abs((signal['target'] - signal['entry_price']) / (signal['stop_loss'] - signal['entry_price']))
                            
                            st.markdown(f"""
                            <div style="border: 2px solid {'#28a745' if signal['type'] == 'BUY' else '#dc3545'}; border-radius: 10px; padding: 15px; margin: 10px 0; background: {'#f8fff8' if signal['type'] == 'BUY' else '#fff8f8'};">
                                <h4>{signal_color} {signal['type']} Signal {strength_badge} | {confidence_color} Confidence: {confidence}%</h4>
                                <p><strong>🎯 Strategy:</strong> {signal['reason']}</p>
                                <p><strong>💪 Strength:</strong> {signal['strength']} | <strong>🎯 Confidence:</strong> {confidence}%</p>
                                
                                <div style="display: flex; justify-content: space-between; margin: 10px 0;">
                                    <div><strong>💰 Entry:</strong> ${signal['entry_price']:.2f}</div>
                                    <div><strong>🛑 Stop Loss:</strong> ${signal['stop_loss']:.2f}</div>
                                    <div><strong>🎯 Target:</strong> ${signal['target']:.2f}</div>
                                    <div><strong>⚖️ R/R:</strong> {risk_reward:.2f}:1</div>
                                </div>
                                
                                <div style="margin-top: 10px;">
                                    <strong>📊 Potential:</strong> 
                                    Loss: {((signal['stop_loss'] - signal['entry_price']) / signal['entry_price'] * 100):.1f}% | 
                                    Gain: {((signal['target'] - signal['entry_price']) / signal['entry_price'] * 100):.1f}%
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
            
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
        st.markdown(f"<center><i>© 2024 {APP_NAME} | Developed by {APP_CONCEPT_OWNER} | All Rights Reserved</i></center>", 
                   unsafe_allow_html=True)

if __name__ == "__main__":
    main()