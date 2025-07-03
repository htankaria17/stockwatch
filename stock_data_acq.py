import yfinance as yf
import pandas as pd
import sys
import csv
import datetime
import requests
from bs4 import BeautifulSoup
import numpy as np
import json
import time
import os
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class StockAnalyzer:
    def __init__(self, output_dir="./data"):
        self.output_dir = output_dir
        self.ensure_output_dir()
        
    def ensure_output_dir(self):
        """Create output directory if it doesn't exist"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def get_stock_fundamentals(self, symbol: str) -> Dict:
        """Get comprehensive fundamental analysis data for a stock"""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            history = stock.history(period="5y")
            financials = stock.financials
            balance_sheet = stock.balance_sheet
            cashflow = stock.cashflow
            
            # Calculate key financial ratios
            fundamentals = {
                'symbol': symbol,
                'company_name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'current_price': info.get('currentPrice', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'forward_pe': info.get('forwardPE', 0),
                'pb_ratio': info.get('priceToBook', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'roe': info.get('returnOnEquity', 0),
                'roa': info.get('returnOnAssets', 0),
                'profit_margin': info.get('profitMargins', 0),
                'revenue_growth': info.get('revenueGrowth', 0),
                'earnings_growth': info.get('earningsGrowth', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'peg_ratio': info.get('pegRatio', 0),
                'beta': info.get('beta', 0),
                'book_value': info.get('bookValue', 0),
                'price_to_sales': info.get('priceToSalesTrailing12Months', 0),
                'enterprise_value': info.get('enterpriseValue', 0),
                'ebitda': info.get('ebitda', 0),
                'free_cashflow': info.get('freeCashflow', 0),
                'operating_cashflow': info.get('operatingCashflow', 0),
                'recommendation': info.get('recommendationMean', 0),
                'target_price': info.get('targetMeanPrice', 0),
                '52_week_high': info.get('fiftyTwoWeekHigh', 0),
                '52_week_low': info.get('fiftyTwoWeekLow', 0),
            }
            
            # Calculate additional metrics
            if len(history) > 0:
                current_price = history['Close'].iloc[-1]
                fundamentals['price_momentum_3m'] = (current_price / history['Close'].iloc[-63]) - 1 if len(history) >= 63 else 0
                fundamentals['price_momentum_1y'] = (current_price / history['Close'].iloc[-252]) - 1 if len(history) >= 252 else 0
                fundamentals['volatility'] = history['Close'].pct_change().std() * np.sqrt(252)
                fundamentals['avg_volume'] = history['Volume'].mean()
            
            return fundamentals
            
        except Exception as e:
            print(f"Error getting fundamentals for {symbol}: {e}")
            return {}
    
    def scrape_financial_news(self, symbol: str) -> List[Dict]:
        """Scrape recent financial news for sentiment analysis"""
        news_data = []
        try:
            # Yahoo Finance news
            url = f"https://finance.yahoo.com/quote/{symbol}/news"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Look for news headlines
                headlines = soup.find_all(['h3', 'h4'], class_=lambda x: x and ('headline' in x.lower() or 'title' in x.lower()))
                
                for headline in headlines[:10]:  # Limit to 10 headlines
                    text = headline.get_text().strip()
                    if text and len(text) > 10:
                        news_data.append({
                            'headline': text,
                            'source': 'Yahoo Finance',
                            'date': datetime.datetime.now().strftime('%Y-%m-%d')
                        })
            
            time.sleep(1)  # Be respectful to servers
            
        except Exception as e:
            print(f"Error scraping news for {symbol}: {e}")
        
        return news_data
    
    def calculate_investment_score(self, fundamentals: Dict) -> Tuple[float, Dict]:
        """Calculate investment opportunity score based on fundamental analysis"""
        score = 0
        criteria = {}
        
        try:
            # Valuation metrics (30% weight)
            pe_ratio = fundamentals.get('pe_ratio', 0)
            if 0 < pe_ratio < 15:
                score += 30
                criteria['pe_ratio'] = 'Excellent (< 15)'
            elif 15 <= pe_ratio < 25:
                score += 20
                criteria['pe_ratio'] = 'Good (15-25)'
            elif 25 <= pe_ratio < 35:
                score += 10
                criteria['pe_ratio'] = 'Fair (25-35)'
            else:
                criteria['pe_ratio'] = 'Poor (> 35 or negative)'
            
            # Price to Book ratio
            pb_ratio = fundamentals.get('pb_ratio', 0)
            if 0 < pb_ratio < 1.5:
                score += 15
                criteria['pb_ratio'] = 'Excellent (< 1.5)'
            elif 1.5 <= pb_ratio < 3:
                score += 10
                criteria['pb_ratio'] = 'Good (1.5-3)'
            elif 3 <= pb_ratio < 5:
                score += 5
                criteria['pb_ratio'] = 'Fair (3-5)'
            else:
                criteria['pb_ratio'] = 'Poor (> 5)'
            
            # Profitability metrics (25% weight)
            roe = fundamentals.get('roe', 0)
            if roe > 0.15:
                score += 20
                criteria['roe'] = 'Excellent (> 15%)'
            elif roe > 0.10:
                score += 15
                criteria['roe'] = 'Good (10-15%)'
            elif roe > 0.05:
                score += 10
                criteria['roe'] = 'Fair (5-10%)'
            else:
                criteria['roe'] = 'Poor (< 5%)'
            
            profit_margin = fundamentals.get('profit_margin', 0)
            if profit_margin > 0.20:
                score += 15
                criteria['profit_margin'] = 'Excellent (> 20%)'
            elif profit_margin > 0.10:
                score += 10
                criteria['profit_margin'] = 'Good (10-20%)'
            elif profit_margin > 0.05:
                score += 5
                criteria['profit_margin'] = 'Fair (5-10%)'
            else:
                criteria['profit_margin'] = 'Poor (< 5%)'
            
            # Growth metrics (20% weight)
            revenue_growth = fundamentals.get('revenue_growth', 0)
            if revenue_growth > 0.15:
                score += 15
                criteria['revenue_growth'] = 'Excellent (> 15%)'
            elif revenue_growth > 0.10:
                score += 10
                criteria['revenue_growth'] = 'Good (10-15%)'
            elif revenue_growth > 0.05:
                score += 5
                criteria['revenue_growth'] = 'Fair (5-10%)'
            else:
                criteria['revenue_growth'] = 'Poor (< 5%)'
            
            # Financial health (15% weight)
            debt_to_equity = fundamentals.get('debt_to_equity', 0)
            if debt_to_equity < 0.3:
                score += 10
                criteria['debt_to_equity'] = 'Excellent (< 0.3)'
            elif debt_to_equity < 0.6:
                score += 7
                criteria['debt_to_equity'] = 'Good (0.3-0.6)'
            elif debt_to_equity < 1.0:
                score += 4
                criteria['debt_to_equity'] = 'Fair (0.6-1.0)'
            else:
                criteria['debt_to_equity'] = 'Poor (> 1.0)'
            
            # Dividend yield (10% weight)
            dividend_yield = fundamentals.get('dividend_yield', 0)
            if dividend_yield > 0.03:
                score += 10
                criteria['dividend_yield'] = f'Good ({dividend_yield*100:.1f}%)'
            elif dividend_yield > 0.01:
                score += 5
                criteria['dividend_yield'] = f'Fair ({dividend_yield*100:.1f}%)'
            else:
                criteria['dividend_yield'] = 'Low or None'
            
        except Exception as e:
            print(f"Error calculating score: {e}")
        
        return score, criteria
    
    def analyze_stock(self, symbol: str) -> Dict:
        """Comprehensive analysis of a single stock"""
        print(f"Analyzing {symbol}...")
        
        # Get fundamental data
        fundamentals = self.get_stock_fundamentals(symbol)
        if not fundamentals:
            return {}
        
        # Get news sentiment
        news = self.scrape_financial_news(symbol)
        
        # Calculate investment score
        score, criteria = self.calculate_investment_score(fundamentals)
        
        # Determine recommendation
        if score >= 80:
            recommendation = "STRONG BUY"
            risk_level = "Low"
        elif score >= 65:
            recommendation = "BUY"
            risk_level = "Medium-Low"
        elif score >= 50:
            recommendation = "HOLD"
            risk_level = "Medium"
        elif score >= 35:
            recommendation = "WEAK HOLD"
            risk_level = "Medium-High"
        else:
            recommendation = "AVOID"
            risk_level = "High"
        
        analysis = {
            'symbol': symbol,
            'fundamentals': fundamentals,
            'news': news,
            'investment_score': score,
            'recommendation': recommendation,
            'risk_level': risk_level,
            'score_criteria': criteria,
            'analysis_date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return analysis
    
    def analyze_multiple_stocks(self, symbols: List[str]) -> List[Dict]:
        """Analyze multiple stocks and rank by investment potential"""
        analyses = []
        
        for symbol in symbols:
            try:
                analysis = self.analyze_stock(symbol)
                if analysis:
                    analyses.append(analysis)
                time.sleep(2)  # Rate limiting
            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")
                continue
        
        # Sort by investment score
        analyses.sort(key=lambda x: x.get('investment_score', 0), reverse=True)
        
        return analyses
    
    def export_analysis(self, analyses: List[Dict], filename: str = None):
        """Export analysis results to CSV and JSON"""
        if not analyses:
            print("No analysis data to export")
            return
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        if not filename:
            filename = f"investment_analysis_{timestamp}"
        
        # Export to CSV
        csv_file = os.path.join(self.output_dir, f"{filename}.csv")
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Headers
            headers = ['Symbol', 'Company', 'Sector', 'Investment Score', 'Recommendation', 
                      'Risk Level', 'Current Price', 'Market Cap', 'PE Ratio', 'PB Ratio', 
                      'ROE', 'Profit Margin', 'Revenue Growth', 'Debt to Equity', 
                      'Dividend Yield', 'Analysis Date']
            writer.writerow(headers)
            
            # Data rows
            for analysis in analyses:
                fund = analysis.get('fundamentals', {})
                row = [
                    analysis.get('symbol', ''),
                    fund.get('company_name', ''),
                    fund.get('sector', ''),
                    analysis.get('investment_score', 0),
                    analysis.get('recommendation', ''),
                    analysis.get('risk_level', ''),
                    fund.get('current_price', 0),
                    fund.get('market_cap', 0),
                    fund.get('pe_ratio', 0),
                    fund.get('pb_ratio', 0),
                    fund.get('roe', 0),
                    fund.get('profit_margin', 0),
                    fund.get('revenue_growth', 0),
                    fund.get('debt_to_equity', 0),
                    fund.get('dividend_yield', 0),
                    analysis.get('analysis_date', '')
                ]
                writer.writerow(row)
        
        # Export to JSON
        json_file = os.path.join(self.output_dir, f"{filename}.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(analyses, f, indent=2, default=str)
        
        print(f"Analysis exported to {csv_file} and {json_file}")
    
    def generate_investment_report(self, analyses: List[Dict]):
        """Generate a comprehensive investment report"""
        if not analyses:
            print("No data for report generation")
            return
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = os.path.join(self.output_dir, f"investment_report_{timestamp}.txt")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("INVESTMENT OPPORTUNITY ANALYSIS REPORT\n")
            f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            # Top opportunities
            top_picks = [a for a in analyses if a.get('investment_score', 0) >= 65]
            f.write(f"TOP INVESTMENT OPPORTUNITIES ({len(top_picks)} stocks):\n")
            f.write("-" * 50 + "\n")
            
            for i, analysis in enumerate(top_picks, 1):
                fund = analysis.get('fundamentals', {})
                f.write(f"{i}. {fund.get('company_name', 'N/A')} ({analysis.get('symbol', '')})\n")
                f.write(f"   Score: {analysis.get('investment_score', 0)}/100\n")
                f.write(f"   Recommendation: {analysis.get('recommendation', '')}\n")
                f.write(f"   Sector: {fund.get('sector', 'N/A')}\n")
                f.write(f"   Current Price: ${fund.get('current_price', 0):.2f}\n")
                f.write(f"   PE Ratio: {fund.get('pe_ratio', 0):.2f}\n")
                f.write(f"   ROE: {fund.get('roe', 0)*100:.1f}%\n")
                f.write(f"   Revenue Growth: {fund.get('revenue_growth', 0)*100:.1f}%\n")
                f.write("\n")
            
            # Summary statistics
            f.write("\nSUMMARY STATISTICS:\n")
            f.write("-" * 30 + "\n")
            scores = [a.get('investment_score', 0) for a in analyses]
            f.write(f"Total stocks analyzed: {len(analyses)}\n")
            f.write(f"Average investment score: {np.mean(scores):.1f}\n")
            f.write(f"Highest score: {max(scores):.1f}\n")
            f.write(f"Lowest score: {min(scores):.1f}\n")
            
            # Recommendations breakdown
            recommendations = {}
            for analysis in analyses:
                rec = analysis.get('recommendation', 'Unknown')
                recommendations[rec] = recommendations.get(rec, 0) + 1
            
            f.write(f"\nRECOMMENDATIONS BREAKDOWN:\n")
            f.write("-" * 30 + "\n")
            for rec, count in recommendations.items():
                f.write(f"{rec}: {count} stocks\n")
        
        print(f"Investment report generated: {report_file}")

def main():
    """Main execution function"""
    # Initialize analyzer
    analyzer = StockAnalyzer()
    
    # List of stocks to analyze (you can modify this list)
    stock_symbols = [
        "NESTLEIND.NS",  # Original stock
        "RELIANCE.NS",   # Reliance Industries
        "TCS.NS",        # Tata Consultancy Services
        "INFY.NS",       # Infosys
        "HDFCBANK.NS",   # HDFC Bank
        "ICICIBANK.NS",  # ICICI Bank
        "ITC.NS",        # ITC Limited
        "HINDUNILVR.NS", # Hindustan Unilever
        "SBIN.NS",       # State Bank of India
        "BHARTIARTL.NS", # Bharti Airtel
        # Add US stocks
        "AAPL",          # Apple
        "MSFT",          # Microsoft
        "GOOGL",         # Alphabet
        "AMZN",          # Amazon
        "TSLA",          # Tesla
    ]
    
    print("Starting comprehensive stock analysis for long-term investment opportunities...")
    print(f"Analyzing {len(stock_symbols)} stocks...")
    
    # Analyze all stocks
    analyses = analyzer.analyze_multiple_stocks(stock_symbols)
    
    if analyses:
        # Export results
        analyzer.export_analysis(analyses)
        
        # Generate report
        analyzer.generate_investment_report(analyses)
        
        # Print top 5 recommendations
        print("\n" + "="*60)
        print("TOP 5 INVESTMENT OPPORTUNITIES:")
        print("="*60)
        
        for i, analysis in enumerate(analyses[:5], 1):
            fund = analysis.get('fundamentals', {})
            print(f"{i}. {fund.get('company_name', 'N/A')} ({analysis.get('symbol', '')})")
            print(f"   Score: {analysis.get('investment_score', 0)}/100")
            print(f"   Recommendation: {analysis.get('recommendation', '')}")
            print(f"   Risk Level: {analysis.get('risk_level', '')}")
            print(f"   Current Price: ${fund.get('current_price', 0):.2f}")
            print()
    else:
        print("No successful analyses completed.")

if __name__ == "__main__":
    main()
