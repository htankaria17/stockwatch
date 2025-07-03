#!/usr/bin/env python3
"""
Daily scheduler for automated email notifications
Run this script in the background to send daily investment recommendations
"""

import schedule
import time
import sqlite3
import json
from datetime import datetime, timedelta
from stock_investment_app import (
    DatabaseManager, EnhancedStockAnalyzer, EmailNotifier, UserProfile
)
import logging
import sys
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scheduler.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

class DailyScheduler:
    def __init__(self):
        self.db = DatabaseManager()
        self.analyzer = EnhancedStockAnalyzer()
        self.emailer = EmailNotifier()
        self.logger = logging.getLogger(__name__)
        
        # Email configuration (set these as environment variables for security)
        self.sender_email = os.getenv('SENDER_EMAIL')
        self.sender_password = os.getenv('SENDER_PASSWORD')
        
        if not self.sender_email or not self.sender_password:
            self.logger.warning("Email credentials not found in environment variables.")
            self.logger.warning("Set SENDER_EMAIL and SENDER_PASSWORD environment variables.")
    
    def get_daily_users(self):
        """Get users who want daily notifications"""
        conn = sqlite3.connect(self.db.db_name)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT name, email, country, investment_amount, desired_return, 
                   risk_tolerance, notification_frequency
            FROM users 
            WHERE notification_frequency = 'Daily'
        ''')
        
        users = []
        for row in cursor.fetchall():
            user = UserProfile(
                name=row[0], email=row[1], country=row[2],
                investment_amount=row[3], desired_return=row[4],
                risk_tolerance=row[5], notification_frequency=row[6]
            )
            users.append(user)
        
        conn.close()
        return users
    
    def send_daily_recommendations(self):
        """Send daily recommendations to all daily users"""
        self.logger.info("Starting daily recommendation process...")
        
        if not self.sender_email or not self.sender_password:
            self.logger.error("Email credentials not configured. Skipping email notifications.")
            return
        
        daily_users = self.get_daily_users()
        self.logger.info(f"Found {len(daily_users)} users for daily notifications")
        
        for user in daily_users:
            try:
                self.logger.info(f"Processing recommendations for {user.email}")
                
                # Analyze stocks for user's country
                analyses = self.analyzer.analyze_stocks_for_country(user.country, user)
                
                if analyses:
                    # Send email
                    success = self.emailer.send_daily_recommendations(
                        user.email, analyses, self.sender_email, self.sender_password
                    )
                    
                    if success:
                        self.logger.info(f"Successfully sent recommendations to {user.email}")
                        self._log_email_sent(user.email, "Daily Recommendations", "SUCCESS")
                    else:
                        self.logger.error(f"Failed to send email to {user.email}")
                        self._log_email_sent(user.email, "Daily Recommendations", "FAILED")
                else:
                    self.logger.warning(f"No analysis data available for {user.email}")
                
                # Add delay between users to avoid rate limiting
                time.sleep(30)
                
            except Exception as e:
                self.logger.error(f"Error processing user {user.email}: {e}")
                self._log_email_sent(user.email, "Daily Recommendations", f"ERROR: {e}")
        
        self.logger.info("Daily recommendation process completed")
    
    def _log_email_sent(self, user_email: str, subject: str, status: str):
        """Log email sending status to database"""
        conn = sqlite3.connect(self.db.db_name)
        cursor = conn.cursor()
        
        # Get user ID
        cursor.execute('SELECT id FROM users WHERE email = ?', (user_email,))
        user_row = cursor.fetchone()
        
        if user_row:
            user_id = user_row[0]
            cursor.execute('''
                INSERT INTO email_log (user_id, subject, status)
                VALUES (?, ?, ?)
            ''', (user_id, subject, status))
            conn.commit()
        
        conn.close()
    
    def send_weekly_recommendations(self):
        """Send weekly recommendations (runs on Sundays)"""
        self.logger.info("Starting weekly recommendation process...")
        # Similar to daily but for weekly users
        # Implementation can be added here
        pass
    
    def send_monthly_recommendations(self):
        """Send monthly recommendations (runs on 1st of month)"""
        self.logger.info("Starting monthly recommendation process...")
        # Similar to daily but for monthly users
        # Implementation can be added here
        pass
    
    def cleanup_old_data(self):
        """Clean up old analysis data (keep last 30 days)"""
        self.logger.info("Cleaning up old data...")
        
        conn = sqlite3.connect(self.db.db_name)
        cursor = conn.cursor()
        
        # Delete analysis older than 30 days
        cutoff_date = datetime.now() - timedelta(days=30)
        cursor.execute('''
            DELETE FROM analysis_history 
            WHERE analysis_date < ?
        ''', (cutoff_date.date(),))
        
        # Delete email logs older than 90 days
        cutoff_date_logs = datetime.now() - timedelta(days=90)
        cursor.execute('''
            DELETE FROM email_log 
            WHERE sent_at < ?
        ''', (cutoff_date_logs,))
        
        conn.commit()
        conn.close()
        
        self.logger.info("Data cleanup completed")

def main():
    """Main scheduler function"""
    print("=" * 60)
    print("🕒 Daily Investment Scheduler Starting...")
    print("=" * 60)
    
    scheduler = DailyScheduler()
    
    # Schedule daily recommendations at 8:00 AM
    schedule.every().day.at("08:00").do(scheduler.send_daily_recommendations)
    
    # Schedule weekly recommendations on Sunday at 9:00 AM
    schedule.every().sunday.at("09:00").do(scheduler.send_weekly_recommendations)
    
    # Schedule monthly recommendations on 1st day at 10:00 AM
    schedule.every().month.do(scheduler.send_monthly_recommendations)
    
    # Schedule cleanup every week on Monday at 2:00 AM
    schedule.every().monday.at("02:00").do(scheduler.cleanup_old_data)
    
    # Test mode: uncomment to run immediately for testing
    # scheduler.send_daily_recommendations()
    
    print("Scheduler configured:")
    print("- Daily recommendations: 8:00 AM")
    print("- Weekly recommendations: Sunday 9:00 AM")
    print("- Monthly recommendations: 1st of month 10:00 AM")
    print("- Data cleanup: Monday 2:00 AM")
    print("\nPress Ctrl+C to stop the scheduler")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        print("\nScheduler stopped.")

if __name__ == "__main__":
    main()