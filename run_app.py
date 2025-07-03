#!/usr/bin/env python3
"""
Launcher script for the AI-Powered Stock Investment Analyzer
Version: V0.1.1 - Enhanced with SENSEX stocks and Python 3.13 compatibility
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required dependencies"""
    print("Installing required dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--break-system-packages", "-r", "requirements_gui.txt"])

def run_streamlit_app():
    """Run the Streamlit application"""
    print("Starting AI-Powered Stock Investment Analyzer...")
    print("The application will open in your default web browser.")
    print("If it doesn't open automatically, go to: http://localhost:8501")
    print("\nPress Ctrl+C to stop the application.")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "stock_investment_app.py", "--server.port", "8501"])
    except KeyboardInterrupt:
        print("\nApplication stopped.")

def main():
    """Main function"""
    print("=" * 70)
    print("🤖 AI-Powered Stock Investment Analyzer V0.1.1")
    print("📊 Enhanced: SENSEX stocks + Python 3.13 compatibility")
    print("=" * 70)
    
    # Check if requirements file exists
    if not os.path.exists("requirements_gui.txt"):
        print("Error: requirements_gui.txt not found!")
        return
    
    # Check if main app file exists
    if not os.path.exists("stock_investment_app.py"):
        print("Error: stock_investment_app.py not found!")
        return
    
    # Install requirements
    try:
        install_requirements()
    except subprocess.CalledProcessError as e:
        print(f"Failed to install requirements: {e}")
        print("Please install manually: pip install -r requirements_gui.txt")
        return
    
    # Run the app
    run_streamlit_app()

if __name__ == "__main__":
    main()