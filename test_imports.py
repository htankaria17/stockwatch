#!/usr/bin/env python3
"""
Test script to verify all imports work correctly
Run this before starting the main application
"""

import sys
import importlib

def test_import(module_name, alternative_names=None):
    """Test if a module can be imported"""
    try:
        importlib.import_module(module_name)
        print(f"✅ {module_name} - OK")
        return True
    except ImportError as e:
        if alternative_names:
            for alt_name in alternative_names:
                try:
                    importlib.import_module(alt_name)
                    print(f"✅ {alt_name} (alternative for {module_name}) - OK")
                    return True
                except ImportError:
                    continue
        print(f"❌ {module_name} - FAILED: {e}")
        return False

def main():
    print("Testing Stock Investment App Dependencies...")
    print("=" * 50)
    
    # Test core dependencies
    modules_to_test = [
        'streamlit',
        'yfinance', 
        'pandas',
        'numpy',
        'plotly.graph_objects',
        'plotly.express',
        'sqlite3',
        'smtplib',
        'email.mime.text',
        'email.mime.multipart',
        'requests',
        'bs4',
        'sklearn.ensemble',
        'sklearn.preprocessing',
        'joblib',
        'schedule'
    ]
    
    failed_modules = []
    
    for module in modules_to_test:
        if not test_import(module):
            failed_modules.append(module)
    
    print("\n" + "=" * 50)
    if failed_modules:
        print(f"❌ {len(failed_modules)} modules failed to import:")
        for module in failed_modules:
            print(f"   - {module}")
        print("\nTo install missing dependencies, run:")
        print("pip install -r requirements_gui.txt")
    else:
        print("✅ All modules imported successfully!")
        print("You can now run the stock investment app.")
    
    print(f"\nPython version: {sys.version}")
    print(f"Python executable: {sys.executable}")

if __name__ == "__main__":
    main()