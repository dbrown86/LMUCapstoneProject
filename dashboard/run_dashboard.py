#!/usr/bin/env python3
"""
Dashboard Startup Script
Easy way to run the donor prediction dashboard
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Start the Streamlit dashboard"""
    
    # Get the dashboard directory
    dashboard_dir = Path(__file__).parent
    
    # Change to dashboard directory
    os.chdir(dashboard_dir)
    
    print("🎯 Starting Donor Prediction Dashboard...")
    print("=" * 50)
    print("Dashboard will be available at: http://localhost:8501")
    print("Press Ctrl+C to stop the dashboard")
    print("=" * 50)
    
    try:
        # Run Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you're in the project root directory")
        print("2. Install requirements: pip install -r dashboard/requirements.txt")
        print("3. Run the model pipeline first: python scripts/advanced_multimodal_ensemble.py")

if __name__ == "__main__":
    main()

