#!/usr/bin/env python3
"""
Launcher script for the Biomedical Knowledge Graph Platform
"""
import subprocess
import sys
import os

def main():
    """Launch the main app with multipage navigation"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_app_path = os.path.join(script_dir, "homepage.py")
    
    print("🚀 Starting Biomedical Knowledge Graph Platform...")
    print("📱 The app will open in your default web browser")
    print("�� Make sure your Neo4j database is running!")
    print("-" * 50)
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", main_app_path], check=True)
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running app: {e}")
        print("💡 Make sure you have streamlit installed: pip install streamlit")

if __name__ == "__main__":
    main()