#!/usr/bin/env python
"""
Startup script for License Plate Recognition Web Interface
Tập lệnh khởi động cho giao diện web nhận dạng biển số xe
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    print("=" * 60)
    print("🚗 License Plate Recognition Web Interface")
    print("   Giao diện web nhận dạng biển số xe")
    print("=" * 60)
    print()

    # Check if in virtual environment
    venv_path = Path('.venv') if Path('.venv').exists() else Path('venv') if Path('venv').exists() else None
    
    if venv_path:
        print(f"✓ Virtual environment found at: {venv_path}")
    else:
        print("⚠ No virtual environment found")
        print("  Recommended: Create one with 'python -m venv venv'")
        response = input("  Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return

    # Check required files
    required_files = ['web_app.py', 'templates/index.html', 'static/style.css', 'static/app.js']
    print("\n📋 Checking required files...")
    for file in required_files:
        if Path(file).exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} NOT FOUND")
            return

    # Create directories
    print("\n📁 Creating directories...")
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    print("  ✓ uploads/")
    print("  ✓ results/")

    # Install dependencies
    print("\n📦 Installing dependencies...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt', '-q'], check=True)
        print("  ✓ Dependencies installed")
    except subprocess.CalledProcessError:
        print("  ✗ Failed to install dependencies")
        return

    # Start web server
    print("\n" + "=" * 60)
    print("🌐 Starting Web Interface...")
    print("=" * 60)
    print()
    print("📍 Open your browser and go to: http://localhost:5000")
    print()
    print("🎯 Features:")
    print("   • Real-time webcam recognition")
    print("   • Image file processing")
    print("   • Video file processing")
    print("   • Detection history")
    print("   • Real-time statistics")
    print()
    print("⌨️  Press Ctrl+C to stop the server")
    print("=" * 60)
    print()

    try:
        subprocess.run([sys.executable, 'web_app.py'])
    except KeyboardInterrupt:
        print("\n\n✓ Server stopped")

if __name__ == '__main__':
    main()
