#!/bin/bash

# Step 1: Setup crawl4ai dependencies (installs Chromium, etc.)
echo "🔧 Running crawl4ai setup..."
crew4ai setup

# Step 2: Start the Telegram bot
echo "🚀 Starting bot..."
python3 main3.py
