#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Test Chrome in distroless container
# This script can be run inside a distroless container to test Chrome workarounds

set -e

echo "========================================================================"
echo "Testing Chrome in Distroless Environment"
echo "========================================================================"
echo ""

# Check if Chrome exists
if [ -f "/opt/chrome/chrome" ]; then
    CHROME="/opt/chrome/chrome"
    echo "✅ Found Chrome at: $CHROME"
elif [ -f "/usr/bin/google-chrome" ]; then
    CHROME="/usr/bin/google-chrome"
    echo "✅ Found Chrome at: $CHROME"
else
    echo "❌ Chrome not found"
    exit 1
fi

# Test Chrome version
echo "Testing Chrome version..."
$CHROME --version || echo "❌ Failed to get Chrome version"
echo ""

# Test 1: Chrome with --remote-debugging-pipe (Kaleido's default)
echo "========================================================================"
echo "TEST 1: Chrome with --remote-debugging-pipe (Kaleido's approach)"
echo "========================================================================"
echo "Starting Chrome with pipe-based debugging..."
timeout 5 $CHROME --headless=new --remote-debugging-pipe --no-sandbox --disable-dev-shm-usage about:blank &
PIPE_PID=$!
sleep 2

if ps -p $PIPE_PID > /dev/null 2>&1; then
    echo "⚠️  Chrome is running with --remote-debugging-pipe"
    
    # Check file descriptors
    if [ -d "/proc/$PIPE_PID/fd" ]; then
        echo "File descriptors for Chrome process:"
        ls -la /proc/$PIPE_PID/fd/ 2>/dev/null || echo "Cannot read file descriptors"
    fi
    
    kill $PIPE_PID 2>/dev/null || true
    wait $PIPE_PID 2>/dev/null || true
else
    echo "❌ Chrome exited immediately with --remote-debugging-pipe"
fi
echo ""

# Test 2: Chrome with --remote-debugging-port (Network-based approach)
echo "========================================================================"
echo "TEST 2: Chrome with --remote-debugging-port (Network approach)"
echo "========================================================================"
echo "Starting Chrome with network-based debugging..."
$CHROME --headless=new --remote-debugging-port=9222 --no-sandbox --disable-dev-shm-usage about:blank &
PORT_PID=$!
sleep 3

if ps -p $PORT_PID > /dev/null 2>&1; then
    echo "✅ Chrome is running with --remote-debugging-port"
    
    # Try to connect
    echo "Attempting to connect to Chrome DevTools Protocol..."
    
    # Using Python to make HTTP request if available
    if command -v python3 &> /dev/null; then
        python3 << 'PYEOF'
import socket
import sys

try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    result = sock.connect_ex(('localhost', 9222))
    sock.close()
    
    if result == 0:
        print("✅ Successfully connected to port 9222!")
        print("   Chrome DevTools Protocol is accessible")
        sys.exit(0)
    else:
        print("❌ Port 9222 is not accessible")
        sys.exit(1)
except Exception as e:
    print(f"❌ Connection failed: {e}")
    sys.exit(1)
PYEOF
        CONNECT_RESULT=$?
    else
        echo "⚠️  Python not available to test connection"
        CONNECT_RESULT=2
    fi
    
    kill $PORT_PID 2>/dev/null || true
    wait $PORT_PID 2>/dev/null || true
    
    if [ $CONNECT_RESULT -eq 0 ]; then
        echo ""
        echo "🎉 SUCCESS! Chrome with --remote-debugging-port works in distroless!"
        echo ""
        echo "This means we can:"
        echo "  1. Patch Kaleido to use --remote-debugging-port"
        echo "  2. Use pyppeteer/playwright for Chrome control"
        echo "  3. Write a custom Chrome DevTools Protocol wrapper"
    elif [ $CONNECT_RESULT -eq 1 ]; then
        echo ""
        echo "⚠️  Chrome runs but DevTools Protocol is not accessible"
        echo "This needs further investigation"
    fi
else
    echo "❌ Chrome exited immediately with --remote-debugging-port"
    echo "Both approaches failed - distroless may need a different base image"
fi
echo ""

echo "========================================================================"
echo "Environment Information"
echo "========================================================================"
echo "Checking /proc/self/fd..."
ls -la /proc/self/fd/ 2>/dev/null || echo "Cannot read /proc/self/fd"
echo ""

echo "Checking system calls availability..."
uname -a || echo "uname not available"

echo ""
echo "========================================================================"
echo "Test Complete"
echo "========================================================================"

