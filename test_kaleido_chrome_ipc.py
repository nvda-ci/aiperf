#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test to understand HOW Kaleido communicates with Chrome in distroless.
"""

import os
import subprocess
import sys

print("=" * 70)
print("Testing Kaleido's Chrome Communication Method")
print("=" * 70)

# Test 1: Check what Kaleido is using
print("\n1. Checking Kaleido's Chrome communication method...")
try:
    import kaleido
    import kaleido.scopes.plotly as kaleido_plotly
    
    scope = kaleido_plotly.PlotlyScope()
    print(f"   Kaleido version: {kaleido.__version__}")
    print(f"   Chrome path: {scope.chromium}")
    
    # Check the actual command Kaleido uses
    if hasattr(scope, '_build_proc_args'):
        print("   Kaleido has _build_proc_args method")
    
    # Try to see what arguments Kaleido uses
    import inspect
    if hasattr(scope, '_启动_chromium'):
        source = inspect.getsource(scope._启动_chromium)
        print(f"   Source preview: {source[:200]}...")
    
except Exception as e:
    print(f"   Error inspecting Kaleido: {e}")

# Test 2: Actually use Kaleido to generate a plot
print("\n2. Testing actual Kaleido PNG generation...")
try:
    import plotly.graph_objects as go
    
    fig = go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[4, 5, 6])])
    fig.write_image("/tmp/kaleido_test.png", width=800, height=600)
    print("   ✅ Kaleido PNG generation works!")
    
    # Check if the file exists and has content
    if os.path.exists("/tmp/kaleido_test.png"):
        size = os.path.getsize("/tmp/kaleido_test.png")
        print(f"   Generated PNG size: {size} bytes")
except Exception as e:
    print(f"   ❌ Kaleido PNG generation failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Try to manually test Chrome with pipe
print("\n3. Testing Chrome with --remote-debugging-pipe manually...")
chrome_path = "/opt/aiperf/venv/lib/python3.13/site-packages/choreographer/cli/browser_exe/chrome-linux64/chrome"

if os.path.exists(chrome_path):
    print(f"   Chrome found at: {chrome_path}")
    
    try:
        # Try to start Chrome with pipe-based debugging
        proc = subprocess.Popen(
            [chrome_path, "--headless=new", "--remote-debugging-pipe", 
             "--no-sandbox", "--disable-dev-shm-usage"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
        import time
        time.sleep(1)
        
        if proc.poll() is None:
            print("   ✅ Chrome started with --remote-debugging-pipe")
            print(f"   Chrome PID: {proc.pid}")
            
            # Check file descriptors
            print(f"   Checking /proc/{proc.pid}/fd/...")
            fd_path = f"/proc/{proc.pid}/fd"
            if os.path.exists(fd_path):
                fds = os.listdir(fd_path)
                print(f"   File descriptors: {len(fds)} open")
                for fd in sorted(fds[:10]):  # Show first 10
                    try:
                        link = os.readlink(f"{fd_path}/{fd}")
                        print(f"     fd {fd} -> {link}")
                    except:
                        print(f"     fd {fd} -> (cannot read)")
            
            proc.terminate()
            proc.wait(timeout=5)
            print("   Chrome terminated cleanly")
        else:
            stdout, stderr = proc.communicate()
            print(f"   ❌ Chrome exited immediately")
            print(f"   STDERR: {stderr.decode()[:500]}")
    except Exception as e:
        print(f"   ❌ Error testing Chrome: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"   ❌ Chrome not found at {chrome_path}")

# Test 4: Check choreographer (kaleido's dependency)
print("\n4. Checking choreographer (Kaleido's Chrome launcher)...")
try:
    import choreographer
    print(f"   choreographer version: {choreographer.__version__}")
    
    # Check what method choreographer uses
    import choreographer.browsers.chromium as chromium_module
    print(f"   chromium module: {chromium_module.__file__}")
    
    # Look for the launch method
    if hasattr(chromium_module, 'Chromium'):
        print("   Chromium class found")
        if hasattr(chromium_module.Chromium, 'launch'):
            print("   launch method exists")
            
except Exception as e:
    print(f"   Error checking choreographer: {e}")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print("""
If Kaleido PNG generation works, then either:
1. The pipe-based IPC actually works in this distroless environment
2. Kaleido/choreographer has a fallback mechanism
3. The shared libraries we copied are sufficient for pipe IPC

The fact that it works suggests the NVIDIA distroless environment 
with the copied shared libraries CAN support Chrome's pipe-based IPC!
""")

