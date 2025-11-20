#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test script to explore Chrome workarounds for Kaleido in distroless environment.

Tests:
1. Can Chrome run with --remote-debugging-port instead of --remote-debugging-pipe?
2. Can we communicate with Chrome over HTTP/WebSocket?
3. Can we patch Kaleido to use port-based debugging?
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests


def test_chrome_remote_debugging_port():
    """Test if Chrome works with --remote-debugging-port."""
    print("=" * 70)
    print("TEST 1: Chrome with --remote-debugging-port")
    print("=" * 70)

    # Find Chrome executable
    chrome_paths = [
        "/usr/bin/google-chrome",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
        str(Path.home() / ".local/bin/chrome"),
        # Kaleido's Chrome location
        str(
            Path(sys.executable).parent.parent
            / "lib"
            / "python3.13"
            / "site-packages"
            / "kaleido"
            / "executable"
            / "chrome"
        ),
    ]

    chrome_bin = None
    for path in chrome_paths:
        if os.path.exists(path):
            chrome_bin = path
            print(f"✅ Found Chrome at: {chrome_bin}")
            break

    if not chrome_bin:
        print("❌ Chrome not found in any expected location")
        return False

    # Test Chrome version
    try:
        result = subprocess.run(
            [chrome_bin, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        print(f"Chrome version: {result.stdout.strip()}")
    except Exception as e:
        print(f"❌ Failed to get Chrome version: {e}")
        return False

    # Start Chrome with remote debugging port
    print("\nStarting Chrome with --remote-debugging-port=9222...")
    try:
        proc = subprocess.Popen(
            [
                chrome_bin,
                "--headless=new",
                "--remote-debugging-port=9222",
                "--no-sandbox",  # Often needed in containers
                "--disable-dev-shm-usage",  # Helps with limited /dev/shm
                "--disable-gpu",
                "--disable-software-rasterizer",
                "about:blank",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Give Chrome time to start
        time.sleep(2)

        # Check if Chrome is still running
        if proc.poll() is not None:
            stdout, stderr = proc.communicate()
            print(f"❌ Chrome exited immediately")
            print(f"STDOUT: {stdout.decode()}")
            print(f"STDERR: {stderr.decode()}")
            return False

        print("✅ Chrome started successfully")

        # Try to connect to Chrome DevTools Protocol
        print("\nTrying to connect to Chrome DevTools Protocol...")
        try:
            response = requests.get("http://localhost:9222/json/version", timeout=5)
            if response.status_code == 200:
                version_info = response.json()
                print("✅ Successfully connected to Chrome DevTools Protocol!")
                print(f"   Browser: {version_info.get('Browser')}")
                print(f"   Protocol Version: {version_info.get('Protocol-Version')}")
                print(f"   WebSocket URL: {version_info.get('webSocketDebuggerUrl')}")
                result = True
            else:
                print(f"❌ HTTP {response.status_code} from Chrome")
                result = False
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to connect: {e}")
            result = False

        # Cleanup
        proc.terminate()
        proc.wait(timeout=5)
        return result

    except Exception as e:
        print(f"❌ Failed to start Chrome: {e}")
        return False


def test_kaleido_chrome_location():
    """Check where Kaleido expects to find Chrome."""
    print("\n" + "=" * 70)
    print("TEST 2: Kaleido Chrome Location")
    print("=" * 70)

    try:
        import kaleido

        # Get Kaleido's executable path
        import kaleido.scopes.plotly as kaleido_plotly

        scope = kaleido_plotly.PlotlyScope()

        print(f"Kaleido version: {kaleido.__version__}")
        print(f"Kaleido scope chromium: {scope.chromium}")

        if hasattr(scope, "_chromium_args"):
            print(f"Chromium args: {scope._chromium_args}")

        return True
    except ImportError:
        print("❌ Kaleido not installed")
        return False
    except Exception as e:
        print(f"❌ Error inspecting Kaleido: {e}")
        return False


def test_plotly_export_methods():
    """Test different plotly export methods."""
    print("\n" + "=" * 70)
    print("TEST 3: Plotly Export Methods")
    print("=" * 70)

    try:
        import plotly.graph_objects as go

        # Create a simple figure
        fig = go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[4, 5, 6])])

        print("Testing plotly export methods...")

        # Method 1: write_html (always works, no Chrome needed)
        try:
            html_path = "/tmp/test_plot.html"
            fig.write_html(html_path)
            print(f"✅ HTML export works: {html_path}")
        except Exception as e:
            print(f"❌ HTML export failed: {e}")

        # Method 2: write_image (requires Kaleido/Chrome)
        try:
            png_path = "/tmp/test_plot.png"
            fig.write_image(png_path, width=800, height=600)
            print(f"✅ PNG export works: {png_path}")
            return True
        except Exception as e:
            print(f"❌ PNG export failed: {e}")
            print(f"   This is expected if Kaleido can't communicate with Chrome")
            return False

    except ImportError:
        print("❌ Plotly not installed")
        return False


def test_alternative_png_export():
    """Test alternative methods for PNG export without Kaleido."""
    print("\n" + "=" * 70)
    print("TEST 4: Alternative PNG Export Methods")
    print("=" * 70)

    try:
        import matplotlib.pyplot as plt
        import numpy as np

        # Create a simple matplotlib plot
        fig, ax = plt.subplots(figsize=(8, 6))
        x = np.linspace(0, 10, 100)
        ax.plot(x, np.sin(x), label="sin(x)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Test Plot")
        ax.legend()

        # Export to PNG
        png_path = "/tmp/test_matplotlib.png"
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"✅ Matplotlib PNG export works: {png_path}")
        print(
            "   Matplotlib doesn't need Chrome/Kaleido and works in any environment"
        )
        return True

    except ImportError:
        print("❌ Matplotlib not installed")
        return False
    except Exception as e:
        print(f"❌ Matplotlib export failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("Chrome/Kaleido Workaround Tests")
    print("=" * 70 + "\n")

    results = {
        "Chrome with remote-debugging-port": test_chrome_remote_debugging_port(),
        "Kaleido inspection": test_kaleido_chrome_location(),
        "Plotly export methods": test_plotly_export_methods(),
        "Alternative matplotlib": test_alternative_png_export(),
    }

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")

    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)

    if results["Chrome with remote-debugging-port"]:
        print(
            """
✅ Chrome works with --remote-debugging-port!

This means we have a viable workaround:
1. Patch Kaleido to use --remote-debugging-port instead of --remote-debugging-pipe
2. Or use an alternative library that supports network-based Chrome control
3. The pipe-based IPC issue can be bypassed using TCP sockets

Next steps:
- Patch Kaleido's chromium launcher to use port-based debugging
- Or switch to pyppeteer/playwright for Chrome control
- Or write a lightweight Chrome DevTools Protocol wrapper
"""
        )
    else:
        print(
            """
❌ Chrome with --remote-debugging-port failed

This means:
- The distroless environment may have other limitations beyond pipe IPC
- We may need to consider alternative approaches:
  1. Use matplotlib instead of plotly for PNG generation
  2. Generate HTML plots only (no PNG) in distroless
  3. Use a sidecar container for image rendering
  4. Switch to a less minimal base image
"""
        )

    if results["Alternative matplotlib"]:
        print(
            """
ℹ️  Matplotlib works perfectly in distroless!

Consider:
- Using matplotlib for all PNG generation (no Chrome needed)
- Keep plotly for HTML/interactive visualizations
- Matplotlib can create publication-quality plots with full customization
"""
        )

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

