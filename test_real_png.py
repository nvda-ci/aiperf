#!/usr/bin/env python3
import plotly.graph_objects as go
import os

# Test with aiperf's actual parameters
fig = go.Figure(data=[go.Bar(y=[2, 3, 1], name="Test Data")])
fig.update_layout(title="Test Chart", xaxis_title="X", yaxis_title="Y")

output_path = "/tmp/bar_chart.png"
fig.write_image(output_path, width=1200, height=800, scale=2.0)

print("✅ PNG export successful with aiperf parameters!")
print(f"File size: {os.path.getsize(output_path)} bytes")
print(f"File location: {output_path}")

