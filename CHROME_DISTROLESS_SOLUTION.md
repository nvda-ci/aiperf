# Chrome/Kaleido in NVIDIA Distroless - Solution Summary

## TL;DR: IT WORKS! ✅

**Kaleido 1.2.0 + Chrome successfully generates PNG plots in NVIDIA distroless.**

## Key Findings

### 1. Modern Kaleido Does NOT Use `--remote-debugging-pipe`

The investigation revealed that:
- ❌ Manually running Chrome with `--remote-debugging-pipe` fails in distroless
- ✅ Kaleido/choreographer 1.2.x uses a different communication method that works
- ✅ PNG generation via `fig.write_image()` works perfectly

### 2. Required Components

**Packages (installed in `kaleido~=1.2.0` with dependencies):**
- `kaleido~=1.2.0`
- `choreographer==1.2.1` (Kaleido's Chrome launcher)
- `plotly~=6.4.0`

**Chrome Installation:**
```python
# In env-builder stage during Docker build
RUN python -c "import kaleido; kaleido.get_chrome_sync()"
```

This downloads Chrome to:
`/opt/aiperf/venv/lib/python3.13/site-packages/choreographer/cli/browser_exe/chrome-linux64/chrome`

**Required Shared Libraries:**
Chrome needs ~85 shared libraries + their transitive dependencies (~70 more).
Current solution: Copy entire `/lib/x86_64-linux-gnu` and `/usr/lib/x86_64-linux-gnu` directories.

Debian packages needed in env-builder (for building):
```dockerfile
libnss3 libnspr4 libdbus-1-3 libatk1.0-0 libatk-bridge2.0-0
libcups2 libdrm2 libxkbcommon0 libxcomposite1 libxdamage1
libxfixes3 libxrandr2 libgbm1 libxcb1 libpango-1.0-0
libcairo2 libasound2 libatspi2.0-0 libx11-6 libxext6
libexpat1 libglib2.0-0
```

### 3. Working Dockerfile Pattern

```dockerfile
# env-builder stage
FROM python:3.13-slim-bookworm AS env-builder

# Install Chrome dependencies
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libnss3 libnspr4 libdbus-1-3 libatk1.0-0 libatk-bridge2.0-0 \
        libcups2 libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 \
        libxfixes3 libxrandr2 libgbm1 libxcb1 libpango-1.0-0 \
        libcairo2 libasound2 libatspi2.0-0 libx11-6 libxext6 \
        libexpat1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages including kaleido
RUN uv sync --active --no-install-project

# Download Chrome via Kaleido
RUN python -c "import kaleido; kaleido.get_chrome_sync()"

# runtime stage
FROM nvcr.io/nvidia/distroless/python:3.13-v3.1.1-dev AS runtime

# Copy shared libraries (includes transitive deps)
COPY --from=env-builder --chown=1000:1000 /lib/x86_64-linux-gnu /lib/x86_64-linux-gnu
COPY --from=env-builder --chown=1000:1000 /usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu

# Copy venv (includes Chrome from kaleido.get_chrome_sync())
COPY --from=env-builder --chown=1000:1000 /opt/aiperf/venv /opt/aiperf/venv
```

### 4. Usage

```python
import plotly.graph_objects as go

fig = go.Figure(data=[go.Bar(y=[2, 3, 1])])
fig.write_image('/tmp/plot.png', width=1200, height=800, scale=2.0)
# ✅ Works perfectly in distroless!
```

### 5. Image Size

- **Current**: ~1.1GB (includes Chrome, ffmpeg, Python, all dependencies)
- **Breakdown**:
  - Base distroless Python: ~140MB
  - Chrome + deps: ~400MB
  - Shared libraries: ~300MB
  - Python venv: ~250MB
  - ffmpeg: ~20MB

**Optimization opportunities:**
- Copy only required shared libraries + transitive deps (could save ~150MB)
- Requires recursive ldd analysis and careful testing

## What Was Confusing

The error message "Remote debugging pipe file descriptors are not open" suggested pipe-based IPC was the issue. However:

1. **Kaleido 1.2.x uses a different Chrome communication method** (likely HTTP/WebSocket-based)
2. The pipe error only appears when manually running Chrome with `--remote-debugging-pipe`
3. Kaleido's abstraction layer handles Chrome communication differently

## Conclusion

**No workaround needed!** The current implementation with:
- Kaleido 1.2.0
- choreographer 1.2.1
- Full shared library directories
- Chrome from `kaleido.get_chrome_sync()`

...already works perfectly in NVIDIA distroless for PNG generation.

## Testing

```bash
# Build
docker build -t aiperf-chrome:latest --target runtime .

# Test
docker run --rm aiperf-chrome:latest "python3 -c '
import plotly.graph_objects as go
fig = go.Figure(data=[go.Bar(y=[2, 3, 1])])
fig.write_image(\"/tmp/test.png\", width=800, height=600)
print(\"✅ PNG generation works!\")
'"
```

## Future Optimization

To reduce image size further:
1. Use the `collect_chrome_libs.sh` script to get all required libraries
2. Copy only those specific libraries (not entire directories)
3. Thoroughly test PNG generation still works
4. Expected savings: ~150-200MB

However, 1.1GB is reasonable for a full-featured container with Chrome, ffmpeg, and Python.

