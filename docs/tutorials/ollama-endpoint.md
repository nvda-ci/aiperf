<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Ollama Generate Endpoint

The Ollama generate endpoint enables benchmarking of [Ollama](https://ollama.com/) models using the `/api/generate` endpoint. It supports both streaming and non-streaming text generation with full access to Ollama's configuration options.

## When to Use

Use the `ollama_generate` endpoint when:
- Benchmarking models running on Ollama
- You need access to Ollama-specific features like system prompts, JSON formatting, or raw mode
- You want to test Ollama's streaming capabilities

## Basic Example

Benchmark an Ollama model with default settings:

```bash
aiperf profile \
  --model llama2 \
  --url http://localhost:11434 \
  --endpoint-type ollama_generate \
  --synthetic-input-tokens-mean 100 \
  --output-tokens-mean 50 \
  --concurrency 4 \
  --request-count 20
```

## Configuration

Configure the endpoint using `--extra-inputs` for Ollama-specific options:

### Top-Level Parameters

- **`system`**: System prompt to guide model behavior
- **`format`**: Output format (`"json"` or a JSON schema)
- **`raw`**: Skip prompt templating (boolean)
- **`keep_alive`**: Model persistence duration (e.g., `"5m"`, `"1h"`)
- **`images`**: List of base64-encoded images for vision models

### Model Options

Pass model parameters using the `options` object:

- **`temperature`**: Sampling temperature (0.0-2.0)
- **`top_p`**: Nucleus sampling threshold
- **`top_k`**: Top-k sampling limit
- **`seed`**: Random seed for reproducibility
- **`num_ctx`**: Context window size
- **`stop`**: Stop sequences

## Examples

### Basic Text Generation

```bash
aiperf profile \
  --model llama2 \
  --url http://localhost:11434 \
  --endpoint-type ollama_generate \
  --synthetic-input-tokens-mean 200 \
  --output-tokens-mean 100 \
  --concurrency 8 \
  --request-count 50
```

### With System Prompt

```bash
aiperf profile \
  --model mistral \
  --url http://localhost:11434 \
  --endpoint-type ollama_generate \
  --extra-inputs system:"You are a helpful AI assistant" \
  --synthetic-input-tokens-mean 150 \
  --output-tokens-mean 75 \
  --concurrency 4 \
  --request-count 25
```

### With Model Options

```bash
aiperf profile \
  --model llama2 \
  --url http://localhost:11434 \
  --endpoint-type ollama_generate \
  --extra-inputs options:'{
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "seed": 42
  }' \
  --synthetic-input-tokens-mean 100 \
  --output-tokens-mean 50 \
  --concurrency 6 \
  --request-count 30
```

### JSON Mode

Force structured JSON output:

```bash
aiperf profile \
  --model llama2 \
  --url http://localhost:11434 \
  --endpoint-type ollama_generate \
  --extra-inputs format:json \
  --extra-inputs system:"Return responses as valid JSON" \
  --synthetic-input-tokens-mean 100 \
  --output-tokens-mean 50 \
  --concurrency 4 \
  --request-count 20
```

### Streaming Mode

Enable streaming for token-by-token generation:

```bash
aiperf profile \
  --model llama2 \
  --url http://localhost:11434 \
  --endpoint-type ollama_generate \
  --streaming \
  --synthetic-input-tokens-mean 200 \
  --output-tokens-mean 150 \
  --concurrency 2 \
  --request-count 10
```

### With Custom Keep-Alive

Control how long the model stays in memory:

```bash
aiperf profile \
  --model codellama \
  --url http://localhost:11434 \
  --endpoint-type ollama_generate \
  --extra-inputs keep_alive:10m \
  --synthetic-input-tokens-mean 500 \
  --output-tokens-mean 200 \
  --concurrency 4 \
  --request-count 15
```

### Vision Model (with Images)

Benchmark vision-capable models:

```bash
aiperf profile \
  --model llava \
  --url http://localhost:11434 \
  --endpoint-type ollama_generate \
  --extra-inputs images:'["base64_encoded_image_data"]' \
  --synthetic-input-tokens-mean 100 \
  --output-tokens-mean 50 \
  --concurrency 2 \
  --request-count 10
```

### Complete Configuration

Combine multiple options:

```bash
aiperf profile \
  --model mistral \
  --url http://localhost:11434 \
  --endpoint-type ollama_generate \
  --streaming \
  --extra-inputs system:"You are a technical documentation writer" \
  --extra-inputs format:json \
  --extra-inputs keep_alive:5m \
  --extra-inputs options:'{
    "temperature": 0.3,
    "top_p": 0.95,
    "seed": 123,
    "num_ctx": 4096
  }' \
  --synthetic-input-tokens-mean 300 \
  --output-tokens-mean 200 \
  --concurrency 4 \
  --request-count 50
```

## Response Handling

The endpoint automatically:
- Extracts generated text from the `response` field
- Parses token counts when `done: true`:
  - `prompt_eval_count` → `prompt_tokens`
  - `eval_count` → `completion_tokens`
  - Calculates `total_tokens`
- Handles streaming chunks progressively

## Tips

- **Use `--streaming`** to benchmark Ollama's streaming performance
- **Set `keep_alive`** to avoid model reload overhead between requests
- **Use `format:json`** with a system prompt for structured output
- **Set `raw:true`** to skip Ollama's automatic prompt templating
- **Use `-v` or `-vv`** to see detailed request/response logs
- **Check `artifacts/<run-name>/`** for detailed metrics

## Troubleshooting

**Model not responding**
- Verify Ollama is running: `ollama list`
- Check the base URL is correct (default: `http://localhost:11434`)

**Slow performance**
- Increase `keep_alive` to keep the model in memory
- Reduce concurrency if you're hitting resource limits

**Invalid JSON responses**
- Add a system prompt when using `format:json`
- Not all models support JSON mode equally well

**Token counts missing**
- Token counts only appear in the final response when `done: true`
- Check the model supports token counting

## API Reference

For complete Ollama API documentation, see:
- [Ollama Generate API](https://docs.ollama.com/api/generate)
