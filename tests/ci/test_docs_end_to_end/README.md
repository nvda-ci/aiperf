<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Adding New End-to-End Tests for Documentation Examples

## IMPORTANT: Code Examples in This File

**The bash code examples in this documentation use backslashes (`\`) before the triple backticks** to prevent them from being parsed as actual test commands by the test framework.

**When copying examples from this file, you MUST remove the backslashes (`\`) before using them.**

For example, this file shows examples like `\```bash` but you should write `​```bash` (without the backslash).

---

This guide explains how to add new end-to-end tests for server examples in the AIPerf documentation.

## Overview

The end-to-end test framework automatically discovers and tests server examples from markdown documentation files. It:
1. Parses markdown files for specially tagged bash commands
2. Builds an AIPerf Docker container
3. For each discovered server:
   - Runs the server setup command
   - Waits for the server to become healthy
   - Executes AIPerf benchmark commands
   - Validates results and cleans up

## How Tests Are Discovered

The test parser (`parser.py`) scans all markdown files (`*.md`) in the repository and looks for HTML comment tags with specific patterns:

- **Setup commands**: `<!-- setup-{server-name}-endpoint-server -->`
- **Health checks**: `<!-- health-check-{server-name}-endpoint-server -->`
- **AIPerf commands**: `<!-- aiperf-run-{server-name}-endpoint-server -->`

Each tag must be followed by a bash code block (` ```bash ... ``` `) containing the actual command.

## Adding a New Server Test

To add tests for a new server, you need to add three types of tagged commands to your documentation:

### 1. Server Setup Command

Tag the bash command that starts your server:

```markdown
<!-- setup-myserver-endpoint-server -->
\```bash
# Start your server
docker run --gpus all -p 8000:8000 myserver/image:latest \
  --model my-model \
  --host 0.0.0.0 --port 8000
\```
<!-- /setup-myserver-endpoint-server -->
```

**Important notes:**
- The server name (`myserver` in this example) must be consistent across all three tag types
- The setup command runs in the background
- The command should start a long-running server process
- Use port 8000 or ensure your health check targets the correct port

### 2. Health Check Command

Tag a bash command that waits for your server to be ready:

```markdown
<!-- health-check-myserver-endpoint-server -->
\```bash
timeout 900 bash -c 'while [ "$(curl -s -o /dev/null -w "%{http_code}" localhost:8000/health -H "Content-Type: application/json")" != "200" ]; do sleep 2; done' || { echo "Server not ready after 15min"; exit 1; }
\```
<!-- /health-check-myserver-endpoint-server -->
```

**Important notes:**
- The health check should poll the server until it responds successfully
- Use a reasonable timeout (e.g., 900 seconds = 15 minutes)
- The command must exit with code 0 when the server is healthy
- The command must exit with non-zero code if the server fails to start

### 3. AIPerf Run Commands

Tag one or more AIPerf benchmark commands:

```markdown
<!-- aiperf-run-myserver-endpoint-server -->
\```bash
aiperf profile \
    --model my-model \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --service-kind openai \
    --streaming \
    --num-prompts 10 \
    --max-tokens 100
\```
<!-- /aiperf-run-myserver-endpoint-server -->
```

You can have multiple `aiperf-run` commands for the same server. Each will be executed sequentially:

```markdown
<!-- aiperf-run-myserver-endpoint-server -->
\```bash
# First test: streaming mode
aiperf profile \
    --model my-model \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --service-kind openai \
    --streaming \
    --num-prompts 10
\```
<!-- /aiperf-run-myserver-endpoint-server -->

<!-- aiperf-run-myserver-endpoint-server -->
\```bash
# Second test: non-streaming mode
aiperf profile \
    --model my-model \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --service-kind openai \
    --num-prompts 10
\```
<!-- /aiperf-run-myserver-endpoint-server -->
```

**Important notes:**
- Do NOT include `--ui-type` flag - the test framework adds `--ui-type simple` automatically
- Each command is executed inside the AIPerf Docker container
- Commands should complete in a reasonable time (default timeout: 300 seconds)
- Use small values for `--num-prompts` and `--max-tokens` to keep tests fast

## Complete Example

Here's a complete example for a new server called "fastapi":

```markdown
### Running FastAPI Server

Start the FastAPI server:

<!-- setup-fastapi-endpoint-server -->
\```bash
docker run --gpus all -p 8000:8000 mycompany/fastapi-llm:latest \
  --model-name meta-llama/Llama-3.2-1B \
  --host 0.0.0.0 \
  --port 8000
\```
<!-- /setup-fastapi-endpoint-server -->

Wait for the server to be ready:

<!-- health-check-fastapi-endpoint-server -->
\```bash
timeout 600 bash -c 'while [ "$(curl -s -o /dev/null -w "%{http_code}" localhost:8000/v1/models)" != "200" ]; do sleep 2; done' || { echo "FastAPI server not ready after 10min"; exit 1; }
\```
<!-- /health-check-fastapi-endpoint-server -->

Profile the model:

<!-- aiperf-run-fastapi-endpoint-server -->
\```bash
aiperf profile \
    --model meta-llama/Llama-3.2-1B \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --service-kind openai \
    --streaming \
    --num-prompts 20 \
    --max-tokens 50
\```
<!-- /aiperf-run-fastapi-endpoint-server -->
```

## Running the Tests

### Run all discovered tests:

```bash
cd tests/ci/test_docs_end_to_end
python main.py
```

### Dry run to see what would be tested:

```bash
python main.py --dry-run
```

### Test specific servers:

Currently, the framework tests the first discovered server by default. Use `--all-servers` to test all:

```bash
python main.py --all-servers
```

## Validation Rules

The test framework validates that each server has:
- Exactly ONE setup command (duplicates cause test failure)
- Exactly ONE health check command (duplicates cause test failure)
- At least ONE aiperf command

If any of these requirements are not met, the tests will fail with a clear error message.

## Test Execution Flow

For each server, the test runner:

1. **Build Phase**: Builds the AIPerf Docker container (once for all tests)
2. **Setup Phase**: Starts the server in the background
3. **Health Check Phase**: Waits for server to be ready (runs in parallel with setup)
4. **Test Phase**: Executes all AIPerf commands sequentially
5. **Cleanup Phase**: Gracefully shuts down the server and cleans up Docker resources

## Common Patterns

### Pattern: OpenAI-compatible API

```markdown
<!-- setup-myserver-endpoint-server -->
\```bash
docker run --gpus all -p 8000:8000 myserver:latest \
  --model model-name \
  --host 0.0.0.0 --port 8000
\```
<!-- /setup-myserver-endpoint-server -->

<!-- health-check-myserver-endpoint-server -->
\```bash
timeout 900 bash -c 'while [ "$(curl -s -o /dev/null -w "%{http_code}" localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"model-name\",\"messages\":[{\"role\":\"user\",\"content\":\"test\"}],\"max_tokens\":1}")" != "200" ]; do sleep 2; done' || { echo "Server not ready"; exit 1; }
\```
<!-- /health-check-myserver-endpoint-server -->

<!-- aiperf-run-myserver-endpoint-server -->
\```bash
aiperf profile \
    --model model-name \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --service-kind openai \
    --streaming \
    --num-prompts 10 \
    --max-tokens 100
\```
<!-- /aiperf-run-myserver-endpoint-server -->
```

## Troubleshooting

### Tests not discovered

- Verify tag format: `setup-{name}-endpoint-server`, `health-check-{name}-endpoint-server`, `aiperf-run-{name}-endpoint-server`
- Ensure bash code block immediately follows the tag
- Check that the server name is consistent across all three tag types
- Run `python main.py --dry-run` to see what's discovered

### Health check timeout

- Increase the timeout value in your health check command
- Verify the health check endpoint is correct
- Check server logs: the test runner shows setup output for 30 seconds
- Ensure your server starts on the expected port

### AIPerf command fails

- Test your AIPerf command manually first
- Use small values for `--num-prompts` and `--max-tokens`
- Verify the model name matches what the server expects
- Check that the endpoint URL is correct

### Duplicate command errors

If you see errors like "DUPLICATE SETUP COMMAND", you have multiple commands with the same server name:
- Search your docs for all instances of that tag
- Ensure each server has a unique name
- Or remove duplicate tags if they're truly duplicates

## Best Practices

1. **Keep tests fast**: Use minimal `--num-prompts` (10-20) and small `--max-tokens` values
2. **Use standard ports**: Default to 8000 for consistency
3. **Add timeouts**: Always include timeouts in health checks
4. **Test locally first**: Run commands manually before adding tags
5. **One server per doc section**: Avoid mixing multiple servers in the same doc section
6. **Clear error messages**: Include helpful error messages in health checks
7. **Document requirements**: Note any GPU, memory, or dependency requirements in surrounding text

## Architecture Reference

Key files in the test framework:

- `main.py`: Entry point, orchestrates parsing and testing
- `parser.py`: Markdown parser that discovers tagged commands
- `test_runner.py`: Executes tests for each server
- `constants.py`: Configuration constants (timeouts, tag patterns)
- `data_types.py`: Data models for commands and servers
- `utils.py`: Utility functions for Docker operations

## Constants and Configuration

Key constants in `constants.py`:

- `SETUP_MONITOR_TIMEOUT`: 30 seconds (how long to monitor setup output)
- `CONTAINER_BUILD_TIMEOUT`: 600 seconds (Docker build timeout)
- `AIPERF_COMMAND_TIMEOUT`: 300 seconds (per-command timeout)
- `AIPERF_UI_TYPE`: "simple" (auto-added to all aiperf commands)

To modify these, edit `constants.py`.
