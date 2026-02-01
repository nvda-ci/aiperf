<!--
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->

# AIPerf Reference Documentation

This directory contains reference documentation with **actual command outputs** and **typed schemas** for all AIPerf commands and artifacts.

## Contents

### [command-quick-reference.md](./command-quick-reference.md)
**Visual guide to command outputs** - Shows exactly what you'll see when running AIPerf commands:
- Command examples with real console output
- Common errors and solutions
- UI types (Dashboard, Simple, None)
- Artifact directory structures
- Quick metric reference table

**Start here** for a quick understanding of what to expect when running commands.

### [command-outputs-and-artifacts.md](./command-outputs-and-artifacts.md)
**Complete technical reference** with typed schemas and real examples:
- Full JSON/JSONL schemas with TypeScript type annotations
- Per-request record format (`profile_export.jsonl`)
- Aggregated statistics format (`profile_export_aiperf.json`)
- GPU telemetry data structures
- Server metrics formats
- CSV export structures
- Real data examples from test fixtures

**Use this** when you need to:
- Parse AIPerf output files programmatically
- Understand exact field types and units
- Integrate with other tools
- Write scripts to process results

## Quick Navigation

| I want to... | Go to... |
|-------------|----------|
| See what a command looks like when I run it | [command-quick-reference.md](./command-quick-reference.md) |
| Understand JSON schema and field types | [command-outputs-and-artifacts.md](./command-outputs-and-artifacts.md) |
| Parse output files in my script | [command-outputs-and-artifacts.md](./command-outputs-and-artifacts.md) → Export File Formats |
| Fix an error message | [command-quick-reference.md](./command-quick-reference.md) → Common Errors |
| Know what files will be generated | [command-quick-reference.md](./command-quick-reference.md) → Output Artifacts |
| Understand metric definitions | [../metrics_reference.md](../metrics_reference.md) |
| Learn CLI options | [../cli_options.md](../cli_options.md) |

## Related Documentation

- **[../metrics_reference.md](../metrics_reference.md)** - Complete metric definitions with formulas
- **[../cli_options.md](../cli_options.md)** - All command-line options
- **[../tutorials/](../tutorials/)** - Step-by-step guides for specific use cases
- **[../environment_variables.md](../environment_variables.md)** - Environment configuration

## Examples

### Example 1: Basic Profile Command

```bash
aiperf profile \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --concurrency 8 \
  --request-count 100
```

See [command-quick-reference.md](./command-quick-reference.md#simple-benchmark-concurrency-mode) for expected output.

### Example 2: Parsing JSON Output

```python
import json

with open("artifacts/run1/profile_export_aiperf.json") as f:
    data = json.load(f)

print(f"Average TTFT: {data['time_to_first_token']['avg']} ms")
print(f"P99 TTFT: {data['time_to_first_token']['p99']} ms")
print(f"Throughput: {data['output_token_throughput']['avg']} tokens/sec")
```

See [command-outputs-and-artifacts.md](./command-outputs-and-artifacts.md#1-aggregated-statistics-json-profile_export_aiperfj son) for full schema.

### Example 3: Reading Per-Request Data

```python
import json

with open("artifacts/run1/profile_export.jsonl") as f:
    for line in f:
        record = json.loads(line)
        request_id = record["metadata"]["x_request_id"]
        ttft = record["metrics"]["time_to_first_token"]["value"]
        print(f"Request {request_id}: TTFT = {ttft} ms")
```

See [command-outputs-and-artifacts.md](./command-outputs-and-artifacts.md#2-per-request-records-jsonl-profile_exportjsonl) for record schema.

---

**Both documents include real examples captured from actual AIPerf runs.**
