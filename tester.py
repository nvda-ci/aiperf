import json
from pathlib import Path
from aiperf.common.models.dataset_models import InputsFile

# Try to validate both files
files = [
'/Users/warnold/proj/playground/aiperf_standard_inputs.json',
'/Users/warnold/proj/playground/aiperf_resumed_inputs.json'
]

for filepath in files:
    print(f'\n=== Validating {Path(filepath).name} ===')
    try:
        with open(filepath) as f:
            data = json.load(f)
        inputs_file = InputsFile.model_validate(data)
        print(f'✓ Valid InputsFile format')
        print(f'  - Number of sessions: {len(inputs_file.data)}')
        if inputs_file.data:
            print(f'  - First session ID: {inputs_file.data[0].session_id}')
            print(f'  - First session payloads: {len(inputs_file.data[0].payloads)}')
            if inputs_file.data[0].payloads:
                print(f'  - Sample payload keys: {list(inputs_file.data[0].payloads[0].keys())[:5]}')
    except Exception as e:
        print(f'✗ Validation error: {type(e).__name__}: {e}')
        # Try to get more details
        try:
            with open(filepath) as f:
                data = json.load(f)
            print(f'  File structure: {list(data.keys())}')
            if 'data' in data and len(data['data']) > 0:
                print(f'  First session keys: {list(data["data"][0].keys())}')
        except Exception as e2:
            print(f'  Error reading file: {e2}')
