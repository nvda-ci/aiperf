#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert InputsFile format to MultiTurn JSONL format for aiperf."""

import json
import sys
from pathlib import Path
from typing import Any

from aiperf.common.models.dataset_models import InputsFile


def extract_text_from_messages(messages: list[dict[str, Any]]) -> str:
    """Extract text content from OpenAI chat messages format.
    
    Args:
        messages: List of message objects with 'role' and 'content' fields
        
    Returns:
        Combined text content from user messages
    """
    texts = []
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            # Handle multimodal content (list of content parts)
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        texts.append(item.get("text", ""))
                    elif "text" in item:
                        texts.append(item["text"])
    return "\n".join(texts)


def convert_inputs_file_to_multiturn_jsonl(
    input_file: Path, output_file: Path
) -> None:
    """Convert InputsFile format to MultiTurn JSONL format.
    
    Args:
        input_file: Path to InputsFile format JSON file
        output_file: Path to write JSONL output file
    """
    print(f"Reading InputsFile from: {input_file}")
    
    # Load and validate InputsFile format
    with open(input_file) as f:
        data = json.load(f)
    
    inputs_file = InputsFile.model_validate(data)
    print(f"Found {len(inputs_file.data)} sessions")
    
    # Convert to MultiTurn format
    multiturn_objects = []
    for session_payloads in inputs_file.data:
        session_id = session_payloads.session_id
        turns = []
        
        for payload in session_payloads.payloads:
            # Extract text from messages
            messages = payload.get("messages", [])
            text = extract_text_from_messages(messages)
            
            if not text:
                print(f"Warning: No text found in payload for session {session_id}")
                continue
            
            # Create a turn object
            turn = {"text": text}
            
            # Add optional fields if present
            if "timestamp" in payload:
                turn["timestamp"] = payload["timestamp"]
            if "delay" in payload:
                turn["delay"] = payload["delay"]
            if "model" in payload:
                turn["model"] = payload["model"]
            if "max_tokens" in payload or "max_completion_tokens" in payload:
                turn["max_tokens"] = payload.get("max_tokens") or payload.get("max_completion_tokens")
            
            turns.append(turn)
        
        if not turns:
            print(f"Warning: No valid turns found for session {session_id}, skipping")
            continue
        
        # Create MultiTurn object
        multiturn = {
            "session_id": session_id,
            "turns": turns
        }
        multiturn_objects.append(multiturn)
    
    print(f"Converted {len(multiturn_objects)} sessions to MultiTurn format")
    
    # Write as JSONL (one JSON object per line)
    print(f"Writing JSONL to: {output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        for multiturn in multiturn_objects:
            f.write(json.dumps(multiturn) + "\n")
    
    print(f"✅ Conversion complete! Wrote {len(multiturn_objects)} lines to {output_file}")
    print(f"\nYou can now use this file with aiperf:")
    print(f"  --input-file {output_file}")
    print(f"  --custom-dataset-type multi_turn")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: convert_inputs_file_to_multiturn.py <input_file> [output_file]")
        print("\nConverts AIPerf InputsFile format to MultiTurn JSONL format.")
        print("\nExample:")
        print("  python convert_inputs_file_to_multiturn.py inputs.json inputs.jsonl")
        sys.exit(1)
    
    input_file = Path(sys.argv[1])
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)
    
    if len(sys.argv) >= 3:
        output_file = Path(sys.argv[2])
    else:
        # Default to same name with .jsonl extension
        output_file = input_file.with_suffix(".jsonl")
    
    try:
        convert_inputs_file_to_multiturn_jsonl(input_file, output_file)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
