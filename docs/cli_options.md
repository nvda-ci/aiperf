<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# aiperf

NVIDIA AIPerf

## Table of Contents

- [`profile`](#aiperf-profile)

**Usage**:

```console
$ aiperf COMMAND
```

**Commands**:

* `profile`: Run the Profile subcommand.

## `aiperf profile`

Run the Profile subcommand.

**Usage**:

```console
$ aiperf profile MODEL-NAMES [ARGS]
```

**Endpoint**:

* `MODEL-NAMES`: Model name(s) to be benchmarked. Can be a comma-separated list or a single model name.  **[required]**
* `MODEL-SELECTION-STRATEGY, --model-selection-strategy`: When multiple models are specified, this is how a specific model should be assigned to a prompt. round_robin: nth prompt
in the list gets assigned to n-mod len(models). random: assignment is uniformly random  *[choices: round-robin, random]*  *[default: round-robin]*
* `CUSTOM-ENDPOINT, --custom-endpoint, --endpoint`: Set a custom endpoint that differs from the OpenAI defaults.
* `ENDPOINT-TYPE, --endpoint-type`: The endpoint type to send requests to on the server.  *[choices: chat, completions, cohere-rankings, embeddings, hf-tei-rankings, huggingface-generate, nim-rankings, solido-rag, template]*  *[default: chat]*
* `STREAMING, --streaming`: An option to enable the use of the streaming API.
* `-u, --url`: URL of the endpoint to target for benchmarking.  *[default: localhost:8000]*
* `REQUEST-TIMEOUT-SECONDS, --request-timeout-seconds`: The timeout in floating-point seconds for each request to the endpoint.  *[default: 600.0]*
* `API-KEY, --api-key`: The API key to use for the endpoint. If provided, it will be sent with every request as a header: Authorization: Bearer
<api_key>.
* `TRANSPORT, --transport, --transport-type`: The transport to use for the endpoint. If not provided, it will be auto-detected from the URL.This can also be used to
force an alternative transport or implementation.  *[choices: http]*

**Input**:

* `EXTRA-INPUTS, --extra-inputs`: Provide additional inputs to include with every request. Inputs should be in an 'input_name:value' format.
Alternatively, a string representing a json formatted dict can be provided.  *[default: []]*
* `-H, --header`: Adds a custom header to the requests. Headers must be specified as 'Header:Value' pairs. Alternatively, a string
representing a json formatted dict can be provided.  *[default: []]*
* `INPUT-FILE, --input-file`: The file or directory path that contains the dataset to use for profiling. This parameter is used in conjunction with
the custom_dataset_type parameter to support different types of user provided datasets.
* `FIXED-SCHEDULE, --fixed-schedule`: Specifies to run a fixed schedule of requests. This is normally inferred from the --input-file parameter, but can be set
manually here.
* `FIXED-SCHEDULE-AUTO-OFFSET, --fixed-schedule-auto-offset`: Specifies to automatically offset the timestamps in the fixed schedule, such that the first timestamp is considered 0,
and the rest are shifted accordingly. If disabled, the timestamps will be assumed to be relative to 0.
* `FIXED-SCHEDULE-START-OFFSET, --fixed-schedule-start-offset`: Specifies the offset in milliseconds to start the fixed schedule at. By default, the schedule starts at 0, but this
option can be used to start at a reference point further in the schedule. This option cannot be used in conjunction with
the --fixed-schedule-auto-offset. The schedule will include any requests at the start offset.
* `FIXED-SCHEDULE-END-OFFSET, --fixed-schedule-end-offset`: Specifies the offset in milliseconds to end the fixed schedule at. By default, the schedule ends at the last timestamp
in the trace dataset, but this option can be used to only run a subset of the trace. The schedule will include any
requests at the end offset.
* `PUBLIC-DATASET, --public-dataset`: The public dataset to use for the requests.  *[choices: sharegpt]*
* `CUSTOM-DATASET-TYPE, --custom-dataset-type`: The type of custom dataset to use. This parameter is used in conjunction with the --input-file parameter. [choices:
single_turn, multi_turn, random_pool, mooncake_trace]
* `DATASET-SAMPLING-STRATEGY, --dataset-sampling-strategy`: The strategy to use for sampling the dataset. sequential: Iterate through the dataset sequentially, then wrap around to
the beginning. random: Randomly select a conversation from the dataset. Will randomly sample with replacement. shuffle:
Shuffle the dataset and iterate through it. Will randomly sample without replacement. Once the end of the dataset is
reached, shuffle the dataset again and start over.  *[choices: sequential, random, shuffle]*
* `RANDOM-SEED, --random-seed`: The seed used to generate random values. Set to some value to make the synthetic data generation deterministic. It will
use system default if not provided.
* `GOODPUT, --goodput`: Specify service level objectives (SLOs) for goodput as space-separated 'KEY:VALUE' pairs, where KEY is a metric tag and
VALUE is a number in the metric’s display unit (falls back to its base unit if no display unit is defined). Examples:
'request_latency:250' (ms), 'inter_token_latency:10' (ms), output_token_throughput_per_user:600 (tokens/s). Only metrics
applicable to the current endpoint/config are considered. For more context on the definition of goodput, refer to
DistServe paper: https://arxiv.org/pdf/2401.09670 and the blog: https://hao-ai-lab.github.io/blogs/distserve
* `RANKINGS-PASSAGES-MEAN, --rankings-passages-mean`: Mean number of passages per rankings entry (per query)(default 1).  *[default: 1]*
* `RANKINGS-PASSAGES-STDDEV, --rankings-passages-stddev`: Stddev for passages per rankings entry (default 0).  *[default: 0]*

**Output**:

* `OUTPUT-ARTIFACT-DIR, --output-artifact-dir, --artifact-dir`: The directory to store all the (output) artifacts generated by AIPerf.  *[default: artifacts]*
* `PROFILE-EXPORT-PREFIX, --profile-export-prefix, --profile-export-file`: The prefix for the profile export file names. Will be suffixed with .csv, .json, .jsonl, and _raw.jsonl.If not provided,
the default profile export file names will be used: profile_export_aiperf.csv, profile_export_aiperf.json,
profile_export.jsonl, and profile_export_raw.jsonl.
* `EXPORT-LEVEL, --export-level, --profile-export-level`: The level of profile export files to create.  *[choices: summary, records, raw]*  *[default: records]*
* `SLICE-DURATION, --slice-duration`: The duration (in seconds) of an individual time slice to be used post-benchmark in time-slicing mode.

**Tokenizer**:

* `TOKENIZER, --tokenizer`: The HuggingFace tokenizer to use to interpret token metrics from prompts and responses. The value can be the name of a
tokenizer or the filepath of the tokenizer. The default value is the model name.
* `TOKENIZER-REVISION, --tokenizer-revision`: The specific model version to use. It can be a branch name, tag name, or commit ID.  *[default: main]*
* `TOKENIZER-TRUST-REMOTE-CODE, --tokenizer-trust-remote-code`: Allows custom tokenizer to be downloaded and executed. This carries security risks and should only be used for
repositories you trust. This is only necessary for custom tokenizers stored in HuggingFace Hub.

**Load Generator**:

* `BENCHMARK-DURATION, --benchmark-duration`: The duration in seconds for benchmarking.
* `BENCHMARK-GRACE-PERIOD, --benchmark-grace-period`: The grace period in seconds to wait for responses after benchmark duration ends. Only applies when --benchmark-duration
is set. Responses received within this period are included in metrics.  *[default: 30.0]*
* `CONCURRENCY, --concurrency`: The concurrency value to benchmark.
* `REQUEST-RATE, --request-rate`: Sets the request rate for the load generated by AIPerf. Unit: requests/second
* `REQUEST-RATE-MODE, --request-rate-mode`: Sets the request rate mode for the load generated by AIPerf. Valid values: constant, poisson. constant: Generate
requests at a fixed rate. poisson: Generate requests using a poisson distribution.  *[default: poisson]*
* `REQUEST-COUNT, --request-count, --num-requests`: The number of requests to use for measurement.  *[default: 10]*
* `WARMUP-REQUEST-COUNT, --warmup-request-count, --num-warmup-requests`: The number of warmup requests to send before benchmarking.  *[default: 0]*
* `REQUEST-CANCELLATION-RATE, --request-cancellation-rate`: The percentage of requests to cancel.  *[default: 0.0]*
* `REQUEST-CANCELLATION-DELAY, --request-cancellation-delay`: The delay in seconds before cancelling requests. This is used when --request-cancellation-rate is greater than 0.  *[default: 0.0]*

**Conversation Input**:

* `CONVERSATION-NUM, --conversation-num, --num-conversations, --num-sessions`: The total number of unique conversations to generate. Each conversation represents a single request session between
client and server. Supported on synthetic mode and the custom random_pool dataset. The number of conversations will be
used to determine the number of entries in both the custom random_pool and synthetic datasets and will be reused until
benchmarking is complete.
* `NUM-DATASET-ENTRIES, --num-dataset-entries, --num-prompts`: The total number of unique dataset entries to generate for the dataset. Each entry represents a single turn used in a
request.  *[default: 100]*
* `CONVERSATION-TURN-MEAN, --conversation-turn-mean, --session-turns-mean`: The mean number of turns within a conversation.  *[default: 1]*
* `CONVERSATION-TURN-STDDEV, --conversation-turn-stddev, --session-turns-stddev`: The standard deviation of the number of turns within a conversation.  *[default: 0]*
* `CONVERSATION-TURN-DELAY-MEAN, --conversation-turn-delay-mean, --session-turn-delay-mean`: The mean delay between turns within a conversation in milliseconds.  *[default: 0.0]*
* `CONVERSATION-TURN-DELAY-STDDEV, --conversation-turn-delay-stddev, --session-turn-delay-stddev`: The standard deviation of the delay between turns within a conversation in milliseconds.  *[default: 0.0]*
* `CONVERSATION-TURN-DELAY-RATIO, --conversation-turn-delay-ratio, --session-delay-ratio`: A ratio to scale multi-turn delays.  *[default: 1.0]*

**Input Sequence Length (ISL)**:

* `PROMPT-INPUT-TOKENS-MEAN, --prompt-input-tokens-mean, --synthetic-input-tokens-mean, --isl`: The mean of number of tokens in the generated prompts when using synthetic data.  *[default: 550]*
* `PROMPT-INPUT-TOKENS-STDDEV, --prompt-input-tokens-stddev, --synthetic-input-tokens-stddev, --isl-stddev`: The standard deviation of number of tokens in the generated prompts when using synthetic data.  *[default: 0.0]*
* `PROMPT-INPUT-TOKENS-BLOCK-SIZE, --prompt-input-tokens-block-size, --synthetic-input-tokens-block-size, --isl-block-size`: The block size of the prompt.  *[default: 512]*
* `SEQ-DIST, --seq-dist, --sequence-distribution`: Sequence length distribution specification for varying ISL/OSL pairs

**Output Sequence Length (OSL)**:

* `PROMPT-OUTPUT-TOKENS-MEAN, --prompt-output-tokens-mean, --output-tokens-mean, --osl`: The mean number of tokens in each output.
* `PROMPT-OUTPUT-TOKENS-STDDEV, --prompt-output-tokens-stddev, --output-tokens-stddev, --osl-stddev`: The standard deviation of the number of tokens in each output.  *[default: 0]*

**Prompt**:

* `-b, --prompt-batch-size, --batch-size-text, --batch-size`: The batch size of text requests AIPerf should send. This is currently supported with the embeddings and rankings
endpoint types  *[default: 1]*

**Prefix Prompt**:

* `PROMPT-PREFIX-POOL-SIZE, --prompt-prefix-pool-size, --prefix-prompt-pool-size, --num-prefix-prompts`: The total size of the prefix prompt pool to select prefixes from. If this value is not zero, these are prompts that are
prepended to input prompts. This is useful for benchmarking models that use a K-V cache.  *[default: 0]*
* `PROMPT-PREFIX-LENGTH, --prompt-prefix-length, --prefix-prompt-length`: The number of tokens in each prefix prompt. This is only used if "num" is greater than zero. Note that due to the prefix
and user prompts being concatenated, the number of tokens in the final prompt may be off by one.  *[default: 0]*

**Audio Input**:

* `AUDIO-BATCH-SIZE, --audio-batch-size, --batch-size-audio`: The batch size of audio requests AIPerf should send. This is currently supported with the OpenAI chat endpoint type  *[default: 1]*
* `AUDIO-LENGTH-MEAN, --audio-length-mean`: The mean length of the audio in seconds.  *[default: 0.0]*
* `AUDIO-LENGTH-STDDEV, --audio-length-stddev`: The standard deviation of the length of the audio in seconds.  *[default: 0.0]*
* `AUDIO-FORMAT, --audio-format`: The format of the audio files (wav or mp3).  *[choices: wav, mp3]*  *[default: wav]*
* `AUDIO-DEPTHS, --audio-depths`: A list of audio bit depths to randomly select from in bits.  *[default: [16]]*
* `AUDIO-SAMPLE-RATES, --audio-sample-rates`: A list of audio sample rates to randomly select from in kHz. Common sample rates are 16, 44.1, 48, 96, etc.  *[default: [16.0]]*
* `AUDIO-NUM-CHANNELS, --audio-num-channels`: The number of audio channels to use for the audio data generation.  *[default: 1]*

**Image Input**:

* `IMAGE-WIDTH-MEAN, --image-width-mean`: The mean width of images when generating synthetic image data.  *[default: 0.0]*
* `IMAGE-WIDTH-STDDEV, --image-width-stddev`: The standard deviation of width of images when generating synthetic image data.  *[default: 0.0]*
* `IMAGE-HEIGHT-MEAN, --image-height-mean`: The mean height of images when generating synthetic image data.  *[default: 0.0]*
* `IMAGE-HEIGHT-STDDEV, --image-height-stddev`: The standard deviation of height of images when generating synthetic image data.  *[default: 0.0]*
* `IMAGE-BATCH-SIZE, --image-batch-size, --batch-size-image`: The image batch size of the requests AIPerf should send.  *[default: 1]*
* `IMAGE-FORMAT, --image-format`: The compression format of the images.  *[choices: png, jpeg, random]*  *[default: png]*

**Video Input**:

* `VIDEO-BATCH-SIZE, --video-batch-size, --batch-size-video`: The video batch size of the requests AIPerf should send.  *[default: 1]*
* `VIDEO-DURATION, --video-duration`: Seconds per clip (default: 5.0).  *[default: 5.0]*
* `VIDEO-FPS, --video-fps`: Frames per second (default/recommended for Cosmos: 4).  *[default: 4]*
* `VIDEO-WIDTH, --video-width`: Video width in pixels.
* `VIDEO-HEIGHT, --video-height`: Video height in pixels.
* `VIDEO-SYNTH-TYPE, --video-synth-type`: Synthetic generator type.  *[choices: moving-shapes, grid-clock]*  *[default: moving-shapes]*
* `VIDEO-FORMAT, --video-format`: The video format of the generated files.  *[choices: mp4, webm]*  *[default: webm]*
* `VIDEO-CODEC, --video-codec`: The video codec to use for encoding. Common options: libvpx-vp9 (CPU, BSD-licensed, default for WebM), libx264 (CPU,
GPL-licensed, widely compatible), libx265 (CPU, GPL-licensed, smaller files), h264_nvenc (NVIDIA GPU), hevc_nvenc
(NVIDIA GPU, smaller files). Any FFmpeg-supported codec can be used.  *[default: libvpx-vp9]*

**Service**:

* `LOG-LEVEL, --log-level`: Logging level  *[choices: trace, debug, info, notice, warning, success, error, critical]*  *[default: info]*
* `-v, --verbose`: Equivalent to --log-level DEBUG. Enables more verbose logging output, but lacks some raw message logging.
* `-vv, --extra-verbose`: Equivalent to --log-level TRACE. Enables the most verbose logging output possible.
* `RECORD-PROCESSOR-SERVICE-COUNT, --record-processor-service-count, --record-processors`: Number of services to spawn for processing records. The higher the request rate, the more services should be spawned in
order to keep up with the incoming records. If not specified, the number of services will be automatically determined
based on the worker count.
* `UI-TYPE, --ui-type, --ui`: Type of UI to use  *[choices: none, simple, dashboard]*  *[default: dashboard]*

**Telemetry**:

* `GPU-TELEMETRY, --gpu-telemetry`: Enable GPU telemetry console display and optionally specify: (1) 'dashboard' for realtime dashboard mode, (2) custom
DCGM exporter URLs (e.g., http://node1:9401/metrics), (3) custom metrics CSV file (e.g., custom_gpu_metrics.csv).
Default endpoints localhost:9400 and localhost:9401 are always attempted. Example: --gpu-telemetry dashboard node1:9400
custom.csv

**Workers**:

* `WORKERS-MAX, --workers-max, --max-workers`: Maximum number of workers to create. If not specified, the number of workers will be determined by the formula
min(concurrency, (num CPUs * 0.75) - 1),  with a default max cap of 32. Any value provided will still be capped by the
concurrency value (if specified), but not by the max cap.

**ZMQ Communication**:

* `ZMQ-HOST, --zmq-host`: Host address for TCP connections  *[default: 127.0.0.1]*
* `ZMQ-IPC-PATH, --zmq-ipc-path`: Path for IPC sockets
