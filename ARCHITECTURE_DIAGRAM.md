<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Plugin System Architecture

## The Three-Layer System

```
┌─────────────────────────────────────────────────────────────────┐
│                      APPLICATION STARTUP                        │
│                                                                  │
│  import aiperf.endpoints                                        │
│  EndpointFactory.discover_plugins("aiperf.endpoints")           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────────┐
        │                                             │
        ├──────────────────┬──────────────────────────┤
        │                  │                          │
        ▼                  ▼                          ▼
   ┌─────────┐        ┌──────────┐           ┌────────────┐
   │ LAYER 1 │        │ LAYER 2  │           │  LAYER 3   │
   │ BUILTIN │        │ PLUGINS  │           │  RUNTIME   │
   │REGISTRY │        │ENTRYPTS  │           │ OVERRIDE   │
   └─────────┘        └──────────┘           └────────────┘
        │                  │                        │
        ▼                  ▼                        ▼
    ┌────────────┐    ┌──────────┐        ┌─────────────────┐
    │registry.py │    │pip install       │ use_implementation()
    │            │    │my-plugin│        │                 │
    │register()  │    │          │       │ Anytime override│
    │ × 4 lines  │    │loads at  │       │                 │
    │            │    │startup   │       │ EndpointFactory│
    └────────────┘    └──────────┘       │ .use_impl(...)  │
         ▼                 ▼              └─────────────────┘
    [ChatEndpoint]  [OptimizedChat]              ▼
    [Embeddings]    (if installed)        [Selected:Endpoint]
    [Completions]                         (Active for use)
    [Rankings]

```

---

## How It Works: Step by Step

### Step 1: Builtin Registration (Development)

```
You write:                          System does:
─────────────────────────────────────────────────────────

aiperf/endpoints/registry.py:
  from aiperf.endpoints import ChatEndpoint
  EndpointFactory.register("chat")(ChatEndpoint)

                                    ┌──────────────────┐
                                    │ Registry stores: │
                                    │ "chat" → Chat EP │
                                    │ is_plugin=False  │
                                    └──────────────────┘
```

### Step 2: App Startup

```
import aiperf.endpoints   ──→  imports registry.py  ──→  All builtins registered
                              (__init__.py does this)
```

### Step 3: Plugin Load

```
EndpointFactory.discover_plugins("aiperf.endpoints")

Looks for: aiperf.endpoints.plugins entry points
Finds:     chat → OptimizedChatEndpoint
Registers: "chat" → OptimizedChat, is_plugin=True
           (Same name!)

                                    ┌────────────────────────┐
                                    │ Registry now has:      │
                                    │ "chat" → [            │
                                    │   OptimizedChat (P)   │
                                    │   ChatEndpoint (B)    │
                                    │ ]                      │
                                    │ Selected: Optimized    │
                                    │           (plugin=True)│
                                    └────────────────────────┘
```

### Step 4: Usage

```
EndpointFactory.create_instance("chat", ...)

System: "Get the selected class for 'chat'"
Result: OptimizedChatEndpoint (plugin, highest priority)
```

### Step 5: Runtime Override (Optional)

```
EndpointFactory.use_implementation("chat", ChatEndpoint)

System: "Switch active impl for 'chat' to ChatEndpoint"

                                    ┌────────────────────────┐
                                    │ Registry updated:      │
                                    │ "chat" → [            │
                                    │   OptimizedChat        │
                                    │   ChatEndpoint (B) ✓   │
                                    │ ]                      │
                                    │ Selected: ChatEndpoint │
                                    └────────────────────────┘

Result: Next create_instance("chat") uses ChatEndpoint
```

---

## Discovery Flow

```
Application loads
      │
      ▼
import aiperf.endpoints
      │
      ├─→ __init__.py imports registry
      │        │
      │        ▼
      │    registry.py registers all
      │    (4 lines of registration)
      │        │
      └────────┴─→ All builtins loaded
                    ("chat", "embeddings", "completions", "rankings")

      ▼
EndpointFactory.discover_plugins("aiperf.endpoints")
      │
      ├─→ Checks "aiperf.endpoints.plugins" entry points
      │        │
      │        ▼
      │   Found: nvidia_plugin/chat → OptimizedChatEndpoint
      │   Found: nvidia_plugin/embeddings → OptimizedEmbeddings
      │   Found: custom_plugin/rankings → CustomRankings
      │        │
      └────────┴─→ All plugins loaded
                    (same names override builtins)

      ▼
System ready!
- "chat" → OptimizedChatEndpoint (from plugin)
- "embeddings" → OptimizedEmbeddings (from plugin)
- "completions" → CompletionsEndpoint (builtin, no override)
- "rankings" → CustomRankings (from plugin)
```

---

## Registration Sources & Priority

```
┌─────────────────────────────────────────────────────────┐
│          Priority: Which gets used by default?          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. Runtime Override    ✓                              │
│     EndpointFactory.use_implementation(...)            │
│     (User manually switched)                           │
│                                                         │
│  2. Plugin Entry Point  (is_plugin=True)              │
│     [project.entry-points."aiperf.*.plugins"]          │
│     (Installed via pip)                                │
│                                                         │
│  3. Builtin Registry    (is_plugin=False)             │
│     aiperf/*/registry.py                               │
│     (Always available, fallback)                       │
│                                                         │
│  4. Environment Variable / Config Override             │
│     (Optional: for production use)                     │
│                                                         │
└─────────────────────────────────────────────────────────┘

Decision Tree:

    Is runtime override set?
        Yes ──→ Use override (user chose)
        No ──→ Is plugin available?
                  Yes ──→ Use plugin (is_plugin=True)
                  No ──→ Use builtin (is_plugin=False)
```

---

## File Organization

```
aiperf/
├── endpoints/
│   ├── __init__.py           ← Imports registry
│   ├── registry.py           ← ONE SOURCE OF TRUTH
│   │   ├── from .openai_chat import ChatEndpoint
│   │   ├── from .openai_embeddings import EmbeddingsEndpoint
│   │   ├── register("chat")(ChatEndpoint)
│   │   ├── register("embeddings")(EmbeddingsEndpoint)
│   │   └── ... (all registrations here)
│   │
│   ├── base_endpoint.py      ← Base class
│   ├── openai_chat.py        ← ChatEndpoint (clean)
│   ├── openai_embeddings.py  ← EmbeddingsEndpoint (clean)
│   └── nim_rankings.py       ← RankingsEndpoint (clean)
│
├── transports/               ← Same pattern
│   ├── __init__.py
│   ├── registry.py
│   ├── http_transport.py
│   ├── grpc_transport.py
│   └── ...
│
└── common/
    └── factories.py          ← Factory system
        ├── register(name)
        ├── create_instance(name)
        ├── discover_plugins(group)
        ├── use_implementation(name, class)
        └── list_implementations(name)
```

---

## User Installation Scenarios

### Scenario 1: AIPerf Only

```
$ pip install aiperf

Available endpoints:
  - chat (builtin)
  - embeddings (builtin)
  - completions (builtin)
  - rankings (builtin)

No plugins loaded.
```

### Scenario 2: AIPerf + NVIDIA Plugin

```
$ pip install aiperf nvidia-optimized-endpoints

aiperf/endpoints/registry.py registers:
  ✓ chat
  ✓ embeddings
  ✓ completions
  ✓ rankings

EndpointFactory.discover_plugins() finds:
  ✓ chat (OVERRIDE!)
  ✓ embeddings (OVERRIDE!)
  ✓ completions (OVERRIDE!)

Available endpoints:
  - chat (nvidia plugin) ← Selected
  - chat (builtin)
  - embeddings (nvidia plugin) ← Selected
  - embeddings (builtin)
  - completions (nvidia plugin) ← Selected
  - completions (builtin)
  - rankings (builtin) ← Selected (no plugin override)
```

### Scenario 3: Multiple Plugins

```
$ pip install aiperf nvidia-optimized-endpoints custom-endpoints

Builtins registered:
  chat, embeddings, completions, rankings

Plugins found:
  From nvidia: chat, embeddings, completions
  From custom: rankings

Result:
  - chat → nvidia plugin (first plugin wins) ← Selected
  - embeddings → nvidia plugin ← Selected
  - completions → nvidia plugin ← Selected
  - rankings → custom plugin ← Selected

All available for user override via use_implementation()
```

---

## The Beautiful Part: Seamless Override

```
Same Name = Seamless Override (Zero Friction)

Entry Point Group 1 (Builtin):
  [aiperf.endpoints]
  chat = aiperf.endpoints.openai_chat:ChatEndpoint

Entry Point Group 2 (Plugin):
  [aiperf.endpoints.plugins]
  chat = nvidia.endpoints:OptimizedChatEndpoint

User sees in code:
  EndpointFactory.create_instance("chat")

User doesn't care about implementation details.
System automatically uses highest priority.
Plugin seamlessly replaces builtin.
✓ Perfect!
```

---

## Summary: Three Layers

```
Layer 1: BUILTIN REGISTRATION
  ├─ Location: aiperf/*/registry.py
  ├─ When: import aiperf.*
  ├─ How: EndpointFactory.register()(Class)
  ├─ Dev Speed: ⚡ Instant (no reinstall)
  └─ What: All core implementations

Layer 2: PLUGIN DISCOVERY
  ├─ Location: my_plugin/pyproject.toml
  ├─ When: discover_plugins() at startup
  ├─ How: [project.entry-points."aiperf.*.plugins"]
  ├─ User Speed: ⚡ Instant (pip install)
  └─ What: Optional optimizations, alternatives

Layer 3: RUNTIME OVERRIDE
  ├─ Location: Application code
  ├─ When: Anytime (testing, debugging, config)
  ├─ How: use_implementation(name, class)
  ├─ User Control: ✓ Full
  └─ What: Manual selection for specific needs

All three together = Perfect system!
```
