# gait
An open source version and source control system, inspired by Git, for Artificial Intelligence Agents

GAIT — Git-Like Version Control for AI Conversations

GAIT is a lightweight, Git-inspired version control system for AI conversations, prompts, and memory.

It lets you:

Record user ↔ assistant turns as immutable objects

Commit conversation state to branches

Pin important turns into long-lived “memory”

Rewind history and memory safely

Branch, merge, and experiment without losing context

Think of GAIT as Git for reasoning, not files.

Why GAIT Exists

AI workflows don’t behave like filesystems:

Conversations evolve

“Good” context must be remembered

“Bad” turns must be undone

Experiments should not pollute production memory

GAIT solves this by separating history from memory, while keeping both versioned, inspectable, and reversible.

Core Concepts
Turns

A turn is a single user + assistant interaction.

User:      "What is X?"
Assistant: "X is Y..."


Turns are immutable and stored as content-addressed objects.

Commits

A commit references one or more turns and has parents, metadata, and a branch.

Normal commits contain turns

Merge commits contain no turns

Commits form a DAG (just like Git)

Memory (HEAD+ Memory)

Memory is explicitly pinned context that survives across turns.

Memory is not automatic

Only pinned turns enter memory

Memory is versioned independently of commits

Memory has its own reflog

This prevents accidental context bloat and hallucination drift.

Repository Layout

GAIT stores everything in a .gait/ directory:

.gait/
├── HEAD
├── objects/          # content-addressed objects (turns, commits, memory)
├── refs/
│   ├── heads/        # branches
│   └── memory/       # branch memory refs
├── turns.jsonl       # turn → commit log
└── memory.jsonl      # memory reflog

Installation (Dev / Editable)
python -m venv GAITING
source GAITING/bin/activate
pip install -e .


This installs the gait CLI into your virtualenv.

Quick Start
gait init

Record a turn
gait record-turn \
  --user "What is GAIT?" \
  --assistant "GAIT is Git for AI conversations."


This:

Stores the turn

Creates a commit

Advances HEAD

Pin important context into memory
gait pin --last --note "baseline definition"


Pinned turns now become part of HEAD+ memory.

gait memory

View commit history
gait log

Show a commit’s conversation
gait show HEAD

Branching & Experiments
gait branch experiment
gait checkout experiment


Branches:

Inherit commit history

Optionally inherit memory (--no-inherit-memory supported)

Perfect for prompt experiments.

Merging
gait merge experiment


Optional memory merge:

gait merge experiment --with-memory


Memory merges:

Deduplicate pinned turns

Preserve provenance

Are logged in the memory reflog

Revert vs Reset (Important)
gait revert

A safe undo.

gait revert
gait revert --also-memory


Moves HEAD to the parent of the current commit

Optionally rewinds memory to the correct historical state

Best for:

“Oops, last turn was bad”

Interactive usage

gait reset

A power tool, like git reset.

gait reset <commit>
gait reset --hard


reset → move HEAD only

reset --hard → move HEAD and memory

Best for:

Timeline surgery

Returning to a known-good state

Cleaning experimental branches

Memory Reflog (The Secret Sauce)

Every memory mutation is logged with:

Timestamp

Branch

HEAD commit at the time

Old memory → new memory

Reason (pin, unpin, merge-memory, revert-memory, reset-memory)

This makes memory reversible, auditable, and safe.

Context Export (Agent-Ready)
gait context --json


Produces a structured context bundle:

{
  "schema": "gait.context.v0",
  "branch": "main",
  "memory_id": "...",
  "pinned_items": 2,
  "items": [...]
}


Designed for:

LLM prompts

Agent frameworks

MCP (future)

What GAIT Is Not (Yet)

❌ No file tracking

❌ No automatic memory

❌ No MCP server (coming later)

❌ No remote sync

GAIT is intentionally small, explicit, and correct.

Philosophy

History is cheap.
Memory is intentional.
Reversibility is non-negotiable.

GAIT treats AI context like production infrastructure — not chat logs.

Status

Version: 0.0.1
State: Actively developed, core model stable
Next milestones:

gait reset polish

Memory diff / inspect

MCP integration

Remote transport