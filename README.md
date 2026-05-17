# Codebase Auditor

An AI-powered security and architecture auditing tool built with a raw multi-agent pipeline. No LangChain. No abstractions. Every prompt, response, tool call, and token is traced and logged.

---

## What It Does

Point it at any repository and it runs three specialised AI agents in sequence:

1. **Planner** — reads the file structure and decides what to audit and in what order
2. **Executor** — reads each file using tool calls and identifies security vulnerabilities and architectural issues
3. **Critic** — reviews every finding, removes false positives, calibrates severity, and can send findings back to the executor for deeper investigation before making a final call

The output is a structured markdown report with all findings grouped by severity, a full list of false positives removed, file coverage, and a token usage breakdown per agent.

Every LLM call is traced — prompt sent, response received, tools called, tokens used, duration — and persisted to a local SQLite database for analysis.

---

## Why This Exists

Most AI pipelines are built on top of LangChain or similar frameworks. When things go wrong in production, you end up debugging the framework instead of your own logic. Context gets lost. Errors are hard to trace.

This project was built to understand how multi-agent systems work at the API level — raw tool use loops, explicit context management, custom tracing, and structured output handling — without any abstraction layer hiding what is actually happening.

---

## Architecture

```
main.py
  └── Orchestrator
        ├── DirectoryScanner     (pure Python, no LLM)
        ├── PlannerAgent         (no tools, reasons from file map)
        ├── ExecutorAgent        (uses read_file tool, one call per file)
        └── CriticAgent          (no tools, can trigger reinvestigation)
              └── ExecutorAgent  (reinvestigation calls)

Every agent call → TracerLogger → TracerStorage (SQLite)
```

### Provider Abstraction

Supports both OpenAI and Anthropic. Switch with one line in `.env`. No code changes needed.

```
providers/
  ├── base_provider.py       # shared interface
  ├── anthropic_provider.py  # Claude implementation
  ├── openai_provider.py     # GPT implementation
  └── factory.py             # picks provider from settings
```

---

## Project Structure

```
codebase-auditor/
│
├── .env                          # API keys and config (never committed)
├── .env.example                  # template for env vars
├── requirements.txt
├── README.md
├── main.py                       # entry point
│
├── config/
│   └── settings.py               # loads env vars, exposes constants
│
├── providers/
│   ├── base_provider.py
│   ├── anthropic_provider.py
│   ├── openai_provider.py
│   └── factory.py
│
├── tracer/
│   ├── models.py                 # Pydantic schemas for trace events
│   ├── logger.py                 # records every prompt, response, tool call, tokens
│   └── storage.py                # persists trace events to SQLite
│
├── agents/
│   ├── base.py                   # shared call logic, tool use loop, trace hooks
│   ├── planner.py
│   ├── executor.py
│   └── critic.py
│
├── tools/
│   ├── file_reader.py            # reads files from the target repo
│   ├── directory_scanner.py      # walks the directory tree
│   └── cve_checker.py            # checks dependencies against known CVEs
│
├── pipeline/
│   ├── context.py                # shared state passed between agents
│   └── orchestrator.py           # runs stages in sequence, handles errors
│
├── schemas/
│   ├── planner_output.py
│   ├── executor_output.py
│   └── critic_output.py
│
├── output/
│   ├── report.py                 # formats and saves the final markdown report
│   └── reports/                  # generated reports saved here (gitignored)
│
└── tests/
    ├── test_tracer.py
    ├── test_planner.py
    ├── test_executor.py
    └── test_critic.py
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/yourname/codebase-auditor
cd codebase-auditor
```

### 2. Create a virtual environment

Requires Python 3.11 or 3.12. Python 3.14 is not supported (pre-built wheels unavailable).

```bash
py -3.11 -m venv env

# Windows
env\Scripts\activate

# macOS / Linux
source env/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```bash
# choose your provider
PROVIDER=openai        # or: anthropic

# add the key for your chosen provider
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

# model must match your provider
MODEL=gpt-4o           # openai options: gpt-4o, gpt-4-turbo
                       # anthropic options: claude-sonnet-4-20250514

MAX_TOKENS=4096
DATABASE_URL=sqlite:///./traces.db
```

---

## Usage

```bash
# audit a repository
python main.py --repo /path/to/your/project

# quiet mode — no console output, just the report file
python main.py --repo /path/to/project --quiet
```

The report is saved to `output/reports/` as a markdown file.

---

## What Gets Traced

Every LLM call in the pipeline is recorded with:

- Agent name (planner, executor, critic)
- Full system prompt sent
- Full user message sent
- Full response received
- Tools called and their results
- Token counts (input, output, total)
- Duration in milliseconds
- Success or failure status

All trace events are linked by `session_id` and persisted to `traces.db`. You can query them directly:

## What It Finds

**Security**
- Hardcoded secrets, API keys, passwords, tokens
- SQL injection via string concatenation
- Missing input validation or sanitisation
- Insecure authentication and authorisation patterns
- Sensitive data in logs or error responses
- Vulnerable dependencies (via pip-audit and npm audit)
- Missing rate limiting on critical endpoints

**Architecture**
- God classes and functions doing too many things
- Business logic mixed into route handlers
- Missing or bare error handling
- N+1 database query patterns
- Missing indexes on frequently queried columns
- No retry logic for external API calls
- Synchronous blocking calls in async contexts
- Hardcoded configuration that should be environment variables

---

## Switching Providers

Change two lines in `.env`:

```bash
# switch to Anthropic
PROVIDER=anthropic
MODEL=claude-sonnet-4-20250514

# switch to OpenAI
PROVIDER=openai
MODEL=gpt-4o
```

No code changes required.

---

## Key Design Decisions

**No LangChain.** Every tool call loop, prompt construction, and response parsing is written explicitly. This makes the system easier to debug, easier to trace, and easier to extend.

**Pydantic for all schemas.** Every agent input and output is validated. If a model returns unexpected data it fails immediately with a clear error rather than propagating silently.

**Tracer built first.** The tracing layer was built before any agent logic. Every call is instrumented from day one rather than retrofitted later.

**Critic can trigger reinvestigation.** The critic holds a reference to the executor and can send specific questions back to it before making a final determination. This produces more accurate results than a single-pass review.

**Provider abstraction.** A thin abstraction layer normalises the differences between Anthropic and OpenAI APIs into a single interface. Adding a new provider means implementing three methods.

---

## Requirements

- Python 3.11 or 3.12
- OpenAI API key or Anthropic API key
- pip-audit (installed via requirements.txt)
- npm (optional, for JavaScript CVE checking)