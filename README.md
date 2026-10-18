# WanderWise ReAct Travel Agent with Configurable Personas
**Course:** EECE 503P/798S: Agentic Systems  
**Assignment C4:** Build a Custom ReAct Agent with Configurable Personas  
**Framework:** LangGraph (State Machine Workflows)

---

Builds on the WanderWise business assistant from Assignment C3: [WanderWise-C3-Business-Agent](https://github.com/Jadjnm/WanderWise-C3-Business-Agent)

## 1. Overview & Objectives

This project extends the **WanderWise Travel** assistant by implementing a **ReAct reasoning agent** (*Reasoning + Acting*) using **LangGraph**.

Rather than answering in a single pass, the agent moves through explicit thinking and tool-use steps:
`Thought -> Action -> Observation -> Thought -> Final Answer`

### Core Implementation
* **No Prebuilt Executors:** The ReAct cycle is implemented manually via a custom LangGraph `StateGraph` using states, nodes (`reason`, `act`), and conditional edges (`should_continue`).
* **Configurable Personas & Hyperparameter Alignment (with per-setting rationales):**
  1. **Aria (Friendly Concierge):** Warm, welcoming, encouraging, empathetic, and conversational (`temp 0.7`: warm variety without factual drift; `top_p 0.9` / `top_k 40`: broad hospitable vocabulary; `freq 0.1` / `pres 0.2`: discourages repeating pleasantries and introduces fresh travel ideas; `max_tokens 600`: space for warm greetings and breakdowns).
  2. **Marcus (Budget & Policy Auditor):** Analytical, precise, objective, concise, and uncompromising on guardrails (`temp 0.1`: deterministic and repeatable budget/policy answers; `top_p 0.5` / `top_k 10`: high-probability token selection eliminating speculation; `freq 0.4` / `pres 0.0`: prevents boilerplate repetition and stays strictly on verified numbers; `max_tokens 400`: concise audit brevity).
  3. **Elena (Adventurous Experience Curator):** Evocative, sensory, culturally immersed, and experiential (`temp 0.85`: rich evocative sensory language; `top_p 0.95` / `top_k 64`: vibrant vocabulary for cultural immersion; `freq 0.2` / `pres 0.5`: nudges narrative forward to new sights and details rather than repeating; `max_tokens 800`: room for descriptive vignettes).
* **Prompt Engineering Experiments & Instructions Ablation:** Toggles for Chain-of-Thought on/off, zero-shot vs. few-shot, system prompt vs. in-message instructions, `instructions_mode` (`full` vs `format_only` rules ablation), and `loop_guards` (repeat-action block + forced final answer at step limit), plus per-run overrides of temperature, top-p, top-k, max tokens, penalties, and model, all run and logged by `experiments.py`.
* **Persona/Instruction Conflict Analysis:** Automated `check_persona_conflicts()` scanner documenting tone instructions that pull against operational rules.
* **Dedicated Instructions Block:** Explicit input/output contracts with an embedded worked example.
* **Reflection Questions:** Complete answers to all reflection questions included in [`c4_react_agent.ipynb`](c4_react_agent.ipynb).

---

## 2. Project Files

| File | Purpose |
|---|---|
| `c4_react_agent.ipynb` | C4 notebook: manual LangGraph ReAct state machine, persona design, and experiment evaluation |
| `react_core.py` | Standalone ReAct state machine, tools, persona configurations, and guardrails for C4 |
| `c4_app.py` | Interactive Gradio application for C4 with real-time ReAct trace inspector |
| `experiments.py` | Experiment harness for C4: persona, prompting, parameter, model, guardrails, instructions, and loop_guards suites |
| `tests/test_react_agent.py` | Offline unit tests for C4 ReAct agent and tools (61 tests; no live API key required) |
| `README.md` | Documentation and experimental evaluation guide for C4 |
| `.env.example` | Template for environment variables and provider credentials |
| `requirements.txt` | Python package dependencies |

---

## 3. ReAct State Machine Architecture

```text
               User Query
                   |
            +------------+
            |   reason   | <----------------+
            +------------+                  |
                   |                        |
         [Should Continue?]                 |
          /              \                  |
    Action Requested    Final Answer        |
        /                  \                |
  +----------+             +-----+          |
  |   act    |             | END |          |
  +----------+             +-----+          |
       |                                    |
       +--- Observation Generated ----------+
```

### Optional code guardrails & loop guards
* `run_react_agent(..., guardrails=True)` adds a `guard` node after the Final Answer: it calls `record_unanswered_inquiry` when a policy lookup said the question must be logged but the model skipped it, and removes sentences that quote an airfare price. Guardrails are off by default (all logged experiments ran without them); they are enabled by default in `c4_app.py`. Measure the effect with `python experiments.py --suite guardrails`.
* `run_react_agent(..., loop_guards=True)` adds loop prevention: (1) **Repeat-action block in `act` node:** intercepts duplicate tool calls with the exact same name and arguments earlier in the turn and returns an error observation to break repeating loops; (2) **Forced final answer in `force_answer` node:** if the 5-step limit is reached without a final answer or LLM error, makes ONE more LLM call forcing a final answer from existing observations (`stopped_reason="forced_answer"`). Loop guards default to OFF in `run_react_agent` and ON in `c4_app.py`. Measure with `python experiments.py --suite loop_guards`.
* `run_react_agent(..., instructions_mode="format_only")` (default `"full"`): drops operational rules and worked examples to evaluate instruction adherence under format-only constraints. Measure with `python experiments.py --suite instructions`.
* `check_persona_conflicts()`: scans each persona's tone instructions for phrases that pull against operational rules.

### Available Tools
1. `search_travel_packages(query: str)`: Queries destination packages (Bali, Rome, Santorini, Swiss Alps, Japan).
2. `calculate_trip_budget(package_name, num_travelers, room_upgrade, private_guide)`: Computes base rates and optional add-ons.
3. `lookup_policy_rule(topic: str)`: Audits company policies on flights, visas, cancellations, and bookings.
4. `record_customer_lead(name, email, notes)`: Saves consultation leads to `logs/c4_leads.jsonl`.
5. `record_unanswered_inquiry(question: str)`: Logs unanswerable inquiries to `logs/c4_feedback.jsonl`.

---

## 4. Running the C4 ReAct Agent

### Option A: Launch Interactive Gradio Demo
```bash
python c4_app.py
```
Open your browser at `http://127.0.0.1:7860`. You can switch personas, toggle Chain-of-Thought, toggle code guardrails and loop guards, and watch the Thought, Action, and Observation trace update in real time.

### Option B: Run the Jupyter Notebook
Open [`c4_react_agent.ipynb`](c4_react_agent.ipynb) in Jupyter or Google Colab and run all cells sequentially.

### Option C: Run the Logged Experiments
```bash
python experiments.py --list                                      # show test cases and conditions
python experiments.py --suite personas                            # 3 personas x 6 cases
python experiments.py --suite prompting --persona strict_auditor
python experiments.py --suite params --persona friendly_advisor --sleep 6
python experiments.py --suite models --models model_a,model_b --sleep 5   # every persona on every model
python experiments.py --suite guardrails                                  # personas with guardrails off vs on
python experiments.py --suite instructions --sleep 3                      # instructions ablation (full vs format_only)
python experiments.py --suite loop_guards --repeats 3 --sleep 5           # strict_auditor loop guards off vs on
python experiments.py --suite prompting --repeats 3 --resume               # add repeats or resume after a stop
python experiments.py --rescore                                            # re-score everything, latest run per key
```
Use `--log <file>` to keep a parallel run (e.g. a second provider) in its own log.
Every run (config, full trace, raw model output, metrics) is appended to `logs/c4_experiments.jsonl` (or your `--log` file), and each invocation writes a comparison table to `logs/c4_experiments_summary_<run_id>.md`.

**Provider selection:** `LLM_PROVIDER` in `.env` is always honoured; if blank, the first real key found is used (Gemini, then OpenRouter, then OpenAI, then Bedrock). With no key, or `REACT_SIMULATE=1`, a clearly labelled offline simulator runs instead. A failed live call is reported as an error and is never replaced with simulated output.

**AWS Bedrock:** set `LLM_PROVIDER=bedrock`, plus either `AWS_BEARER_TOKEN_BEDROCK` (a Bedrock API key) or `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` or `AWS_PROFILE`, `AWS_REGION`, and `BEDROCK_MODEL_ID` (an inference-profile id such as `us.anthropic.claude-haiku-4-5-20251001-v1:0`, enabled under *Model access*). Bedrock's Converse API has no frequency/presence penalties, and newer Claude models refuse `top_p` together with `temperature`; parameters a model rejects are dropped and recorded per run under `ignored_params`. Summary tables group runs by model, so results from different providers in the same log are never mixed.

### Option D: Run Unit Tests
```bash
# Run C4 ReAct agent unit tests (61 tests)
python -m unittest tests/test_react_agent.py -v

# Or run every test in this repository
python -m unittest discover -s tests -v
```
