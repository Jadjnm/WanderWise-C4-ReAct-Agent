"""WanderWise ReAct Travel Agent with Configurable Personas.

Course: EECE 503P/798S: Agentic Systems (Assignment C4)
Implements a manual ReAct loop using LangGraph (StateGraph) without prebuilt AgentExecutors.
Supports multiple configurable personas, prompt-engineering toggles (CoT, few-shot,
system vs in-message instructions), per-run LLM parameter overrides, multi-turn memory,
and three live providers (Gemini / OpenRouter / OpenAI) plus a clearly labelled offline simulator.
"""

import ast
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

# Locate project base directory
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
LOG_DIR = BASE_DIR / "logs"

# Provider imports with graceful fallbacks
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None  # type: ignore
    genai_types = None  # type: ignore

# ==============================================================================
# SECTION 1: WanderWise Business Tools
# ==============================================================================
CATALOG = {
    "bali bliss": {
        "name": "Bali Bliss",
        "duration": "7 days / 6 nights",
        "base_price_pp": 1850,
        "highlights": "Ubud rice terraces, traditional cultural experience, beach time in Seminyak",
        "inclusions": "6 nights midrange lodging, daily breakfast, arrival airport transfer, 2 guided experiences, ground transfers",
        "exclusions": "International flights, visas, personal spending, optional activities",
        "themes": "honeymoon, couples, beach, culture, wellness",
    },
    "rome & amalfi escape": {
        "name": "Rome & Amalfi Escape",
        "duration": "6 days / 5 nights",
        "base_price_pp": 1650,
        "highlights": "Rome historic center, coastal viewpoints, authentic Italian food tasting",
        "inclusions": "5 nights midrange lodging, daily breakfast, guided Rome walk, shared Amalfi Coast excursion, ground transfers",
        "exclusions": "International airfare, tourist taxes, dinners, travel insurance",
        "themes": "history, food, couples, coast, italy",
    },
    "santorini romance": {
        "name": "Santorini Romance",
        "duration": "5 days / 4 nights",
        "base_price_pp": 1950,
        "highlights": "Famous Oia sunset viewpoints, cliffside village walk, island leisure",
        "inclusions": "4 nights boutique lodging, daily breakfast, airport transfers, small-group village tour, sunset catamaran cruise",
        "exclusions": "Flights, ferry tickets not specified, alcoholic beverages outside cruise",
        "themes": "honeymoon, couples, romance, island, greece",
    },
    "swiss alpine discovery": {
        "name": "Swiss Alpine Discovery",
        "duration": "5 days / 4 nights",
        "base_price_pp": 2150,
        "highlights": "Lucerne historic center, scenic Alpine rail journeys, mountain panoramas",
        "inclusions": "4 nights midrange lodging, daily breakfast, specified rail passes, guided walking tour, local transit",
        "exclusions": "International flights, cable cars not in itinerary, lunches and dinners",
        "themes": "mountains, scenery, rail, nature, switzerland, alps",
    },
    "japan city & culture": {
        "name": "Japan City & Culture",
        "duration": "9 days / 8 nights",
        "base_price_pp": 2800,
        "highlights": "Tokyo modern neighborhoods, Kyoto heritage temples, Nishiki market food walk",
        "inclusions": "8 nights midrange lodging, daily breakfast, Shinkansen intercity bullet train, 2 guided cultural tours, airport arrival transfer",
        "exclusions": "International flights, visas, personal transit IC card top-ups. (Note: Independent package, NOT a group tour)",
        "themes": "culture, food, cities, temples, tokyo, kyoto",
    },
}

# Add-on rates exist only in this C4 demo tool (the C3 business summary lists room
# upgrades as unconfirmed), so every quote carries a fictional/indicative disclaimer.
ROOM_UPGRADE_PP = 250
PRIVATE_GUIDE_FLAT = 400

POLICIES = {
    "flights": "International airfare is strictly excluded from all packages. Travelers must book their own flights.",
    "visas": "WanderWise does not issue visas or guarantee immigration entry. Travelers must verify official embassy requirements.",
    "cancellation": "No standard cancellation or refund policy is published in current fictional materials. Inquiries must be logged as unanswered feedback.",
    "booking": "Aria and WanderWise concierge assistants take consultation inquiries only. No live seat reservations or credit card charges can be performed.",
    "insurance": "Travel and medical insurance is not included and cannot be purchased through this demo assistant.",
    "japan group tours": "There are no announced group tour departures for Japan next year. The Japan package is strictly independent travel.",
}

# Words a traveler might use for a policy topic -> canonical POLICIES key
POLICY_ALIASES = {
    "refund": "cancellation",
    "cancel": "cancellation",
    "airfare": "flights",
    "flight": "flights",
    "plane": "flights",
    "visa": "visas",
    "immigration": "visas",
    "reservation": "booking",
    "reserve": "booking",
    "payment": "booking",
    "group tour": "japan group tours",
}

_SEARCH_STOPWORDS = {
    "the", "and", "for", "with", "about", "package", "packages", "trip", "trips", "tour",
    "tours", "travel", "vacation", "holiday", "options", "option", "any", "what", "your",
}


def search_travel_packages(query: str) -> dict[str, Any]:
    """Search package catalog by destination name or keywords (e.g. 'bali', 'japan', 'honeymoon')."""
    words = [w for w in re.findall(r"[a-z]+", str(query).lower()) if len(w) > 2 and w not in _SEARCH_STOPWORDS]
    scored = []
    for key, pkg in CATALOG.items():
        haystack = (key + " " + " ".join(str(v) for v in pkg.values())).lower()
        score = sum(1 for w in words if w in haystack)
        if score:
            scored.append((score, pkg))
    if not scored:
        return {
            "status": "not_found",
            "message": f"No packages matching '{query}'. Available destinations: Bali, Rome & Amalfi, Santorini, Swiss Alps, Japan.",
        }
    scored.sort(key=lambda item: item[0], reverse=True)
    best = scored[0][0]
    matches = [pkg for score, pkg in scored if score == best]
    return {"status": "success", "count": len(matches), "packages": matches}


def _to_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1", "y")
    return bool(value)


def _match_package(package_name: str) -> dict[str, Any] | None:
    key = str(package_name).lower().strip()
    if not key:
        return None
    for k, v in CATALOG.items():
        if key in k or k in key:
            return v
    # Fall back to any significant word overlap ("Swiss Alps" -> Swiss Alpine Discovery)
    words = [w for w in re.findall(r"[a-z]+", key) if len(w) > 3]
    for k, v in CATALOG.items():
        if any(w in k for w in words):
            return v
    return None


def calculate_trip_budget(package_name: str, num_travelers: int, room_upgrade: bool = False, private_guide: bool = False) -> dict[str, Any]:
    """Calculate indicative total budget and breakdown for a package based on travelers and optional add-ons."""
    matched_pkg = _match_package(package_name)
    if not matched_pkg:
        return {"status": "error", "message": f"Package '{package_name}' not found in catalog."}

    try:
        num = max(1, int(num_travelers))
    except (TypeError, ValueError):
        return {"status": "error", "message": f"num_travelers must be a whole number, got {num_travelers!r}."}
    base_pp = matched_pkg["base_price_pp"]
    total = base_pp * num
    addons = []

    if _to_bool(room_upgrade):
        upgrade_cost = ROOM_UPGRADE_PP * num
        total += upgrade_cost
        addons.append(f"Deluxe room upgrade (+${upgrade_cost})")

    if _to_bool(private_guide):
        total += PRIVATE_GUIDE_FLAT
        addons.append(f"Private VIP guide add-on (+${PRIVATE_GUIDE_FLAT})")

    return {
        "status": "success",
        "package": matched_pkg["name"],
        "travelers": num,
        "base_rate_per_person": f"${base_pp:,}",
        "estimated_total_usd": f"${total:,}",
        "addons": addons or ["None (Standard package)"],
        "essential_disclaimer": "Indicative fictional quote based on double occupancy. Excludes international airfare and personal expenses.",
    }


def lookup_policy_rule(topic: str) -> dict[str, str]:
    """Check approved company policies regarding flights, visas, cancellation, refunds, insurance, or booking."""
    t = str(topic).lower().strip()
    for key, text in POLICIES.items():
        if key in t:
            return {"status": "found", "topic": key, "policy": text}
    for alias, key in POLICY_ALIASES.items():
        if alias in t:
            return {"status": "found", "topic": key, "policy": POLICIES[key]}
    return {
        "status": "unknown_policy",
        "topic": topic,
        "policy": "No approved policy found in business reference. Must be logged via record_unanswered_inquiry.",
    }


def record_customer_lead(name: str, email: str, notes: str = "") -> dict[str, str]:
    """Log customer contact details for a travel consultation inquiry."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not str(name).strip() or not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", str(email).strip()):
        return {"status": "error", "message": "Valid name and email address required."}
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "name": str(name).strip(),
        "email": str(email).strip(),
        "notes": str(notes).strip(),
    }
    with (LOG_DIR / "c4_leads.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"status": "success", "message": f"Consultation inquiry logged for {name}. Follow-up scheduled with trip designer."}


def record_unanswered_inquiry(question: str) -> dict[str, str]:
    """Log an unanswered business question or customer feedback."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "question": str(question).strip(),
    }
    with (LOG_DIR / "c4_feedback.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"status": "success", "message": "Query logged for the guest-experience team; no answer fabricated."}


TOOLS_BY_NAME: dict[str, Callable[..., Any]] = {
    "search_travel_packages": search_travel_packages,
    "calculate_trip_budget": calculate_trip_budget,
    "lookup_policy_rule": lookup_policy_rule,
    "record_customer_lead": record_customer_lead,
    "record_unanswered_inquiry": record_unanswered_inquiry,
}

TOOL_DESCRIPTIONS = """
Available Tools:
1. search_travel_packages(query: str) -> dict
   Searches vacation packages by destination or interest (e.g. "japan", "honeymoon").
2. calculate_trip_budget(package_name: str, num_travelers: int, room_upgrade: bool = False, private_guide: bool = False) -> dict
   Calculates total indicative trip budget and per-person breakdown.
3. lookup_policy_rule(topic: str) -> dict
   Looks up policies on flights, visas, cancellation/refunds, booking, insurance, Japan group tours.
4. record_customer_lead(name: str, email: str, notes: str) -> dict
   Saves contact details for custom travel planning inquiries.
5. record_unanswered_inquiry(question: str) -> dict
   Logs questions not answerable from approved reference materials.
""".strip()

# ==============================================================================
# SECTION 2: Configurable Personas & Parameter Profiles
# ==============================================================================
# Each persona couples prompt traits with a matching sampling configuration:
#   temperature/top_p/top_k -> how adventurous the wording is
#   frequency/presence penalty -> repetition vs. topic exploration
#   max_tokens -> how long the persona is allowed to talk per step
PERSONA_PROFILES: dict[str, dict[str, Any]] = {
    "friendly_advisor": {
        "name": "Aria (Friendly Concierge)",
        "tone": "Warm, welcoming, encouraging, empathetic, and conversational.",
        "task": "First contact, general questions and leads.",
        "llm_config": {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.2,
            "max_tokens": 600,
        },
        "config_rationale": {
            "temperature": "0.7 provides warm, conversational naturalness and varied phrasing without drifting into factual instability.",
            "top_p": "0.9 samples a broad, hospitable vocabulary while pruning low-probability nonsensical tokens.",
            "top_k": "40 keeps tone conversational and adaptable across diverse traveler queries without excessive divergence.",
            "frequency_penalty": "0.1 gently discourages repeated pleasantries while preserving an inviting, friendly cadence.",
            "presence_penalty": "0.2 encourages introducing new travel ideas and helpful suggestions during vacation exploration.",
            "max_tokens": "600 provides ample room for a warm greeting, thorough journey breakdown, and hospitable next steps.",
        },
        "description": "Ideal for first-time travelers, honeymooners, and general vacation exploration. Emphasizes excitement and personal care.",
        "prompt_tone_instructions": (
            "You are Aria, the friendly concierge of WanderWise Travel. Greet travelers warmly. "
            "Use encouraging language, share thoughtful travel inspiration, and help visitors discover their dream journey."
        ),
        "prompt_tone_instructions_v2": (
            "You are Aria, the friendly concierge of WanderWise Travel. Greet travelers warmly, "
            "and inspire travelers using only details from WanderWise's catalog and tool results; "
            "for costs or services outside the catalog (such as flights), say WanderWise doesn't quote them."
        ),
    },
    "strict_auditor": {
        "name": "Marcus (Budget & Policy Auditor)",
        "tone": "Analytical, precise, objective, concise, and uncompromising on guardrails.",
        "task": "Prices, budgets and policy checks.",
        "llm_config": {
            "temperature": 0.1,
            "top_p": 0.5,
            "top_k": 10,
            "frequency_penalty": 0.4,
            "presence_penalty": 0.0,
            "max_tokens": 400,
        },
        "config_rationale": {
            "temperature": "0.1 keeps budget calculations and policy statements deterministic, repeatable, and strictly factual.",
            "top_p": "0.5 restricts token selection to high-probability words for maximum analytical precision.",
            "top_k": "10 cuts off vocabulary tails entirely to eliminate speculative phrasing or ambiguous interpretations.",
            "frequency_penalty": "0.4 prevents looping over boilerplate policy language and forces concise, direct auditing.",
            "presence_penalty": "0.0 keeps Marcus focused strictly on verified numbers and explicit exclusions without introducing unneeded topics.",
            "max_tokens": "400 enforces brevity and concise audit summaries, preventing rambling or speculative prose.",
        },
        "description": "Ideal for budget scrutiny, inclusion/exclusion audits, and strict policy enforcement. Never speculates.",
        "prompt_tone_instructions": (
            "You are Marcus, the senior travel auditor at WanderWise. You communicate with absolute factual precision, "
            "highlight exact numbers, clearly itemize exclusions (no flights, no visas), and refuse to estimate undefined policies."
        ),
        "prompt_tone_instructions_v2": (
            "You are Marcus, the senior travel auditor at WanderWise. You communicate with absolute factual precision, "
            "highlight exact numbers, clearly itemize exclusions (no flights, no visas), and refuse to estimate undefined policies."
        ),
    },
    "creative_curator": {
        "name": "Elena (Adventurous Experience Curator)",
        "tone": "Evocative, sensory, culturally immersed, and experiential.",
        "task": "Destination experiences and highlights.",
        "llm_config": {
            "temperature": 0.85,
            "top_p": 0.95,
            "top_k": 64,
            "frequency_penalty": 0.2,
            "presence_penalty": 0.5,
            "max_tokens": 800,
        },
        "config_rationale": {
            "temperature": "0.85 encourages rich, evocative sensory language and imaginative destination storytelling.",
            "top_p": "0.95 unlocks an extensive, vibrant vocabulary for culturally immersed descriptions.",
            "top_k": "64 allows diverse descriptive phrasing while staying anchored within package boundaries.",
            "frequency_penalty": "0.2 minimizes repeated adjectives in colorful narrative prose.",
            "presence_penalty": "0.5 continually nudges the narrative forward to new sights, sounds, and experiences rather than dwelling on one detail.",
            "max_tokens": "800 provides generous room for detailed sensory vignettes and vivid journey narratives.",
        },
        "description": "Ideal for culture enthusiasts and off-the-beaten-path travelers seeking unique moments within package boundaries.",
        "prompt_tone_instructions": (
            "You are Elena, the Lead Experience Curator at WanderWise. Paint vivid pictures of destinations (sounds, flavors, vistas) "
            "while strictly respecting verified package inclusions and pricing."
        ),
        "prompt_tone_instructions_v2": (
            "You are Elena, the Lead Experience Curator at WanderWise. Paint vivid pictures of what the packages include; "
            "describe only what the packages include, and never estimate prices, airfares or anything the tools didn't return."
        ),
    },
}

LLM_PARAM_KEYS = ("temperature", "top_p", "top_k", "frequency_penalty", "presence_penalty", "max_tokens", "model")

PERSONA_CONFLICT_RULES = [
    (
        "share thoughtful travel inspiration",
        "Rule 5 (Never invent flight prices) / Rule 4 (Answer prices only from Observations)",
        "Encourages open-ended ideation that can tempt the model to invent flight estimates or unverified logistics when asked about airfare.",
    ),
    (
        "encouraging language",
        "Rule 4 (Answer prices/policies only from Observations) / Rule 6 (Record unanswered inquiry)",
        "Desire to be accommodating and encouraging can pressure the agent to offer speculative estimates rather than firmly stating policy exclusions or unapproved terms.",
    ),
    (
        "help visitors discover their dream journey",
        "Rule 5 (Never invent flight prices, visa rules, or cancellation policies)",
        "Broad helpfulness prompts the model to elaborate beyond verified catalog bounds.",
    ),
    (
        "Paint vivid pictures of destinations (sounds, flavors, vistas)",
        "Rule 5 (Never invent flight prices) / Rule 4 (Answer prices only from Observations)",
        "Sensory, imaginative scene-painting pulls the model toward descriptive embellishment and narrative fabrication beyond catalog facts.",
    ),
]


def check_persona_conflicts(version: str = "v1") -> list[tuple[str, str, str, str]]:
    """Scan each persona's prompt_tone_instructions for phrasing that pulls against operational rules.

    Returns a list of tuples: (persona_name, matched_phrase, rule_conflicted, evidence_note).
    Transparent, documented phrase scan rather than an LLM call.
    """
    key = "prompt_tone_instructions_v2" if version == "v2" else "prompt_tone_instructions"
    conflicts: list[tuple[str, str, str, str]] = []
    for pkey, profile in PERSONA_PROFILES.items():
        instructions = profile.get(key, profile.get("prompt_tone_instructions", ""))
        for phrase, rule, evidence in PERSONA_CONFLICT_RULES:
            if phrase.lower() in instructions.lower():
                conflicts.append((profile["name"], phrase, rule, evidence))
    return conflicts


_STRICT_AUDITOR_KEYWORDS = ("budget", "cost", "price", "refund", "policy", "visa", "flight")
_CREATIVE_CURATOR_KEYWORDS = ("highlights", "highlight", "experience", "culture", "food", "what is it like")


def route_persona_with_reason(query: str) -> tuple[str, str]:
    """Route query and return (persona_key, reason_string)."""
    q = str(query or "").lower()
    for kw in _STRICT_AUDITOR_KEYWORDS:
        if kw in q:
            return "strict_auditor", f"matched '{kw}' keyword -> strict_auditor (prices, budgets, policies)"
    for kw in _CREATIVE_CURATOR_KEYWORDS:
        if kw in q:
            return "creative_curator", f"matched '{kw}' keyword -> creative_curator (experiences, highlights)"
    return "friendly_advisor", "default fallback -> friendly_advisor (first contact, general questions, leads)"


def route_persona(query: str) -> str:
    """Route a customer query to the best-suited persona based on documented keyword rules.

    Rules:
    - budget / cost / price / refund / policy / visa / flight -> strict_auditor (Marcus: prices, budgets, policies)
    - highlights / experience / culture / food / 'what is it like' -> creative_curator (Elena: experiences, highlights)
    - Fallback -> friendly_advisor (Aria: first contact, general questions, leads)
    """
    key, _ = route_persona_with_reason(query)
    return key


# ==============================================================================
# SECTION 3: Dedicated Instructions with Worked Example
# ==============================================================================
INSTRUCTION_FORMAT = """
INPUT YOU RECEIVE:
- The recent conversation with the traveler (may be empty).
- The traveler's current message.
- The interaction trace so far: your earlier Thought/Action lines and the Observation returned by each tool.

OUTPUT YOU MUST RETURN (exactly one of these per reply):
- A next step:   "Thought: <one sentence>" on one line, then "Action: tool_name(param="value", ...)" on the next line.
- Or the finish: "Thought: <one sentence>" on one line, then "Final Answer: <reply to the traveler>".
Never write an "Observation:" line yourself; the system adds it after running your tool.
""".strip()

INSTRUCTION_OPERATIONAL_RULES = """
OPERATIONAL RULES:
1. Use the cycle Thought -> Action -> Observation, repeated as needed, then Final Answer.
2. In each Thought, state which fact you still need.
3. Call exactly one tool per Action, using keyword arguments.
4. Answer prices, inclusions and policies only from tool Observations.
5. Never invent flight prices, visa rules, or cancellation policies.
6. When a policy lookup returns no approved answer, call record_unanswered_inquiry with the traveler's question.
7. When a traveler gives a name and email and wants to book or consult, call record_customer_lead.
8. Write the Final Answer in your persona's voice.
""".strip()

INSTRUCTION_RULES = f"{INSTRUCTION_FORMAT}\n\n{INSTRUCTION_OPERATIONAL_RULES}"

WORKED_EXAMPLE = """
WORKED EXAMPLE:
User Question: "How much would a 7-day trip to Bali cost for 2 people?"
Thought: I need to check the Bali package pricing and calculate the total cost for 2 travelers.
Action: calculate_trip_budget(package_name="Bali Bliss", num_travelers=2)
Observation: {"status": "success", "package": "Bali Bliss", "travelers": 2, "base_rate_per_person": "$1,850", "estimated_total_usd": "$3,700"}
Thought: I have the verified calculation ($1,850 per person, $3,700 total for 2). Now I can provide the final answer clearly explaining inclusions and exclusions.
Final Answer: A 7-day Bali Bliss journey for two travelers is estimated at $3,700 total ($1,850 per person). This includes 6 nights midrange lodging, daily breakfast, and 2 guided tours, but excludes international airfare.
""".strip()

# Second example, used only in the few-shot condition, demonstrates the "unknown policy" path.
EXTRA_FEW_SHOT_EXAMPLE = """
SECOND EXAMPLE:
User Question: "Can I get my money back if I cancel two weeks before departure?"
Thought: I need the approved cancellation policy before answering.
Action: lookup_policy_rule(topic="cancellation")
Observation: {"status": "found", "topic": "cancellation", "policy": "No standard cancellation or refund policy is published in current fictional materials. Inquiries must be logged as unanswered feedback."}
Thought: No approved refund terms exist, so I must log the question instead of guessing.
Action: record_unanswered_inquiry(question="Can I get my money back if I cancel two weeks before departure?")
Observation: {"status": "success", "message": "Query logged for the guest-experience team; no answer fabricated."}
Thought: The question is logged. I can now explain this honestly.
Final Answer: We don't have published cancellation or refund terms yet, so I won't guess. I've passed your question to our guest-experience team so they can follow up with the exact terms.
""".strip()

# Backward-compatible combined block (rules + the required worked example)
DEDICATED_INSTRUCTIONS = f"{INSTRUCTION_RULES}\n\n{WORKED_EXAMPLE}"

INSTRUCTION_PLACEMENTS = ("system", "user")
INSTRUCTION_MODES = ("full", "format_only")
MAX_ITERATIONS = 5
HISTORY_TURNS = 6  # most recent chat messages fed back as short-term memory
STOP_SEQUENCES = ["\nObservation:"]
FORCE_ANSWER_PROMPT = (
    "You have reached the step limit. Using only the observations above, give your "
    "Final Answer now. If information is missing, say so instead of guessing."
)

# ==============================================================================
# SECTION 4: Provider Selection & LLM Calls
# ==============================================================================
DEFAULT_MODELS = {
    "gemini": "gemini-3.8-flash",
    "openrouter": "dots-studio/dots-3-note-preview:free",
    "openai": "gpt-4o-mini",
    "bedrock": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "simulated": "offline-simulator",
}
_KEY_ENV = {"gemini": "GEMINI_API_KEY", "openrouter": "OPENROUTER_API_KEY", "openai": "OPENAI_API_KEY"}
_MODEL_ENV = {"gemini": "GEMINI_MODEL", "openrouter": "OPENROUTER_MODEL", "openai": "OPENAI_MODEL",
              "bedrock": "BEDROCK_MODEL_ID"}
PROVIDERS = ("gemini", "openrouter", "openai", "bedrock")
_GEMINI_NO_PENALTY_MODELS: set[str] = set()  # learned at runtime, so the rejected call isn't repeated
RATE_LIMIT_RETRIES = 3


class LLMError(RuntimeError):
    """Raised when a live provider call fails; never silently replaced by simulated output."""


def _env_key(name: str) -> str:
    """Return an env var, treating the .env.example placeholders as unset."""
    value = os.getenv(name, "").strip()
    return "" if not value or value.lower().startswith("your_") else value


def get_active_provider() -> str:
    """Pick the provider: REACT_SIMULATE=1 > explicit LLM_PROVIDER > first key found > simulated."""
    if os.getenv("REACT_SIMULATE", "").strip() in ("1", "true", "yes"):
        return "simulated"
    explicit = os.getenv("LLM_PROVIDER", "").strip().lower()
    if explicit in (*PROVIDERS, "simulated"):
        return explicit
    for provider in ("gemini", "openrouter", "openai"):
        if _env_key(_KEY_ENV[provider]):
            return provider
    if _has_aws_credentials():
        return "bedrock"
    return "simulated"


def _has_aws_credentials() -> bool:
    """Bedrock API key (bearer token) or classic access keys / named profile."""
    return bool(
        _env_key("AWS_BEARER_TOKEN_BEDROCK")
        or (_env_key("AWS_ACCESS_KEY_ID") and _env_key("AWS_SECRET_ACCESS_KEY"))
        or _env_key("AWS_PROFILE")
    )


def get_active_model(provider: str | None = None) -> str:
    p = provider or get_active_provider()
    if p == "simulated":
        return DEFAULT_MODELS["simulated"]
    return os.getenv(_MODEL_ENV[p], "").strip() or DEFAULT_MODELS[p]


def resolve_llm_config(persona_key: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Persona defaults merged with per-run overrides (used by experiments)."""
    profile = PERSONA_PROFILES.get(persona_key, PERSONA_PROFILES["friendly_advisor"])
    cfg = dict(profile["llm_config"])
    for k, v in (overrides or {}).items():
        if k not in LLM_PARAM_KEYS:
            raise ValueError(f"Unknown LLM parameter '{k}'. Allowed: {', '.join(LLM_PARAM_KEYS)}")
        if v is not None:
            cfg[k] = v
    return cfg


def _call_gemini(system: str | None, user: str, cfg: dict[str, Any], model: str) -> tuple[str, dict[str, Any]]:
    if genai is None:
        raise LLMError("google-genai is not installed: pip install google-genai")
    api_key = _env_key("GEMINI_API_KEY")
    if not api_key:
        raise LLMError("LLM_PROVIDER is gemini but GEMINI_API_KEY is not set in .env")
    params: dict[str, Any] = {
        "temperature": cfg["temperature"],
        "top_p": cfg["top_p"],
        "top_k": cfg["top_k"],
        "max_output_tokens": cfg["max_tokens"],
        "presence_penalty": cfg["presence_penalty"],
        "frequency_penalty": cfg["frequency_penalty"],
        "stop_sequences": STOP_SEQUENCES,
    }
    extra: dict[str, Any] = {}
    if system:
        extra["system_instruction"] = system
    if "flash" in model:
        # Disable hidden "thinking" so the visible Thought lines are the only reasoning and
        # max_tokens is spent on the reply, not on invisible reasoning tokens (otherwise a
        # 120-token budget can be used up entirely by thinking and the reply comes back cut off).
        extra["thinking_config"] = genai_types.ThinkingConfig(thinking_budget=0)

    ignored: list[str] = []
    if model in _GEMINI_NO_PENALTY_MODELS:
        params.pop("presence_penalty")
        params.pop("frequency_penalty")
        ignored = ["presence_penalty", "frequency_penalty"]

    client = genai.Client(api_key=api_key)
    rate_limit_retries = 0
    while True:
        try:
            resp = client.models.generate_content(
                model=model, contents=user, config=genai_types.GenerateContentConfig(**params, **extra)
            )
            break
        except Exception as err:
            msg = str(err)
            if "penalty" in msg.lower() and "presence_penalty" in params:
                # This model rejects penalties: remember that, drop them, and record it per run.
                _GEMINI_NO_PENALTY_MODELS.add(model)
                params.pop("presence_penalty")
                params.pop("frequency_penalty")
                ignored = ["presence_penalty", "frequency_penalty"]
                continue
            if ("429" in msg or "RESOURCE_EXHAUSTED" in msg) and "PerDay" in msg:
                # Daily quota: waiting a minute won't help, so fail fast with a readable message.
                limit = re.search(r"quotaValue['\"]?\s*:\s*['\"]?(\d+)", msg)
                delay = re.search(r"retryDelay['\"]?\s*:\s*['\"]?(\d+)", msg)
                raise LLMError(
                    f"Gemini daily quota exhausted for {model}"
                    + (f" ({limit.group(1)} requests/day)" if limit else "")
                    + (f"; resets in ~{int(delay.group(1)) / 3600:.1f} h" if delay else "")
                    + ". Wait, switch GEMINI_MODEL, or enable billing."
                ) from err
            if ("429" in msg or "RESOURCE_EXHAUSTED" in msg) and rate_limit_retries < RATE_LIMIT_RETRIES:
                rate_limit_retries += 1
                delay = re.search(r"retry(?:Delay)?['\"]?\s*[:=]?\s*['\"]?(\d+(?:\.\d+)?)s", msg, flags=re.I)
                wait = min(float(delay.group(1)) + 1 if delay else 30.0, 90.0)
                print(f"[REACT NOTE] Gemini rate limit hit; waiting {wait:.0f}s (retry {rate_limit_retries}/{RATE_LIMIT_RETRIES})")
                time.sleep(wait)
                continue
            raise LLMError(f"Gemini call failed: {err}") from err
    applied = {k: v for k, v in cfg.items() if k not in ignored and k != "model"}
    return resp.text or "", {"applied_config": applied, "ignored_params": ignored}


def _call_openai_compatible(provider: str, system: str | None, user: str, cfg: dict[str, Any], model: str) -> tuple[str, dict[str, Any]]:
    if OpenAI is None:
        raise LLMError("openai SDK is not installed: pip install openai")
    api_key = _env_key(_KEY_ENV[provider])
    if not api_key:
        raise LLMError(f"LLM_PROVIDER is {provider} but {_KEY_ENV[provider]} is not set in .env")

    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": cfg["temperature"],
        "top_p": cfg["top_p"],
        "frequency_penalty": cfg["frequency_penalty"],
        "presence_penalty": cfg["presence_penalty"],
        "stop": STOP_SEQUENCES,
    }
    ignored: list[str] = []
    if provider == "openrouter":
        # max_retries=0: rate limits are handled below so daily limits fail fast instead of looping
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            max_retries=0,
            default_headers={"HTTP-Referer": "https://wanderwise.local", "X-Title": "WanderWise ReAct"},
        )
        kwargs["max_tokens"] = cfg["max_tokens"]
        kwargs["extra_body"] = {"top_k": cfg["top_k"]}
        # OpenRouter silently drops parameters a model doesn't support; record which ones.
        supported = _openrouter_supported_params(model)
        if supported is not None:
            ignored = [p for p in ("temperature", "top_p", "top_k", "frequency_penalty", "presence_penalty", "max_tokens")
                       if p not in supported]
            if "reasoning" in supported:
                # Same reason as Gemini's thinking_budget=0: keep the visible Thought lines as the only
                # reasoning, and stop hidden reasoning from leaking into the reply or eating max_tokens.
                kwargs["extra_body"]["reasoning"] = {"enabled": False}
    else:
        client = OpenAI(api_key=api_key, max_retries=0)
        kwargs["max_completion_tokens"] = cfg["max_tokens"]
        ignored.append("top_k")  # OpenAI Chat Completions has no top_k parameter

    def retryable(msg: str) -> bool:
        low = msg.lower()
        return any(s in low for s in ("429", "rate limit", "rate-limited", "502", "503", "504", "temporarily"))

    rate_limit_retries = 0
    while True:
        try:
            res = client.chat.completions.create(**kwargs)
        except Exception as err:
            msg = str(err)
            if "reasoning" in msg.lower() and "reasoning" in kwargs.get("extra_body", {}):
                kwargs["extra_body"].pop("reasoning")  # model insists on reasoning; let it
                continue
            if "per-day" in msg.lower() or "insufficient_quota" in msg:
                raise LLMError(f"{provider} daily quota exhausted for {model}: {msg[:200]}") from err
            if not (retryable(msg) and rate_limit_retries < RATE_LIMIT_RETRIES):
                raise LLMError(f"{provider} call failed: {err}") from err
        else:
            if res.choices:
                break
            # OpenRouter can return HTTP 200 with an error body (e.g. an upstream 429) instead of choices
            msg = str(getattr(res, "error", None) or res)
            if "per-day" in msg.lower():
                raise LLMError(f"{provider} daily quota exhausted for {model}: {msg[:200]}")
            if not (retryable(msg) and rate_limit_retries < RATE_LIMIT_RETRIES):
                raise LLMError(f"{provider} returned no choices: {msg[:300]}")
        rate_limit_retries += 1
        wait = 20.0 * rate_limit_retries
        print(f"[REACT NOTE] {provider} rate limit or upstream error; waiting {wait:.0f}s "
              f"(retry {rate_limit_retries}/{RATE_LIMIT_RETRIES})")
        time.sleep(wait)
    applied = {k: v for k, v in cfg.items() if k not in ignored and k != "model"}
    return res.choices[0].message.content or "", {"applied_config": applied, "ignored_params": ignored}


_OPENROUTER_PARAMS_CACHE: dict[str, list[str]] | None = None


def _openrouter_supported_params(model: str) -> list[str] | None:
    """Look up a model's supported_parameters from OpenRouter's public model list (cached)."""
    global _OPENROUTER_PARAMS_CACHE
    if _OPENROUTER_PARAMS_CACHE is None:
        try:
            import urllib.request
            with urllib.request.urlopen("https://openrouter.ai/api/v1/models", timeout=20) as resp:
                data = json.load(resp)["data"]
            _OPENROUTER_PARAMS_CACHE = {m["id"]: m.get("supported_parameters") or [] for m in data}
        except Exception:
            _OPENROUTER_PARAMS_CACHE = {}
    return _OPENROUTER_PARAMS_CACHE.get(model)


_BEDROCK_CLIENTS: dict[str, Any] = {}
BEDROCK_MAX_TEMPERATURE = 1.0
# Per model, parameters Bedrock rejected once (e.g. newer Claude models refuse temperature + top_p together)
_BEDROCK_REJECTED: dict[str, set[str]] = {}


def _bedrock_client(region: str) -> Any:
    if region not in _BEDROCK_CLIENTS:
        try:
            import boto3
            from botocore.config import Config
        except ImportError as err:
            raise LLMError("boto3 is not installed: pip install boto3") from err
        # boto3 reads AWS_BEARER_TOKEN_BEDROCK or the AWS_ACCESS_KEY_ID/SECRET pair from the environment
        # (load_dotenv has already copied them from .env). Adaptive retries absorb short throttling bursts.
        _BEDROCK_CLIENTS[region] = boto3.client(
            "bedrock-runtime",
            region_name=region,
            config=Config(retries={"max_attempts": 6, "mode": "adaptive"}, read_timeout=120),
        )
    return _BEDROCK_CLIENTS[region]


def _bedrock_top_k_field(model: str, top_k: int) -> dict[str, Any] | None:
    """top_k is not part of Converse's common inferenceConfig; each model family names it differently."""
    if "anthropic." in model:
        return {"top_k": top_k}
    if "amazon.nova" in model:
        return {"inferenceConfig": {"topK": top_k}}
    if "mistral." in model or "cohere." in model:
        return {"top_k": top_k}
    return None


def _call_bedrock(system: str | None, user: str, cfg: dict[str, Any], model: str) -> tuple[str, dict[str, Any]]:
    if not _has_aws_credentials():
        raise LLMError("LLM_PROVIDER is bedrock but no AWS_BEARER_TOKEN_BEDROCK or AWS access keys are set in .env")
    region = os.getenv("AWS_REGION", "").strip() or "us-east-1"
    client = _bedrock_client(region)

    # Converse has no frequency/presence penalty parameters for any model family.
    ignored = ["frequency_penalty", "presence_penalty"]
    rejected = _BEDROCK_REJECTED.setdefault(model, set())

    # Bedrock caps temperature at 1.0 for Claude, Nova, Llama and Mistral; clamp and record it
    # rather than failing the run.
    temperature = float(cfg["temperature"])
    adjusted: dict[str, str] = {}
    if temperature > BEDROCK_MAX_TEMPERATURE:
        adjusted["temperature"] = f"{temperature} -> {BEDROCK_MAX_TEMPERATURE} (Bedrock maximum)"
        temperature = BEDROCK_MAX_TEMPERATURE

    while True:
        inference: dict[str, Any] = {"maxTokens": int(cfg["max_tokens"]), "temperature": temperature}
        if "top_p" not in rejected:
            inference["topP"] = float(cfg["top_p"])
        if "stop" not in rejected:
            inference["stopSequences"] = STOP_SEQUENCES
        request: dict[str, Any] = {
            "modelId": model,
            "messages": [{"role": "user", "content": [{"text": user}]}],
            "inferenceConfig": inference,
        }
        if system:
            request["system"] = [{"text": system}]
        top_k_field = _bedrock_top_k_field(model, int(cfg["top_k"])) if "top_k" not in rejected else None
        if top_k_field:
            request["additionalModelRequestFields"] = top_k_field

        try:
            resp = client.converse(**request)
            break
        except Exception as err:
            msg = str(err)
            low = msg.lower()
            if "ValidationException" in msg:
                # Drop a parameter the model refuses, remember it, and record it as ignored.
                for param, needles in (("top_p", ("top_p", "topp")), ("top_k", ("top_k", "topk")),
                                       ("stop", ("stop",))):
                    if param not in rejected and any(n in low for n in needles):
                        rejected.add(param)
                        break
                else:
                    raise LLMError(f"Bedrock rejected the request for {model}: {msg[:300]}") from err
                continue
            if "AccessDeniedException" in msg:
                raise LLMError(f"Bedrock access denied for {model}. Enable this model in the Bedrock console "
                               f"(Model access) for region {region}, and check the key: {msg[:200]}") from err
            raise LLMError(f"Bedrock call failed: {msg[:300]}") from err

    blocks = resp.get("output", {}).get("message", {}).get("content", [])
    text = "".join(b.get("text", "") for b in blocks)
    ignored += sorted({"top_p": "top_p", "top_k": "top_k", "stop": "stop_sequences"}[p] for p in rejected)
    if "top_k" not in rejected and not _bedrock_top_k_field(model, 1):
        ignored.append("top_k")
    applied = {k: v for k, v in cfg.items() if k not in ignored and k != "model"}
    applied["temperature"] = temperature
    usage = resp.get("usage", {})
    return text, {"applied_config": applied, "ignored_params": ignored, "adjusted_params": adjusted,
                  "usage": {"input_tokens": usage.get("inputTokens"), "output_tokens": usage.get("outputTokens")}}


def _call_llm(system: str | None, user: str, cfg: dict[str, Any], provider: str, model: str) -> tuple[str, dict[str, Any]]:
    """Call the selected live provider. Raises LLMError instead of faking a reply."""
    if provider == "gemini":
        return _call_gemini(system, user, cfg, model)
    if provider in ("openrouter", "openai"):
        return _call_openai_compatible(provider, system, user, cfg, model)
    if provider == "bedrock":
        return _call_bedrock(system, user, cfg, model)
    raise LLMError(f"Unknown provider '{provider}'")


# ------------------------------------------------------------------------------
# Offline simulator (only used when no provider is configured or REACT_SIMULATE=1)
# ------------------------------------------------------------------------------
_SIM_OPENERS = {
    "friendly_advisor": "Happy to help! ",
    "strict_auditor": "Verified figures: ",
    "creative_curator": "Picture this: ",
}


def _simulate_step(state: "ReActAgentState") -> str:
    """Deterministic stand-in for the LLM. Looks only at the traveler's message and the
    trace (never at the prompt text), and builds its Final Answer from real Observations."""
    query = state["user_query"].lower()
    called = [s["content"].split("(")[0] for s in state["react_steps"] if s["type"] == "action"]
    observations = [json.loads(s["content"]) for s in state["react_steps"] if s["type"] == "observation"]
    last_obs = observations[-1] if observations else {}
    opener = _SIM_OPENERS.get(state["persona_key"], "")

    email = re.search(r"[^@\s,]+@[^@\s,]+\.[a-z]+", state["user_query"], flags=re.I)
    if email and "record_customer_lead" not in called:
        name = re.search(r"(?i:i'?m|i am|name(?: is)?:?)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)", state["user_query"])
        return (
            "Thought: The traveler shared contact details and wants a consultation, so I log a lead.\n"
            f'Action: record_customer_lead(name="{name.group(1) if name else "Traveler"}", '
            f'email="{email.group(0)}", notes="{state["user_query"][:120].replace(chr(34), chr(39))}")'
        )
    if email:
        return "Thought: The lead is logged.\nFinal Answer: " + opener + last_obs.get("message", "Your inquiry is noted.")

    if any(w in query for w in ("cancel", "refund")):
        if "lookup_policy_rule" not in called:
            return 'Thought: I need the approved cancellation policy.\nAction: lookup_policy_rule(topic="cancellation")'
        if "record_unanswered_inquiry" not in called:
            q = state["user_query"].replace('"', "'")
            return f'Thought: No approved terms exist, so I log the question.\nAction: record_unanswered_inquiry(question="{q}")'
        return ("Thought: The question is logged.\nFinal Answer: " + opener +
                "We have no published cancellation or refund terms, so I've logged your question for our team.")

    if any(w in query for w in ("flight", "airfare", "visa")):
        if "lookup_policy_rule" not in called:
            topic = "visas" if "visa" in query and "flight" not in query else "flights"
            return f'Thought: I need the approved policy on {topic}.\nAction: lookup_policy_rule(topic="{topic}")'
        return "Thought: The policy is confirmed.\nFinal Answer: " + opener + last_obs.get("policy", "")

    pkg = _match_package(next((k for k in CATALOG if k.split()[0] in query), ""))
    wants_budget = any(w in query for w in ("cost", "budget", "price", "how much", "calculate"))
    if pkg and wants_budget:
        if "calculate_trip_budget" not in called:
            travelers = re.search(r"(\d+)\s*(?:people|persons|travelers|travellers|adults|pax)", query)
            return (
                f"Thought: I need the budget for {pkg['name']}.\n"
                f'Action: calculate_trip_budget(package_name="{pkg["name"]}", '
                f"num_travelers={travelers.group(1) if travelers else 2}, "
                f"room_upgrade={'upgrade' in query}, private_guide={'guide' in query})"
            )
        return (
            "Thought: I have the verified calculation.\nFinal Answer: " + opener +
            f"{last_obs.get('package')} for {last_obs.get('travelers')} travelers is estimated at "
            f"{last_obs.get('estimated_total_usd')} ({last_obs.get('base_rate_per_person')} per person base). "
            "International airfare is excluded."
        )

    if "search_travel_packages" not in called:
        topic = pkg["name"] if pkg else state["user_query"]
        return f'Thought: I need to search the catalog.\nAction: search_travel_packages(query="{topic}")'
    if last_obs.get("status") == "success":
        p = last_obs["packages"][0]
        return (
            "Thought: I found the package details.\nFinal Answer: " + opener +
            f"{p['name']} ({p['duration']}, ${p['base_price_pp']:,} per person): {p['highlights']}."
        )
    return "Thought: Nothing matched.\nFinal Answer: " + opener + last_obs.get("message", "I couldn't find that package.")


# ==============================================================================
# SECTION 5: Response Parsing
# ==============================================================================
# Action / Final Answer / Thought may appear mid-line or wrapped in markdown (**Action:**);
# Observation must start a line so a Thought that merely mentions "the observation" is not cut.
_ACTION_RE = re.compile(r"\baction[\s*`]*:[\s*`]*([A-Za-z_]\w*)\s*\(", re.I)
_FINAL_RE = re.compile(r"\bfinal\s+answer[\s*`]*:[\s*`]*", re.I)
_THOUGHT_RE = re.compile(r"\bthought[\s*`]*:[\s*`]*", re.I)
_OBS_RE = re.compile(r"^[\s>*`]*observation[\s*`]*:", re.I | re.M)


def _balanced_args(text: str, open_idx: int) -> str:
    """Return the text between the '(' at open_idx and its matching ')', honouring quotes."""
    depth, quote, i = 0, None, open_idx
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == "\\":
                i += 1
            elif ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1:i]
        i += 1
    # Unbalanced: take the rest of the line
    return text[open_idx + 1:].split("\n")[0].rstrip(")` ")


def _literal(node: ast.AST) -> Any:
    if isinstance(node, ast.Name):  # JSON-style true/false/null or a bare word
        return {"true": True, "false": False, "null": None, "none": None}.get(node.id.lower(), node.id)
    return ast.literal_eval(node)


def _parse_args(raw_args: str) -> dict[str, Any]:
    """Parse `a="x", b=2` (or positional values) with Python's AST, falling back to regex."""
    raw_args = raw_args.strip()
    if not raw_args:
        return {}
    try:
        call = ast.parse(f"_f({raw_args})", mode="eval").body
        params: dict[str, Any] = {kw.arg: _literal(kw.value) for kw in call.keywords if kw.arg}
        if call.args:
            params["__positional__"] = [_literal(a) for a in call.args]
        return params
    except (SyntaxError, ValueError):
        pass
    params = {}
    for k, dq, sq, bare in re.findall(r'(\w+)\s*[=:]\s*(?:"([^"]*)"|\'([^\']*)\'|([^,\s)]+))', raw_args):
        val: Any = dq or sq or bare
        if bare:
            low = bare.lower()
            if low in ("true", "false"):
                val = low == "true"
            elif re.fullmatch(r"-?\d+", bare):
                val = int(bare)
            elif re.fullmatch(r"-?\d+\.\d+", bare):
                val = float(bare)
        params[k] = val
    return params


def _extract_thought(segment: str) -> str:
    matches = list(_THOUGHT_RE.finditer(segment))
    if not matches:
        return ""
    thought = _OBS_RE.split(segment[matches[-1].end():])[0]
    return thought.strip().rstrip("*`").strip()


def parse_react_output(text: str) -> tuple[str, str | None, dict[str, Any] | None, str | None]:
    """Parse raw LLM response into (thought, action_name, action_params, final_answer).

    - An Action that appears before any Final Answer wins: everything after it (e.g. an
      Observation or Final Answer the model invented itself) is discarded, because only
      the act node may produce Observations.
    - Text with no ReAct markers at all is treated as a direct final answer.
    - Text with a Thought but neither Action nor Final Answer returns (thought, None, None, None),
      which the graph treats as a format error and retries.
    """
    text = (text or "").strip()
    obs = _OBS_RE.search(text)
    action = _ACTION_RE.search(text)
    final = _FINAL_RE.search(text)
    if obs and final and final.start() > obs.start() and not (action and action.start() < obs.start()):
        final = None  # a Final Answer that follows an invented Observation is not trustworthy

    if action and (not final or action.start() < final.start()):
        thought = _extract_thought(text[:action.start()])
        params = _parse_args(_balanced_args(text, action.end() - 1))
        return thought, action.group(1), params, None

    if final:
        thought = _extract_thought(text[:final.start()])
        answer = _OBS_RE.split(text[final.end():])[0].strip()
        return thought, None, None, answer

    if _THOUGHT_RE.search(text):
        return _extract_thought(text), None, None, None
    return "", None, None, text or None


# ==============================================================================
# SECTION 6: ReAct State Definition & LangGraph Flow
# ==============================================================================
class ReActAgentState(TypedDict, total=False):
    """LangGraph state representation for the manual ReAct loop."""
    user_query: str
    persona_key: str
    persona_requested: str
    persona_version: str  # "v1" | "v2"
    use_cot: bool
    few_shot: bool
    instruction_placement: str  # "system" | "user"
    instructions_mode: str  # "full" | "format_only"
    loop_guards: bool
    loop_guard_events: list[str]
    llm_config: dict[str, Any]
    provider: str
    model: str
    conversation_history: list[dict[str, str]]
    react_steps: list[dict[str, str]]  # list of {type: 'thought'|'action'|'observation', content: str}
    current_thought: str
    action_name: str | None
    action_params: dict[str, Any] | None
    final_answer: str | None
    stopped_reason: str | None
    format_error: bool
    format_errors: int
    error: str | None
    llm_calls: list[dict[str, Any]]
    iteration: int
    guardrails: bool
    guardrail_events: list[str]


def _message_text(content: Any) -> str:
    """Gradio may pass message content as a string or a list of parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(_message_text(c) for c in content)
    if isinstance(content, dict):
        return str(content.get("text") or content.get("content") or "")
    return str(content or "")


def _format_history(history: list[dict[str, Any]]) -> str:
    lines = []
    for msg in history[-HISTORY_TURNS:]:
        speaker = "Traveler" if msg.get("role") == "user" else "Agent"
        text = _message_text(msg.get("content")).strip()
        if text:
            lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def build_prompt_parts(state: ReActAgentState) -> tuple[str | None, str]:
    """Return (system_prompt, user_message). With instruction_placement='user' the system
    prompt is None and persona + instructions travel inside the user message instead."""
    persona_data = PERSONA_PROFILES.get(state["persona_key"], PERSONA_PROFILES["friendly_advisor"])
    persona_ver = state.get("persona_version", "v1")
    instr_key = "prompt_tone_instructions_v2" if persona_ver == "v2" else "prompt_tone_instructions"
    persona_prompt_line = persona_data.get(instr_key, persona_data["prompt_tone_instructions"])
    if state.get("use_cot", True):
        cot_directive = "Think step-by-step: always write an explicit 'Thought:' line before every Action or Final Answer."
    else:
        cot_directive = "Do not write Thought lines. Reply with only an 'Action:' line or a 'Final Answer:' line."

    mode = state.get("instructions_mode", "full")
    if mode == "format_only":
        instructions = INSTRUCTION_FORMAT
    else:
        instructions = INSTRUCTION_RULES
        if state.get("few_shot", True):
            instructions += f"\n\n{WORKED_EXAMPLE}\n\n{EXTRA_FEW_SHOT_EXAMPLE}"

    agent_spec = (
        f"PERSONA:\n{persona_prompt_line}\n\n"
        f"INSTRUCTIONS:\n{instructions}\n\n"
        f"{TOOL_DESCRIPTIONS}\n\n"
        f"REASONING DIRECTIVE:\n{cot_directive}"
    )

    task = ""
    history = _format_history(state.get("conversation_history") or [])
    if history:
        task += f"CONVERSATION SO FAR:\n{history}\n\n"
    task += f"CURRENT TASK:\nUser Query: \"{state['user_query']}\"\n\nINTERACTION TRACE SO FAR:\n"
    for step in state.get("react_steps", []):
        task += f"{step['type'].capitalize()}: {step['content']}\n"
    if state.get("format_error"):
        task += ("\n(Your previous reply had neither an Action nor a Final Answer. "
                 "Reply in the required format.)\n")
    task += "\nRespond with your next step (Action: ... OR Final Answer: ...):"

    if state.get("instruction_placement", "system") == "user":
        return None, f"{agent_spec}\n\n{task}"
    return agent_spec, task


# Kept for backward compatibility: the full prompt as a single string
def _build_react_prompt(state: ReActAgentState) -> str:
    system, user = build_prompt_parts(state)
    return f"{system}\n\n{user}" if system else user


# ------------------------------------------------------------------------------
# LangGraph Nodes
# ------------------------------------------------------------------------------
def reason_node(state: ReActAgentState) -> dict[str, Any]:
    """Reasoning step: outputs Thought + Action or Final Answer."""
    iteration = state.get("iteration", 0) + 1
    llm_calls = list(state.get("llm_calls", []))
    system, user = build_prompt_parts(state)
    provider = state.get("provider") or get_active_provider()
    started = time.perf_counter()
    try:
        if provider == "simulated":
            raw_llm, meta = _simulate_step(state), {"applied_config": {}, "ignored_params": ["all (simulated)"]}
        else:
            raw_llm, meta = _call_llm(system, user, state["llm_config"], provider, state["model"])
    except LLMError as err:
        print(f"[REACT ERROR] {err}")
        return {
            "iteration": iteration,
            "error": str(err),
            "final_answer": None,
            "action_name": None,
            "llm_calls": llm_calls + [{"error": str(err), "latency_s": round(time.perf_counter() - started, 3)}],
        }

    llm_calls.append({"raw": raw_llm, "latency_s": round(time.perf_counter() - started, 3), **meta})
    thought, action_name, action_params, final_answer = parse_react_output(raw_llm)
    format_error = not action_name and final_answer is None

    new_steps = list(state.get("react_steps", []))
    if thought:
        new_steps.append({"type": "thought", "content": thought})

    return {
        "current_thought": thought,
        "action_name": action_name,
        "action_params": action_params,
        "final_answer": final_answer,
        "react_steps": new_steps,
        "format_error": format_error,
        "format_errors": state.get("format_errors", 0) + int(format_error),
        "llm_calls": llm_calls,
        "iteration": iteration,
    }


def act_node(state: ReActAgentState) -> dict[str, Any]:
    """Action execution step: runs the selected tool and records Observation."""
    act_name = state.get("action_name")
    act_params = dict(state.get("action_params") or {})
    positional = act_params.pop("__positional__", [])
    new_steps = list(state.get("react_steps", []))
    loop_guard_events = list(state.get("loop_guard_events", []))

    arg_repr = [repr(v) for v in positional] + [f"{k}={v!r}" for k, v in act_params.items()]
    action_content = f"{act_name}({', '.join(arg_repr)})"

    # Repeat-action block: if exact same tool name AND arguments already ran earlier in this turn, don't run it again
    if state.get("loop_guards"):
        earlier_actions = [s["content"] for s in state.get("react_steps", []) if s.get("type") == "action"]
        is_repeat = action_content in earlier_actions
        if not is_repeat:
            curr_key = (act_name, tuple(positional), tuple(sorted((k, repr(v)) for k, v in act_params.items())))
            for ea in earlier_actions:
                match = _ACTION_RE.search(f"Action: {ea}")
                if match:
                    ea_name = match.group(1)
                    ea_raw_args = _balanced_args(f"Action: {ea}", match.end() - 1)
                    ea_params = _parse_args(ea_raw_args)
                    ea_pos = ea_params.pop("__positional__", [])
                    ea_key = (ea_name, tuple(ea_pos), tuple(sorted((k, repr(v)) for k, v in ea_params.items())))
                    if curr_key == ea_key:
                        is_repeat = True
                        break
        if is_repeat:
            new_steps.append({"type": "action", "content": action_content})
            obs_str = json.dumps({
                "status": "error",
                "message": f"You already called {act_name} with these arguments; use the earlier observation or give your Final Answer.",
            })
            new_steps.append({"type": "observation", "content": obs_str})
            loop_guard_events.append(f"Blocked repeat call to {action_content}.")
            return {
                "action_name": None,
                "action_params": None,
                "react_steps": new_steps,
                "loop_guard_events": loop_guard_events,
            }

    new_steps.append({"type": "action", "content": action_content})

    if act_name in TOOLS_BY_NAME:
        try:
            raw_res = TOOLS_BY_NAME[act_name](*positional, **act_params)
            obs_str = json.dumps(raw_res, ensure_ascii=False)
        except Exception as e:
            obs_str = json.dumps({"status": "error", "message": f"Tool execution failed: {e}"})
    else:
        obs_str = json.dumps({"status": "error", "message": f"Unknown tool: '{act_name}'. Use one of: {', '.join(TOOLS_BY_NAME)}"})

    new_steps.append({"type": "observation", "content": obs_str})
    return {
        "action_name": None,
        "action_params": None,
        "react_steps": new_steps,
        "loop_guard_events": loop_guard_events,
    }


# ------------------------------------------------------------------------------
# Optional code guardrails (off by default, so the logged experiments measure the model alone)
# ------------------------------------------------------------------------------
_PACKAGE_PRICES_RE = "|".join(f"{p['base_price_pp']:,}" for p in CATALOG.values())
_AIRFARE_WORD_RE = re.compile(r"\b(airfares?|flights?|fares?|tickets?)\b", re.I)
_DOLLAR_RE = re.compile(r"\$\s?(?!(?:" + _PACKAGE_PRICES_RE + r")\b)\d")
AIRFARE_DISCLAIMER = ("WanderWise doesn't quote airfare, so I can't give flight prices; airlines or a flight "
                      "search site will have current fares for your dates.")


def _redact_airfare_quotes(answer: str) -> tuple[str, int]:
    """Remove sentences that pair airfare/flights with a dollar amount (other than a package price)."""
    removed = 0
    kept_lines = []
    for line in answer.split("\n"):
        sentences = re.split(r"(?<=[.!?])\s+", line)
        keep = [s for s in sentences if not (_AIRFARE_WORD_RE.search(s) and _DOLLAR_RE.search(s))]
        removed += len(sentences) - len(keep)
        if keep or not line.strip():
            kept_lines.append(" ".join(keep))
    return "\n".join(kept_lines).strip(), removed


def guard_node(state: ReActAgentState) -> dict[str, Any]:
    """Deterministic checks applied to the model's finished answer (only when guardrails=True).

    1. Rule 6 enforcement: if a policy lookup said the question must be logged and the model never
       called record_unanswered_inquiry, call it here and add it to the trace.
    2. Rule 5 enforcement: remove any sentence that quotes an airfare dollar amount, and say why.
    """
    steps = list(state.get("react_steps", []))
    events = list(state.get("guardrail_events", []))
    answer = state.get("final_answer") or ""

    actions = [s["content"].split("(")[0] for s in steps if s["type"] == "action"]
    must_log = any(
        s["type"] == "observation" and ("must be logged" in s["content"].lower() or '"unknown_policy"' in s["content"])
        for s in steps
    )
    if must_log and "record_unanswered_inquiry" not in actions:
        result = record_unanswered_inquiry(state["user_query"])
        steps.append({"type": "action", "content": f"record_unanswered_inquiry(question={state['user_query']!r}) [guardrail]"})
        steps.append({"type": "observation", "content": json.dumps(result, ensure_ascii=False)})
        events.append("Logged the question with record_unanswered_inquiry: the policy required it and the model skipped it.")

    redacted, removed = _redact_airfare_quotes(answer)
    if removed:
        answer = f"{redacted}\n\n{AIRFARE_DISCLAIMER}".strip()
        events.append(f"Removed {removed} sentence(s) quoting an airfare price (rule: never invent flight prices).")

    return {"react_steps": steps, "final_answer": answer, "guardrail_events": events}


def should_continue(state: ReActAgentState) -> str:
    """Conditional edge router: 'act' for a tool call, 'reason' to retry a malformed reply,
    'guard' to check a finished answer when guardrails are on, 'force_answer' at step limit if loop_guards, else END."""
    if state.get("error"):
        return END
    if state.get("final_answer"):
        return "guard" if state.get("guardrails") else END
    if state.get("iteration", 0) >= MAX_ITERATIONS:  # Circuit breaker
        if state.get("loop_guards"):
            return "force_answer"
        return END
    if state.get("action_name"):
        return "act"
    if state.get("format_error"):
        return "reason"
    return END


def should_guard_forced(state: ReActAgentState) -> str:
    if state.get("guardrails") and state.get("final_answer"):
        return "guard"
    return END


def force_answer_node(state: ReActAgentState) -> dict[str, Any]:
    """Fallback node triggered when max iterations are hit under loop_guards=True.
    Makes one final LLM call forcing a Final Answer from existing observations."""
    llm_calls = list(state.get("llm_calls", []))
    loop_guard_events = list(state.get("loop_guard_events", []))
    system, user = build_prompt_parts(state)
    force_user = f"{user}\n\n{FORCE_ANSWER_PROMPT}"
    provider = state.get("provider") or get_active_provider()
    started = time.perf_counter()
    try:
        if provider == "simulated":
            raw_llm, meta = _simulate_step(state), {"applied_config": {}, "ignored_params": ["all (simulated)"]}
        else:
            raw_llm, meta = _call_llm(system, force_user, state["llm_config"], provider, state["model"])
    except LLMError as err:
        print(f"[REACT ERROR] {err}")
        return {
            "error": str(err),
            "final_answer": None,
            "stopped_reason": "llm_error",
            "llm_calls": llm_calls + [{"error": str(err), "latency_s": round(time.perf_counter() - started, 3)}],
            "loop_guard_events": loop_guard_events,
        }

    llm_calls.append({"raw": raw_llm, "latency_s": round(time.perf_counter() - started, 3), **meta})
    thought, action_name, action_params, final_answer = parse_react_output(raw_llm)

    new_steps = list(state.get("react_steps", []))
    if thought:
        new_steps.append({"type": "thought", "content": thought})

    if final_answer:
        loop_guard_events.append("Forced final answer produced at step limit.")
        return {
            "final_answer": final_answer,
            "stopped_reason": "forced_answer",
            "react_steps": new_steps,
            "llm_calls": llm_calls,
            "loop_guard_events": loop_guard_events,
        }
    else:
        loop_guard_events.append("Forced final answer attempted at step limit; no final answer returned.")
        return {
            "final_answer": None,
            "stopped_reason": "max_iterations",
            "react_steps": new_steps,
            "llm_calls": llm_calls,
            "loop_guard_events": loop_guard_events,
        }


def build_react_graph() -> Any:
    """Assemble manual LangGraph state graph for the ReAct loop."""
    workflow = StateGraph(ReActAgentState)
    workflow.add_node("reason", reason_node)
    workflow.add_node("act", act_node)
    workflow.add_node("guard", guard_node)
    workflow.add_node("force_answer", force_answer_node)

    workflow.set_entry_point("reason")
    workflow.add_conditional_edges(
        "reason",
        should_continue,
        {
            "act": "act",
            "reason": "reason",
            "guard": "guard",
            "force_answer": "force_answer",
            END: END,
        },
    )
    workflow.add_edge("act", "reason")
    workflow.add_conditional_edges(
        "force_answer",
        should_guard_forced,
        {
            "guard": "guard",
            END: END,
        },
    )
    workflow.add_edge("guard", END)
    return workflow.compile()


# Compiled LangGraph ReAct Application
react_graph = build_react_graph()


def run_react_agent(
    query: str,
    persona_key: str = "friendly_advisor",
    use_cot: bool = True,
    few_shot: bool = True,
    instruction_placement: str = "system",
    instructions_mode: str = "full",
    loop_guards: bool = False,
    llm_overrides: dict[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
    guardrails: bool = False,
    persona_version: str = "v1",
) -> dict[str, Any]:
    """Execute a query through the manual ReAct LangGraph workflow.

    history: prior chat turns as [{"role": "user"|"assistant", "content": ...}] (short-term memory).
    llm_overrides: any of temperature, top_p, top_k, frequency_penalty, presence_penalty, max_tokens, model.
    instructions_mode: "full" (default; operational rules + worked examples) or "format_only" (ablation).
    loop_guards: block repeated tool calls and force final answer from observations at step limit (default False).
    guardrails: run guard_node on the finished answer (auto-log required inquiries, remove airfare quotes).
        Off by default so experiments measure the model and prompt alone; the Gradio app turns it on.
    persona_version: "v1" (default) or "v2" (boundary against outside-catalog cost fabrication).
    """
    persona_requested = persona_key
    route_reason = None
    if persona_key == "auto":
        resolved_persona, route_reason = route_persona_with_reason(query)
    elif persona_key in PERSONA_PROFILES:
        resolved_persona = persona_key
    else:
        raise ValueError(f"Unknown persona '{persona_key}'. Choose from: auto, {', '.join(PERSONA_PROFILES)}")

    if persona_version not in ("v1", "v2"):
        raise ValueError(f"persona_version must be 'v1' or 'v2', got '{persona_version}'")
    if instruction_placement not in INSTRUCTION_PLACEMENTS:
        raise ValueError(f"instruction_placement must be one of {INSTRUCTION_PLACEMENTS}")
    if instructions_mode not in INSTRUCTION_MODES:
        raise ValueError(f"instructions_mode must be one of {INSTRUCTION_MODES}")

    llm_config = resolve_llm_config(resolved_persona, llm_overrides)
    provider = get_active_provider()
    model = llm_config.get("model") or get_active_model(provider)
    initial_state: ReActAgentState = {
        "user_query": query,
        "persona_key": resolved_persona,
        "persona_requested": persona_requested,
        "persona_version": persona_version,
        "use_cot": use_cot,
        "few_shot": few_shot,
        "instruction_placement": instruction_placement,
        "instructions_mode": instructions_mode,
        "loop_guards": loop_guards,
        "loop_guard_events": [],
        "llm_config": llm_config,
        "provider": provider,
        "model": model,
        "conversation_history": list(history or []),
        "react_steps": [],
        "current_thought": "",
        "action_name": None,
        "action_params": None,
        "final_answer": None,
        "stopped_reason": None,
        "format_error": False,
        "format_errors": 0,
        "error": None,
        "llm_calls": [],
        "iteration": 0,
        "guardrails": guardrails,
        "guardrail_events": [],
    }
    started = time.perf_counter()
    final_state = react_graph.invoke(initial_state, config={"recursion_limit": 4 * MAX_ITERATIONS + 4})
    latency = round(time.perf_counter() - started, 3)

    if final_state.get("error"):
        stopped_reason = "llm_error"
        final_answer = f"Sorry, the language model could not be reached ({final_state['error']})."
    elif final_state.get("stopped_reason") == "forced_answer" and final_state.get("final_answer"):
        stopped_reason = "forced_answer"
        final_answer = final_state["final_answer"]
    elif final_state.get("final_answer"):
        stopped_reason = "final_answer"
        final_answer = final_state["final_answer"]
    else:
        stopped_reason = "max_iterations"
        final_answer = (
            f"I couldn't finish reasoning within {MAX_ITERATIONS} steps, so I won't guess. "
            "Could you rephrase or narrow down your question?"
        )

    steps = final_state.get("react_steps", [])
    return {
        "query": query,
        "persona": PERSONA_PROFILES[resolved_persona]["name"],
        "persona_key": resolved_persona,
        "persona_requested": persona_requested,
        "persona_route_reason": route_reason,
        "persona_version": persona_version,
        "use_cot": use_cot,
        "few_shot": few_shot,
        "instruction_placement": instruction_placement,
        "instructions_mode": instructions_mode,
        "loop_guards": loop_guards,
        "provider": provider,
        "model": model,
        "simulated": provider == "simulated",
        "llm_config": {k: v for k, v in llm_config.items() if k != "model"},
        "final_answer": final_answer,
        "stopped_reason": stopped_reason,
        "error": final_state.get("error"),
        "steps": steps,
        "tools_used": [s["content"].split("(")[0] for s in steps if s["type"] == "action"],
        "iterations": final_state.get("iteration", 0),
        "format_errors": final_state.get("format_errors", 0),
        "llm_calls": final_state.get("llm_calls", []),
        "guardrails": guardrails,
        "guardrail_events": final_state.get("guardrail_events", []),
        "loop_guard_events": final_state.get("loop_guard_events", []),
        "latency_s": latency,
    }
