"""Gradio Web Application for Assignment C4: ReAct Agent with Configurable Personas."""

import gradio as gr
from react_core import PERSONA_PROFILES, get_active_model, get_active_provider, run_react_agent

PERSONA_CHOICES = {
    "Auto (route by question)": "auto",
    **{f"{p['name']} - Temp: {p['llm_config']['temperature']}": key for key, p in PERSONA_PROFILES.items()},
}


def chat_react(
    message: str,
    history: list[dict],
    persona_choice: str,
    use_cot: bool,
    few_shot: bool = True,
    placement: str = "system",
    guardrails: bool = True,
    loop_guards: bool = True,
):
    """Run query through manual ReAct state machine and return response + formatted trace."""
    if not message.strip():
        return "Please ask a question about WanderWise travel packages, budgets, or policies.", ""

    persona_key = PERSONA_CHOICES.get(persona_choice, "friendly_advisor")
    result = run_react_agent(
        query=message,
        persona_key=persona_key,
        use_cot=use_cot,
        few_shot=few_shot,
        instruction_placement=placement,
        history=history,
        guardrails=guardrails,
        loop_guards=loop_guards,
    )

    # Format trace
    trace_lines = [f"### 🧠 ReAct Execution Trace ({result['persona']})\n"]
    if result["simulated"]:
        trace_lines.append("> ⚠️ **Offline simulator** — no API key configured. These are not real model outputs.\n")
    if result.get("persona_requested") == "auto":
        trace_lines.append(f"> 🤖 **Auto-routed persona:** {result['persona']} ({result.get('persona_route_reason', '')})\n\n")
    for i, step in enumerate(result["steps"], 1):
        stype = step["type"].upper()
        content = step["content"]
        if stype == "THOUGHT":
            trace_lines.append(f"💭 **Thought {i}:** {content}\n")
        elif stype == "ACTION":
            trace_lines.append(f"⚡ **Action {i}:** `{content}`\n")
        elif stype == "OBSERVATION":
            trace_lines.append(f"🔍 **Observation {i}:**\n```json\n{content}\n```\n")

    for event in result["guardrail_events"]:
        trace_lines.append(f"🛡️ **Guardrail:** {event}\n")

    for event in result.get("loop_guard_events", []):
        trace_lines.append(f"🔁 **Loop guard:** {event}\n")

    persona_footer = f"**Persona:** {result['persona']}"
    if result.get("persona_requested") == "auto":
        persona_footer += f" (routed from auto: {result.get('persona_route_reason', '')})"

    trace_lines.append(
        f"🏁 **Stopped:** {result['stopped_reason']} | **Iterations:** {result['iterations']} | "
        f"**Format errors:** {result['format_errors']} | **Latency:** {result['latency_s']}s\n\n"
        f"{persona_footer}\n\n"
        f"**CoT:** {use_cot} | **Few-shot:** {few_shot} | **Instructions in:** {placement} message | "
        f"**Guardrails:** {guardrails} | **Loop guards:** {loop_guards} | "
        f"**Model:** `{result['provider']}/{result['model']}` | **Config:** `{result['llm_config']}`"
    )
    if result["error"]:
        trace_lines.append(f"\n❌ **LLM error:** {result['error']}")
    return result["final_answer"], "\n".join(trace_lines)


def build_c4_demo():
    persona_options = list(PERSONA_CHOICES)
    aria_default = next(
        (label for label, key in PERSONA_CHOICES.items() if key == "friendly_advisor"),
        persona_options[0],
    )
    provider = get_active_provider()

    with gr.Blocks(title="✈️ WanderWise ReAct Agent | Assignment C4") as demo:
        gr.Markdown(
            "# 🧠 WanderWise ReAct Travel Agent — Assignment C4\n"
            "**EECE 503P/798S | Custom ReAct Agent with Configurable Personas (LangGraph State Machine)**\n\n"
            "Experience how **Persona**, **Operational Instructions**, and **LLM Hyperparameters** "
            "drive step-by-step reasoning (*Thought → Action → Observation → Final Answer*).\n\n"
            f"Active provider: `{provider}` / `{get_active_model(provider)}`"
            + ("  \n⚠️ No API key found: running the **offline simulator**." if provider == "simulated" else "")
        )

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=480, label="ReAct Dialogue")
                user_input = gr.Textbox(
                    placeholder="Ask about packages, budgets, inclusions, or unapproved policies...",
                    label="Customer Query",
                )
                with gr.Row():
                    submit_btn = gr.Button("🚀 Submit Query", variant="primary")
                    clear_btn = gr.Button("🗑️ Clear")

                gr.Examples(
                    examples=[
                        "How much would a 7-day trip to Bali cost for 2 people with a deluxe room upgrade?",
                        "What is your cancellation policy and do you issue Italian tourist visas?",
                        "Can you find packages for Japan and tell me if group tours are planned?",
                        "I'm ready to book a consultation for a Rome honeymoon. Name: Dana, email: dana@example.com.",
                    ],
                    inputs=user_input,
                )

            with gr.Column(scale=2):
                gr.Markdown("### ⚙️ Persona & Thinking Controls")
                persona_selector = gr.Dropdown(
                    choices=persona_options,
                    value=aria_default,
                    label="Active Persona & LLM Configuration",
                )
                cot_toggle = gr.Checkbox(label="Enable Chain-of-Thought (Step-by-Step Reasoning)", value=True)
                few_shot_toggle = gr.Checkbox(label="Few-shot (include worked examples)", value=True)
                guardrail_toggle = gr.Checkbox(
                    label="Code guardrails (auto-log questions a policy says must be logged; remove airfare quotes)",
                    value=True,
                )
                loop_guard_toggle = gr.Checkbox(
                    label="Loop guards (block repeated tool calls; answer from observations at the step limit)",
                    value=True,
                )
                placement_radio = gr.Radio(
                    choices=["system", "user"],
                    value="system",
                    label="Where persona + instructions go (system prompt vs in-message)",
                )

                gr.Markdown("### 🔍 Live ReAct Step Trace")
                trace_box = gr.Markdown(
                    value="*Submit a query to inspect the Thought, Action, and Observation steps in real time.*"
                )

        # Chat interaction wiring
        def respond(message, chat_history, persona, cot, few_shot, placement, guardrails, loop_guards):
            chat_history = chat_history or []
            answer, trace = chat_react(message, chat_history, persona, cot, few_shot, placement, guardrails, loop_guards)
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": answer})
            return "", chat_history, trace

        inputs = [user_input, chatbot, persona_selector, cot_toggle, few_shot_toggle, placement_radio, guardrail_toggle, loop_guard_toggle]
        outputs = [user_input, chatbot, trace_box]
        submit_btn.click(respond, inputs=inputs, outputs=outputs)
        user_input.submit(respond, inputs=inputs, outputs=outputs)
        clear_btn.click(lambda: ([], ""), outputs=[chatbot, trace_box])

    return demo


if __name__ == "__main__":
    build_c4_demo().launch()
