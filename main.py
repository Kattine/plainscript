"""Main local Gradio app for comparing base vs fine-tuned outputs."""

import gradio as gr
from scripts.model import (
    load_base_model,
    load_finetuned_model,
    generate_plain_summary,
)

# Sample medical texts for quick testing.
EXAMPLES = [
    [
        "A meta-analysis of randomized controlled trials demonstrated a "
        "statistically significant reduction in glycated hemoglobin (HbA1c) "
        "levels (mean difference -0.5%, 95% CI -0.7 to -0.3, p<0.001) in "
        "patients receiving the intervention compared to placebo, with no "
        "significant increase in hypoglycemic episodes."
    ],
    [
        "The systematic review found moderate-certainty evidence that "
        "cognitive behavioural therapy (CBT) reduces the severity of "
        "chronic insomnia symptoms compared with treatment as usual, "
        "measured by the Pittsburgh Sleep Quality Index (PSQI) at 8 weeks "
        "post-intervention (SMD -0.98, 95% CI -1.23 to -0.73)."
    ],
    [
        "Prophylactic administration of low-molecular-weight heparin was "
        "associated with reduced incidence of venous thromboembolism in "
        "post-operative orthopaedic patients (RR 0.50, 95% CI 0.33 to 0.76), "
        "though with a concomitant increase in minor bleeding events."
    ],
]


def rewrite_comparison(medical_text: str) -> tuple[str, str]:
    """Generate both base and tuned rewrites for the same input."""
    base_output = generate_plain_summary(medical_text, base_model, base_tokenizer)
    tuned_output = generate_plain_summary(medical_text, tuned_model, tuned_tokenizer)
    return base_output, tuned_output


def create_app() -> gr.Blocks:
    """Build the before/after comparison interface."""
    with gr.Blocks(
        title="Patient-Friendly Medical Rewriter",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            "# Patient-Friendly Medical Rewriter\n"
            "Paste a medical abstract or clinical text below to see it rewritten "
            "in plain language. Compare the **base model** (before fine-tuning) "
            "with the **fine-tuned model** (after QLoRA training on Cochrane + "
            "PLOS data)."
        )

        with gr.Row():
            input_text = gr.Textbox(
                label="Medical Text (Input)",
                placeholder="Paste a medical abstract or clinical text here...",
                lines=6,
            )

        submit_btn = gr.Button("Rewrite", variant="primary")

        with gr.Row():
            base_output = gr.Textbox(
                label="Before (Base Qwen2-1.5B)",
                lines=8,
                interactive=False,
            )
            tuned_output = gr.Textbox(
                label="After (Fine-Tuned)",
                lines=8,
                interactive=False,
            )

        submit_btn.click(
            fn=rewrite_comparison,
            inputs=[input_text],
            outputs=[base_output, tuned_output],
        )

        gr.Examples(
            examples=EXAMPLES,
            inputs=[input_text],
            label="Try these examples",
        )

    return app


def main():
    """Load models and start the web app."""
    global base_model, base_tokenizer, tuned_model, tuned_tokenizer

    print("Loading base model...")
    base_model, base_tokenizer = load_base_model()

    print("Loading fine-tuned model...")
    tuned_model, tuned_tokenizer = load_finetuned_model()

    print("Launching app...")
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
