"""Gradio app for before/after medical text rewriting.

This is the entry file used by local and Hugging Face Spaces.
"""

import os

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


MODEL_ID = "Qwen/Qwen2-1.5B-Instruct"
ADAPTER_DIR = os.environ.get("ADAPTER_DIR", "models/qlora-adapter")

SYSTEM_PROMPT = (
    "You are a medical text simplifier. Rewrite the following medical text "
    "into plain language that a patient with no medical background can "
    "understand. Preserve all key findings and conclusions. Do not add "
    "information not present in the original text."
)

EXAMPLES = [
    "A meta-analysis of randomized controlled trials demonstrated a "
    "statistically significant reduction in glycated hemoglobin (HbA1c) "
    "levels (mean difference -0.5%, 95% CI -0.7 to -0.3, p<0.001) in "
    "patients receiving the intervention compared to placebo.",
    "The systematic review found moderate-certainty evidence that cognitive "
    "behavioural therapy reduces the severity of chronic insomnia symptoms "
    "compared with treatment as usual, measured by the Pittsburgh Sleep "
    "Quality Index at 8 weeks post-intervention.",
    "Prophylactic administration of low-molecular-weight heparin was "
    "associated with reduced incidence of venous thromboembolism in "
    "post-operative orthopaedic patients (RR 0.50, 95% CI 0.33 to 0.76), "
    "though with a concomitant increase in minor bleeding events.",
]


def _device() -> str:
  """Choose the best available runtime device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


print("Loading model (this can take a minute on first launch)...")
DEVICE = _device()
DTYPE = torch.float16 if DEVICE in ("cuda", "mps") else torch.float32

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

_base = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=DTYPE, trust_remote_code=True
)
# Keep the adapter unmerged so we can switch it on/off at runtime.
model = PeftModel.from_pretrained(_base, ADAPTER_DIR)
model = model.to(DEVICE)
model.eval()
print(f"Model loaded on {DEVICE}.")


def _generate(text: str, max_new_tokens: int = 400) -> str:
  """Generate one output with the adapter in its current state."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
        )
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def rewrite(text: str):
  """Return base output first, then tuned output for comparison."""
    if not text or not text.strip():
        return "", "Please paste some medical text above to see the rewrite."

    with model.disable_adapter():
        before = _generate(text)
    after = _generate(text)
    return before, after


# Interface

CUSTOM_CSS = """
:root {
  --teal: #0EA5A4;
  --teal-deep: #0B7C7B;
  --coral: #FB7185;
  --ink: #0F172A;
  --mint: #F0FDFA;
  --card: rgba(255, 255, 255, 0.72);
}

.gradio-container {
  background: linear-gradient(135deg, #F0FDFA 0%, #E0F2FE 50%, #FCE7F3 100%);
  background-size: 400% 400%;
  animation: bgshift 18s ease infinite;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
  max-width: 1100px !important;
  margin: auto !important;
}

@keyframes bgshift {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}

/* decorative medical stickers */
#sticker-layer {
  position: fixed;
  inset: 0;
  pointer-events: none;
  z-index: 0;
  overflow: hidden;
}
.sticker {
  position: absolute;
  font-size: 2.4rem;
  opacity: 0.14;
  animation: float 14s ease-in-out infinite;
}
.sticker:nth-child(1) { left: 6%;  top: 18%; animation-delay: 0s;   }
.sticker:nth-child(2) { left: 84%; top: 12%; animation-delay: 2s;   }
.sticker:nth-child(3) { left: 12%; top: 72%; animation-delay: 4s;   }
.sticker:nth-child(4) { left: 78%; top: 68%; animation-delay: 1s;   }
.sticker:nth-child(5) { left: 46%; top: 8%;  animation-delay: 3s;   }
.sticker:nth-child(6) { left: 90%; top: 44%; animation-delay: 5s;   }
.sticker:nth-child(7) { left: 3%;  top: 45%; animation-delay: 2.5s; }

@keyframes float {
  0%, 100% { transform: translateY(0) rotate(-4deg); }
  50%      { transform: translateY(-26px) rotate(4deg); }
}

.hero {
  position: relative;
  z-index: 1;
  text-align: center;
  padding: 26px 16px 6px;
}
.hero h1 {
  font-size: 2.5rem;
  font-weight: 800;
  letter-spacing: -0.02em;
  color: var(--ink);
  margin: 0;
}
.hero h1 .accent { color: var(--teal-deep); }
.hero p {
  color: #475569;
  font-size: 1.05rem;
  margin: 10px auto 0;
  max-width: 620px;
}

/* animated ECG line */
.ecg {
  width: 220px; height: 40px; margin: 14px auto 0; display: block;
}
.ecg path {
  fill: none; stroke: var(--coral); stroke-width: 2.5;
  stroke-dasharray: 300; stroke-dashoffset: 300;
  animation: trace 2.2s linear infinite;
}
@keyframes trace { to { stroke-dashoffset: -300; } }

.panel-card, .gr-box, .gr-panel { position: relative; z-index: 1; }

/* output boxes */
#before_box textarea {
  background: rgba(255,255,255,0.6) !important;
  border-left: 4px solid #94A3B8 !important;
}
#after_box textarea {
  background: rgba(240,253,250,0.9) !important;
  border-left: 4px solid var(--teal) !important;
  font-weight: 500;
}

button.primary, #go_btn {
  background: linear-gradient(135deg, var(--teal) 0%, var(--teal-deep) 100%) !important;
  border: none !important;
  color: white !important;
  font-weight: 600 !important;
}

.disclaimer {
  text-align: center; color: #64748B; font-size: 0.82rem;
  margin-top: 18px; position: relative; z-index: 1;
}
"""

STICKERS_HTML = """
<div id="sticker-layer">
  <div class="sticker">&#129658;</div>
  <div class="sticker">&#128138;</div>
  <div class="sticker">&#10084;&#65039;</div>
  <div class="sticker">&#129656;</div>
  <div class="sticker">&#128137;</div>
  <div class="sticker">&#127973;</div>
  <div class="sticker">&#129701;</div>
</div>
"""

HERO_HTML = """
<div class="hero">
  <h1>Plain<span class="accent">Script</span></h1>
  <p>Paste a dense medical abstract and watch it turn into language a patient
     can actually understand. See the model before and after fine-tuning.</p>
  <svg class="ecg" viewBox="0 0 220 40">
    <path d="M0,20 L60,20 L70,20 L78,4 L88,36 L98,20 L120,20 L128,12 L136,28 L146,20 L220,20"/>
  </svg>
</div>
"""


def build_app() -> gr.Blocks:
  """Build the Gradio interface."""
    with gr.Blocks(css=CUSTOM_CSS, title="PlainScript — Medical Rewriter") as app:
        gr.HTML(STICKERS_HTML)
        gr.HTML(HERO_HTML)

        with gr.Row():
            input_text = gr.Textbox(
                label="Medical text",
                placeholder="Paste a medical abstract or clinical summary here…",
                lines=6,
                elem_id="input_box",
            )

        go_btn = gr.Button("Rewrite it", variant="primary", elem_id="go_btn")

        with gr.Row():
            before_box = gr.Textbox(
                label="Before  ·  base Qwen2-1.5B",
                lines=9,
                interactive=False,
                elem_id="before_box",
            )
            after_box = gr.Textbox(
                label="After  ·  fine-tuned on Cochrane",
                lines=9,
                interactive=False,
                elem_id="after_box",
            )

        gr.Examples(examples=EXAMPLES, inputs=input_text, label="Try an example")

        gr.HTML(
            '<div class="disclaimer">Drafts for review only — not medical '
            "advice. Outputs may contain errors and must be checked by a "
            "qualified professional before any patient-facing use.</div>"
        )

        go_btn.click(fn=rewrite, inputs=input_text, outputs=[before_box, after_box])

    return app


if __name__ == "__main__":
    demo = build_app()
    demo.launch(server_name="0.0.0.0", server_port=7860)
