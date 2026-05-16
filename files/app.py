"""
CropQuery — Gradio Demo
Run: python app.py (or execute in Kaggle notebook with `demo.launch(share=True)`)

Features
--------
- Image upload
- Optional farmer question in native script
- Language selector (10 Indian languages + English)
- Disease panel: top-3 predictions with confidence bars
- Advice panel: 4-section structured output in selected language
- Low-confidence fallback: shows top-3 options, asks farmer to confirm
"""

import os
import json
import gradio as gr
from PIL import Image
from typing import Optional

from cropquery import CropQueryPipeline
from cropquery.config import MODEL_PATH, CLASS_NAMES_PATH, CONFIDENCE_GATE
from cropquery.translator import IndicTranslator
from cropquery.vision import CropVisionModel

# ── Load pipeline once at startup ────────────────────────────────────────────
print("Loading CropQuery pipeline...")
pipeline = CropQueryPipeline(
    model_path        = MODEL_PATH,
    class_names_path  = CLASS_NAMES_PATH,
    skip_llava        = False,
)
LANGUAGES = IndicTranslator.available_languages()


# ── Core inference function ───────────────────────────────────────────────────
def run_cropquery(
    image: Optional[Image.Image],
    farmer_question: str,
    target_language: str,
    confirmed_class: Optional[str],    # set by user from low-conf dropdown
) -> tuple:
    """
    Returns: (disease_html, advice_text, low_conf_panel_visible, dropdown_choices)
    """
    if image is None:
        return (
            "<p style='color:#888'>Upload an image to start.</p>",
            "",
            gr.update(visible=False),
            gr.update(choices=[], value=None),
        )

    result = pipeline.run(
        image           = image,
        farmer_question = farmer_question or "What should I do?",
        target_language = target_language,
    )

    # ── Disease panel HTML ────────────────────────────────────────────────────
    def _confidence_bar(conf: float) -> str:
        colour = "#2e7d32" if conf >= CONFIDENCE_GATE else "#f57c00" if conf >= 50 else "#c62828"
        return (
            f"<div style='margin-bottom:10px'>"
            f"  <div style='font-weight:600;margin-bottom:3px'>"
            f"    {pred_label(cls)} <span style='color:#888;font-size:13px'>({conf:.1f}%)</span>"
            f"  </div>"
            f"  <div style='background:#e0e0e0;border-radius:4px;height:8px;width:100%'>"
            f"    <div style='background:{colour};border-radius:4px;height:8px;width:{min(conf,100):.0f}%'></div>"
            f"  </div>"
            f"</div>"
        )

    def pred_label(cls: str) -> str:
        crop, cond = CropVisionModel.parse_label(cls)
        return f"{crop} — {cond}"

    bars_html = "".join(_confidence_bar(cls, conf) for cls, conf in result.predictions)
    badge = (
        "<span style='background:#2e7d32;color:white;padding:2px 8px;border-radius:4px;font-size:12px'>HIGH CONFIDENCE</span>"
        if result.is_high_confidence else
        "<span style='background:#f57c00;color:white;padding:2px 8px;border-radius:4px;font-size:12px'>LOW CONFIDENCE</span>"
    )
    disease_html = f"<div>{badge}</div><br>{bars_html}"

    # ── Low-confidence fallback panel ─────────────────────────────────────────
    show_confirm = not result.is_high_confidence
    dropdown_choices = [f"{cls} ({conf:.0f}%)" for cls, conf in result.predictions]

    return (
        disease_html,
        result.advice,
        gr.update(visible=show_confirm),
        gr.update(choices=dropdown_choices, value=dropdown_choices[0] if dropdown_choices else None),
    )


def rerun_with_confirmed(
    image: Optional[Image.Image],
    farmer_question: str,
    target_language: str,
    confirmed_choice: str,
) -> str:
    """Re-run advice generation with a user-confirmed class."""
    if image is None or not confirmed_choice:
        return ""

    # Extract class name from "Crop__Condition (XX%)"
    confirmed_class = confirmed_choice.split(" (")[0]

    from cropquery.retriever import CropRetriever
    from cropquery.advisor   import LLaVAAdvisor
    from cropquery.translator import IndicTranslator

    retriever   = pipeline.retriever
    advisor     = pipeline.advisor
    translator  = pipeline.translator

    rag_chunks  = retriever.retrieve(confirmed_class, farmer_question)
    rag_context = retriever.build_rag_context(rag_chunks)

    if advisor:
        advice = advisor.generate(
            image           = image,
            rag_context     = rag_context,
            farmer_question = farmer_question,
            resnet_hint     = confirmed_class,
            confidence      = 100.0,   # user confirmed — treat as certain
        )
    else:
        advice = f"[RAG-only]\n\n{rag_context}"

    if target_language != "English":
        advice = translator.translate(advice, target_language)

    return advice


# ── Gradio UI ─────────────────────────────────────────────────────────────────
CSS = """
#title { text-align: center; }
#disease-panel { border: 1px solid #e0e0e0; border-radius: 8px; padding: 16px; }
#advice-panel  { background: #f9fbe7; border-radius: 8px; padding: 16px; min-height: 200px; }
#low-conf-box  { background: #fff3e0; border: 1px solid #f57c00; border-radius: 8px; padding: 16px; }
"""

with gr.Blocks(css=CSS, title="CropQuery") as demo:

    gr.HTML("""
    <div id='title'>
        <h1>🌾 CropQuery</h1>
        <p style='color:#555'>AI-Powered Crop Disease Detection & Farmer Advisory · Multilingual</p>
    </div>
    """)

    with gr.Row():
        # ── Left column: inputs ───────────────────────────────────────────────
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil", label="📷 Upload leaf / crop image", height=300
            )
            question_input = gr.Textbox(
                label="❓ Your question (optional — in any language)",
                placeholder="What spray should I use? / कौन सी दवाई लगाएं?",
                lines=2,
            )
            language_dropdown = gr.Dropdown(
                choices=LANGUAGES,
                value="Hindi",
                label="🌐 Output Language",
            )
            run_btn = gr.Button("🔍 Analyse", variant="primary", size="lg")

        # ── Right column: outputs ─────────────────────────────────────────────
        with gr.Column(scale=2):
            disease_panel = gr.HTML(
                value="<p style='color:#888'>Upload an image to start.</p>",
                label="Disease Detection",
                elem_id="disease-panel",
            )

            # Low-confidence confirmation panel (hidden by default)
            with gr.Group(visible=False, elem_id="low-conf-box") as low_conf_panel:
                gr.Markdown(
                    "⚠️ **Low confidence** — please confirm which disease this looks like:"
                )
                confirm_dropdown = gr.Dropdown(
                    choices=[], label="Select the correct disease", interactive=True
                )
                confirm_btn = gr.Button("✅ Confirm and get advice", variant="secondary")

            advice_output = gr.Textbox(
                label="📋 Advice",
                lines=15,
                elem_id="advice-panel",
                show_copy_button=True,
            )

    # ── Event wiring ──────────────────────────────────────────────────────────
    run_btn.click(
        fn=run_cropquery,
        inputs=[image_input, question_input, language_dropdown, confirm_dropdown],
        outputs=[disease_panel, advice_output, low_conf_panel, confirm_dropdown],
    )

    confirm_btn.click(
        fn=rerun_with_confirmed,
        inputs=[image_input, question_input, language_dropdown, confirm_dropdown],
        outputs=[advice_output],
    )

    # Example images
    gr.Examples(
        examples=[
            ["examples/tomato_late_blight.jpg", "What spray should I use?", "Hindi"],
            ["examples/wheat_rust.jpg",          "How serious is this?",     "Telugu"],
            ["examples/corn_gray_leaf.jpg",       "",                          "English"],
        ],
        inputs=[image_input, question_input, language_dropdown],
        label="Example Images",
    )

    gr.HTML("""
    <div style='text-align:center;color:#888;margin-top:16px;font-size:12px'>
        ResNet50 (85.42%) · LLaVA-1.5-7B · ChromaDB RAG · IndicTrans2 · PRISM AI
    </div>
    """)


if __name__ == "__main__":
    demo.launch(
        share        = True,    # generates public URL (required in Kaggle)
        server_port  = 7860,
        show_error   = True,
    )
