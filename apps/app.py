"""
CropQuery — Gradio Demo
Run: python app.py (or execute in Kaggle notebook with `demo.launch(share=True)`)
"""
import sys
sys.path.insert(0, "/kaggle/working")

import os
import json
import gradio as gr
from PIL import Image
from typing import Optional

from cropquery import CropQueryPipeline
from cropquery.config import CLASS_NAMES_PATH, CONFIDENCE_GATE
from cropquery.translator import IndicTranslator
from cropquery.vision import CropVisionModel

# Load pipeline once at startup
print("Loading CropQuery pipeline...")

pipeline = CropQueryPipeline(
    skip_llava = False,
)
LANGUAGES = IndicTranslator.available_languages()

# Core inference function ─
def run_cropquery(
    image: Optional[Image.Image],
    farmer_question: str,
    target_language: str,
    confirmed_class: Optional[str],
):
    if image is None:
        yield (
            "<p style='color:#888'>Upload an image to start.</p>",
            "", gr.update(visible=False),
            gr.update(choices=[], value=None),
            "", gr.update(visible=False),
        )
        return

    # ── Step 1: Vision — show immediately ──────────────────────────────
    preds = pipeline.vision.predict(image)

    def pred_label(cls):
        crop, cond = cls.split("__", 1) if "__" in cls else ("Unknown", cls)
        return f"{crop.replace('_', ' ')} — {cond.replace('_', ' ')}"

    def _confidence_bar(cls, conf):
        colour = "#2e7d32" if conf >= CONFIDENCE_GATE else "#f57c00" if conf >= 50 else "#c62828"
        return (
            f"<div style='margin-bottom:10px'>"
            f"<div style='font-weight:600;margin-bottom:3px'>{pred_label(cls)} "
            f"<span style='color:#888;font-size:13px'>({conf:.1f}%)</span></div>"
            f"<div style='background:#e0e0e0;border-radius:4px;height:8px;width:100%'>"
            f"<div style='background:{colour};border-radius:4px;height:8px;width:{min(conf,100):.0f}%'></div>"
            f"</div></div>"
        )

    bars_html = "".join(_confidence_bar(cls, conf) for cls, conf in preds)
    is_high = preds and preds[0][1] >= CONFIDENCE_GATE
    badge = (
        "<span style='background:#2e7d32;color:white;padding:2px 8px;border-radius:4px;font-size:12px'>HIGH CONFIDENCE</span>"
        if is_high else
        "<span style='background:#f57c00;color:white;padding:2px 8px;border-radius:4px;font-size:12px'>LOW CONFIDENCE</span>"
    )
    disease_html = f"<div>{badge}</div><br>{bars_html}"
    dropdown_choices = [f"{cls} ({conf:.0f}%)" for cls, conf in preds]

    yield (
        disease_html,
        "⏳ Generating advice...",
        gr.update(visible=not is_high),
        gr.update(choices=dropdown_choices, value=dropdown_choices[0] if dropdown_choices else None),
        "", gr.update(visible=False),
    )

    # ── Step 2: Context + RAG (fast) ───────────────────────────────────
    try:
        weather_ctx  = pipeline.context_fetcher.get_weather()
        location_ctx = pipeline.context_fetcher.get_location()
    except Exception:
        weather_ctx = location_ctx = None

    if confirmed_class:
        # strip the "(83%)" suffix if present
        top_class = confirmed_class.split(" (")[0]
        resnet_hint = top_class
    else:
        top_class   = preds[0][0] if preds else None
        resnet_hint = pipeline.vision.gate(preds, pipeline.vision._last_tensor)

    chunks      = pipeline.retriever.retrieve(top_class, farmer_question or "What should I do?")
    rag_context = pipeline.retriever.build_rag_context(chunks)

    # ── Step 3: Stream LLaVA tokens ────────────────────────────────────
    streamer = pipeline.advisor.generate_streaming(
        image=image, rag_context=rag_context,
        farmer_question=farmer_question or "What should I do?",
        resnet_hint=resnet_hint, confidence=preds[0][1] if preds else None,
        weather_ctx=weather_ctx, location_ctx=location_ctx,
    )
    advice = ""
    for token in streamer:
        advice += token
        yield (
            disease_html, advice,
            gr.update(visible=not is_high),
            gr.update(choices=dropdown_choices, value=dropdown_choices[0] if dropdown_choices else None),
            "", gr.update(visible=False),
        )

    # ── Step 4: Translate + irrigation (after streaming) ───────────────
    if target_language != "English":
        advice = pipeline.translator.translate(advice, target_language)

    irrigation_text = ""
    try:
        irrigation_text = pipeline.context_fetcher.get_irrigation_advice(
            crop=top_class.split("__")[0].replace("_", " ") if top_class else None,
            weather_ctx=weather_ctx,
        )
        # translate irrigation too
        if target_language != "English" and irrigation_text:
            irrigation_text = pipeline.translator.translate(
                irrigation_text, target_language
            )
    except Exception:
        pass

    yield (
        disease_html, advice,
        gr.update(visible=not is_high),
        gr.update(choices=dropdown_choices, value=dropdown_choices[0] if dropdown_choices else None),
        irrigation_text, gr.update(visible=bool(irrigation_text)),
    )


def rerun_with_confirmed(
    image: Optional[Image.Image],
    farmer_question: str,
    target_language: str,
    confirmed_choice: str,
) -> str:
    if image is None or not confirmed_choice:
        return ""

    confirmed_class = confirmed_choice.split(" (")[0]
    retriever  = pipeline.retriever
    advisor    = pipeline.advisor
    translator = pipeline.translator

    # fetch context — same as run_cropquery
    try:
        weather_ctx  = pipeline.context_fetcher.get_weather()
        location_ctx = pipeline.context_fetcher.get_location()
    except Exception:
        weather_ctx = location_ctx = None

    rag_chunks  = retriever.retrieve(confirmed_class, farmer_question)
    rag_context = retriever.build_rag_context(rag_chunks)

    if advisor:
        advice = advisor.generate(
            image           = image,
            rag_context     = rag_context,
            farmer_question = farmer_question,
            resnet_hint     = confirmed_class,
            confidence      = 100.0,
            weather_ctx     = weather_ctx,
            location_ctx    = location_ctx,
        )
    else:
        advice = f"[RAG-only]\n\n{rag_context}"

    if target_language != "English":
        advice = translator.translate(advice, target_language)

    return advice


# Gradio UI ─
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

        with gr.Column(scale=2):
            disease_panel = gr.HTML(
                value="<p style='color:#888'>Upload an image to start.</p>",
                label="Disease Detection",
                elem_id="disease-panel",
            )

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
            with gr.Group(visible=False) as irrigation_group:
                irrigation_output = gr.Textbox(
                    label="💧 Irrigation Advisory",
                    lines=3,
                    elem_id="irrigation-panel",
                    show_copy_button=False,
                )

    run_btn.click(
        fn=run_cropquery,
        inputs=[image_input, question_input, language_dropdown, confirm_dropdown],
        outputs=[disease_panel, advice_output, low_conf_panel, confirm_dropdown, irrigation_output, irrigation_group],
    )

    confirm_btn.click(
        fn=rerun_with_confirmed,
        inputs=[image_input, question_input, language_dropdown, confirm_dropdown],
        outputs=[advice_output],
    )

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
        Ensemble Vision · LLaVA-1.5-7B · ChromaDB RAG · IndicTrans2 
    </div>
    """)

if __name__ == "__main__":
    demo.queue()
    demo.launch(
        share        = True,
        server_port  = 7860,
        show_error   = True,
    )
