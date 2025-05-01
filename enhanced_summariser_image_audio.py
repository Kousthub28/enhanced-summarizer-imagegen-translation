import sys
import time
import torch
import fitz                # PyMuPDF
import gradio as gr
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    pipeline as hf_pipeline
)
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from TTS.api import TTS

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Load T5 Summarizer ---
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained(
    "t5-small",
    torch_dtype=(torch.float16 if device.type=="cuda" else torch.float32)
).to(device)
model.eval()
model.config.use_cache = True

# --- Load Stable Diffusion ---
sd_pipeline = None
try:
    safety = StableDiffusionSafetyChecker.from_pretrained(
        "CompVis/stable-diffusion-safety-checker"
    )
    sd_pipeline = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=(torch.float16 if device.type=="cuda" else torch.float32),
        safety_checker=safety,
        use_safetensors=True
    ).to(device)
    sd_pipeline.scheduler = EulerDiscreteScheduler.from_config(sd_pipeline.scheduler.config)
    sd_pipeline.enable_attention_slicing()
    print("✅ Stable Diffusion loaded")
except Exception as e:
    print(f"⚠️ SD load failed: {e}")
    sd_pipeline = None

# --- Load mBART-50 for Multilingual Translation ---
mbart_tokenizer = MBart50TokenizerFast.from_pretrained(
    "facebook/mbart-large-50-many-to-many-mmt"
)
mbart_model = MBartForConditionalGeneration.from_pretrained(
    "facebook/mbart-large-50-many-to-many-mmt"
).to(device)

# --- Sentiment Analysis ---
sentiment_analyzer = hf_pipeline("sentiment-analysis",
                                 device=(0 if device.type=="cuda" else -1))

# --- Text-to-Speech ---
tts_engine = TTS(
    model_name="tts_models/en/ljspeech/tacotron2-DDC",
    progress_bar=False,
    gpu=(device.type=="cuda")
)

# Supported languages and MBART codes
supported_langs = {
    "en": "en_XX",
    "fr": "fr_XX",
    "es": "es_XX",
    "de": "de_DE",
    "it": "it_IT",
    "pt": "pt_XX"
}

def translate(text, src, tgt):
    """Translate via mBART-50 from src to tgt."""
    mbart_tokenizer.src_lang = supported_langs[src]
    encoded = mbart_tokenizer(text, return_tensors="pt").to(device)
    forced_id = mbart_tokenizer.lang_code_to_id[supported_langs[tgt]]
    generated = mbart_model.generate(
        **encoded,
        forced_bos_token_id=forced_id
    )
    return mbart_tokenizer.decode(generated[0], skip_special_tokens=True)

# --- Core Function ---
def summarize_and_generate(
    text: str,
    upload_file,
    src_lang: str,
    tgt_lang: str,
    max_length: int,
    num_beams: int,
    sd_steps: int,
    guidance_scale: float,
    do_image: bool,
    do_tts: bool
):
    # 1️⃣ Load content
    content = text.strip()
    if upload_file:
        fname = upload_file.name.lower()
        if fname.endswith(".pdf"):
            doc = fitz.open(upload_file.name)
            content = "".join(page.get_text() for page in doc)
        else:
            content = upload_file.read().decode("utf-8")
    if not content:
        return "⚠️ Enter text or upload a file.", None, None

    # 2️⃣ Translate input → English if needed
    if src_lang != "en":
        content = translate(content, src_lang, "en")

    # 3️⃣ Summarization
    inputs = tokenizer("summarize: " + content,
                       return_tensors="pt",
                       truncation=True,
                       max_length=1024).to(device)
    input_tokens = inputs.input_ids.size(1)
    start_time = time.time()
    with torch.no_grad():
        summary_ids = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True
        )
    elapsed = time.time() - start_time
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    summary_tokens = len(summary.split())

    # 4️⃣ Translate summary → target lang if needed
    if tgt_lang != "en":
        summary = translate(summary, "en", tgt_lang)

    # 5️⃣ Sentiment
    sent = sentiment_analyzer(summary)[0]
    sent_label, sent_score = sent["label"], sent["score"]

    # 6️⃣ Image generation
    img, img_status = None, ""
    if do_image and sd_pipeline and device.type=="cuda":
        try:
            with torch.autocast("cuda"):
                img = sd_pipeline(
                    prompt=summary,
                    num_inference_steps=sd_steps,
                    guidance_scale=guidance_scale
                ).images[0]
            img_status = " | 🖼️ Image OK"
        except Exception as e:
            img_status = f" | ⚠️ Img err: {e}"

    # 7️⃣ Text-to-speech
    audio_path, tts_status = None, ""
    if do_tts:
        try:
            out_path = "/content/summary.wav"
            tts_engine.tts_to_file(text=summary, file_path=out_path)
            audio_path = out_path
            tts_status = " | 🔊 Audio OK"
        except Exception as e:
            tts_status = f" | ⚠️ TTS err: {e}"

    # 8️⃣ Status
    status = (
        f"Tokens in: {input_tokens} | out: {summary_tokens}"
        f" | ⏱ {elapsed:.1f}s | Sentiment: {sent_label} ({sent_score:.2f})"
        + img_status + tts_status
    )
    return status + "\n\n" + summary, img, audio_path

# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("## Enhanced Summarizer + ImageGen + Translation (mBART-50)")
    with gr.Row():
        txt = gr.Textbox(lines=5, label="Input Text")
        up = gr.File(label="Upload PDF/TXT", file_types=[".pdf", ".txt"])
    with gr.Row():
        src = gr.Dropdown(list(supported_langs.keys()), value="en", label="Source Lang")
        tgt = gr.Dropdown(list(supported_langs.keys()), value="en", label="Target Lang")
    with gr.Row():
        ml = gr.Slider(20, 200, value=60, step=10, label="Max Summary Length")
        bm = gr.Slider(1, 8, value=4, step=1, label="Beam Width")
    with gr.Row():
        st = gr.Slider(1, 50, value=10, step=1, label="SD Steps")
        gs = gr.Slider(1.0, 10.0, value=7.5, step=0.5, label="Guidance Scale")
    with gr.Row():
        cb1 = gr.Checkbox(label="Generate Image?", value=False)
        cb2 = gr.Checkbox(label="Generate Audio?", value=False)
    btn = gr.Button("Submit")
    out_text = gr.Textbox(lines=8, label="Summary & Status")
    out_img  = gr.Image(type="pil", label="Image")
    out_aud  = gr.Audio(label="Audio")
    btn.click(
        summarize_and_generate,
        [txt, up, src, tgt, ml, bm, st, gs, cb1, cb2],
        [out_text, out_img, out_aud]
    )
    demo.launch(share=True)
