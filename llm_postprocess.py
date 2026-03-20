from groq import Groq
from dotenv import load_dotenv
import json
import os
import torch
from datasets import load_dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from tqdm import tqdm
import evaluate

#Config 
load_dotenv()                               # reads .env file
GROQ_API_KEY    = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.1-8b-instant"
MODEL_PATH      = "./outputs/models"
PROCESSOR_PATH  = "microsoft/trocr-small-handwritten"  # use HF cache (local processor dir missing config.json)
NUM_SAMPLES     = 50                        
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# Load OCR model
print("Loading fine-tuned OCR model...")
processor = TrOCRProcessor.from_pretrained(PROCESSOR_PATH)
model     = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH)

model.config.decoder_start_token_id  = processor.tokenizer.cls_token_id
model.config.pad_token_id            = processor.tokenizer.pad_token_id
model.config.eos_token_id            = processor.tokenizer.sep_token_id

model = model.to(DEVICE)
model.eval()

#Load test data
print("Loading test samples...")
ds = load_dataset("Teklia/IAM-line", split="test")
ds = ds.filter(lambda x: len(x["text"].strip()) > 2)
ds = ds.select(range(NUM_SAMPLES))

#Get raw OCR predictions
print(f"\nRunning OCR on {NUM_SAMPLES} samples...")
raw_preds = []
refs      = []

for i in tqdm(range(0, len(ds), 4)):
    batch  = ds.select(range(i, min(i + 4, len(ds))))
    images = [img.convert("RGB") for img in batch["image"]]
    texts  = [t.strip() for t in batch["text"]]

    pixel_values = processor(
        images=images, return_tensors="pt"
    ).pixel_values.to(DEVICE)

    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values,
            max_new_tokens=64,
        )

    preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
    preds = [p.replace("</s>", "").replace("<pad>", "").strip() for p in preds]

    raw_preds.extend(preds)
    refs.extend(texts)

#LLM Post-processing
print("\nRunning LLM post-processing with Groq...")

client = Groq(api_key=GROQ_API_KEY)

def correct_ocr_with_llm(ocr_text: str) -> str:
    """Send OCR output to LLM for correction."""
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an OCR post-correction assistant. "
                        "Fix ONLY clear OCR errors in the text: "
                        "1. Remove repeated words (e.g. 'song song' → 'song') "
                        "2. Fix obvious character swaps (e.g. 'heg' → 'hey', 'teh' → 'the') "
                        "3. Do NOT rewrite or rephrase sentences "
                        "4. Do NOT change proper nouns or names unless clearly wrong "
                        "5. Do NOT add or remove words unless it's an obvious OCR artifact "
                        "Return ONLY the corrected text, nothing else."
                    )
                },
                {
                    "role": "user",
                    "content": f"Correct this OCR text:\n{ocr_text}"
                }
            ],
            max_tokens=128,
            temperature=0.1,   # low temp = more deterministic
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  LLM error: {e}")
        return ocr_text  # fallback to original if API fails

# Process all predictions
llm_preds = []
for pred in tqdm(raw_preds):
    corrected = correct_ocr_with_llm(pred)
    llm_preds.append(corrected)

#Compare metrics
cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")

cer_raw = cer_metric.compute(predictions=raw_preds, references=refs)
wer_raw = wer_metric.compute(predictions=raw_preds, references=refs)

cer_llm = cer_metric.compute(predictions=llm_preds, references=refs)
wer_llm = wer_metric.compute(predictions=llm_preds, references=refs)

print(f"\n{'='*55}")
print(f"  LLM POST-PROCESSING RESULTS ({NUM_SAMPLES} samples)")
print(f"{'='*55}")
print(f"{'':20} {'RAW OCR':>10} {'LLM FIXED':>10} {'DELTA':>10}")
print(f"{'-'*55}")
cer_delta = cer_raw - cer_llm
wer_delta = wer_raw - wer_llm
print(f"{'CER':20} {cer_raw*100:>9.2f}% {cer_llm*100:>9.2f}% {'+' if cer_delta>0 else '-'}{abs(cer_delta)*100:>8.2f}%")
print(f"{'WER':20} {wer_raw*100:>9.2f}% {wer_llm*100:>9.2f}% {'+' if wer_delta>0 else '-'}{abs(wer_delta)*100:>8.2f}%")
print(f"{'='*55}")

#examples
print(f"\nSAMPLE COMPARISONS (5 examples)")
print("-" * 60)
for i in range(5):
    print(f"\n  [{i+1}]")
    print(f"  REF     : {refs[i]}")
    print(f"  RAW OCR : {raw_preds[i]}")
    print(f"  LLM FIX : {llm_preds[i]}")

#Save results
os.makedirs("reports", exist_ok=True)
results = {
    "num_samples"   : NUM_SAMPLES,
    "raw_ocr"       : {"cer": round(cer_raw, 4), "wer": round(wer_raw, 4)},
    "llm_corrected" : {"cer": round(cer_llm, 4), "wer": round(wer_llm, 4)},
    "improvement"   : {
        "cer_delta": round(cer_delta, 4),
        "wer_delta": round(wer_delta, 4)
    },
    "samples": [
        {"ref": refs[i], "raw": raw_preds[i], "llm": llm_preds[i]}
        for i in range(min(20, len(refs)))
    ]
}

with open("reports/llm_postprocess_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Results saved → reports/llm_postprocess_results.json")