import torch 
import json
import os
from datasets import load_dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import evaluate
from tqdm import tqdm

#config
FINETUNED_MODEL = "./outputs/models"
PROCESSOR_PATH = "microsoft/trocr-small-handwritten"
BASELINE_JSON = "./reports/baseline_results.json"
NUM_SAMPLES = 200
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device          : {DEVICE}")
print(f"Finetuned Model : {FINETUNED_MODEL}")

print("\nLoading fine-tuned model...")
processor = TrOCRProcessor.from_pretrained(PROCESSOR_PATH)
model     = VisionEncoderDecoderModel.from_pretrained(FINETUNED_MODEL)

model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id           = processor.tokenizer.pad_token_id
model.config.eos_token_id           = processor.tokenizer.sep_token_id

model = model.to(DEVICE)
model.eval()
#loding finetuned model
print("LOADING IAM DATASET...")
ds = load_dataset("Teklia/IAM-line", split="test")

def filter_sample(example):
    text = example["text"].strip()
    if len(text) <= 2: return False
    if not any(c.isalnum() for c in text): return False
    return True

ds = ds.filter(filter_sample)
ds = ds.select(range(NUM_SAMPLES))
print(f"Evaluating on {len(ds)} samples...")

#inference
all_preds = []
all_refs  = []

for i in tqdm(range(0, len(ds), BATCH_SIZE)):
    batch  = ds.select(range(i, min(i + BATCH_SIZE, len(ds))))
    images = [img.convert("RGB") for img in batch["image"] ]
    references = [t.strip() for t in batch["text"]]

    pixel_values = processor(
        images = images,
        return_tensors = "pt"
    ).pixel_values.to(DEVICE)

    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values,
            max_new_tokens        = 64,
            early_stopping        = True,
            no_repeat_ngram_size  = 3,
            length_penalty        = 2.0,
            num_beams             = 4,
        )

    predictions = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )

    predictions = [p.replace("</s>","").replace("<pad>","").strip()
                   for p in predictions]

    all_preds.extend(predictions)
    all_refs.extend(references)

#metrics evaluation

cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")

cer = cer_metric.compute(predictions=all_preds, references=all_refs)
wer = wer_metric.compute(predictions=all_preds, references=all_refs)

#load baseline to compare
with open(BASELINE_JSON) as f:
    baseline = json.load(f)

baseline_cer = baseline['cer']
baseline_wer = baseline['wer']

#Results
print(f"\n{'='*55}")
print(f"  EVALUATION RESULTS — BEFORE vs AFTER FINE-TUNING")
print(f"{'='*55}")
print(f"  {'Metric':<10} {'Baseline':>12} {'Fine-tuned':>12} {'Change':>10}")
print(f"  {'-'*46}")
print(f"  {'CER':<10} {baseline_cer*100:>11.2f}% {cer*100:>11.2f}% {(cer-baseline_cer)*100:>+9.2f}%")
print(f"  {'WER':<10} {baseline_wer*100:>11.2f}% {wer*100:>11.2f}% {(wer-baseline_wer)*100:>+9.2f}%")
print(f"{'='*55}")

print(f"\nSAMPLE PREDICTIONS (10 examples)")
print("-" * 60)
for i in range(10):
    ref  = all_refs[i]
    pred = all_preds[i]
    match = "✅" if ref.lower() == pred.lower() else "❌"
    print(f"\n  {match} [{i+1}]")
    print(f"  REF  : {ref}")
    print(f"  PRED : {pred}")

#Error analysis
print(f"\n Error Analysis")
exact_matches = sum(1 for r,p in zip(all_refs, all_preds)
                    if r.strip() == p.strip())
print(f"  Exact matches     : {exact_matches} / {NUM_SAMPLES} ({exact_matches/NUM_SAMPLES*100:.1f}%)")

case_matches  = sum(1 for r, p in zip(all_refs, all_preds)
                    if r.strip().lower() == p.strip().lower())
print(f"  Case-insensitive  : {case_matches} / {NUM_SAMPLES} ({case_matches/NUM_SAMPLES*100:.1f}%)")

#save
os.makedirs("reports", exist_ok=True)
results = {
    "model"      : FINETUNED_MODEL,
    "num_samples": NUM_SAMPLES,
    "cer"        : round(cer, 4),
    "wer"        : round(wer, 4),
    "baseline_cer": baseline_cer,
    "baseline_wer": baseline_wer,
    "cer_improvement": round(baseline_cer - cer, 4),
    "wer_improvement": round(baseline_wer - wer, 4),
    "samples"    : [{"ref": all_refs[i], "pred": all_preds[i]}
                    for i in range(min(20, len(all_preds)))]
}

with open("reports/finetuned_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Results saved → reports/finetuned_results.json")