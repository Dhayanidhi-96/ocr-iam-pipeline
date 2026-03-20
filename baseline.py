import torch 
from datasets import load_dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import evaluate
from tqdm import tqdm

#config
MODEL_NAME  = "microsoft/trocr-small-handwritten"
BATCH_SIZE  = 4   
NUM_SAMPLES = 200     
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")
print(f"Model: {MODEL_NAME}")

#loading model
print("\nLoading processor and model...")
processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
model     = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
model     = model.to(DEVICE)
model.eval()


#loading dataset
print("Loading IAM test set...")
ds = load_dataset("Teklia/IAM-line", split="test")

def filter_sample(text):
    text = text.strip()
    if len(text) <= 2:
        return False
    if not any(c.isalnum() for c in text):
        return False
    return True

ds = ds.filter(lambda x: filter_sample(x["text"]))
ds = ds.select(range(NUM_SAMPLES))
print(f"Running baseline on {len(ds)} samples")

#inference
all_preds = []
all_refs = []

for i in tqdm(range(0, len(ds), BATCH_SIZE)):
    batch = ds.select(range(i, min(i + BATCH_SIZE, len(ds))))
    images = [img.convert("RGB") for img in batch["image"]]
    references = [text.strip() for text in batch["text"]]

    pixel_values = processor(
        images = images,
        return_tensors = "pt"
    ).pixel_values.to(DEVICE)

    with torch.no_grad():
        generate_ids = model.generate(
            pixel_values,
            max_new_tokens=64
        )

    predictions = processor.batch_decode(
        generate_ids,
        skip_special_tokens = True
    )
    predictions = [p.replace("</s>", "").replace("<pad>", "").strip() 
               for p in predictions]
    all_preds.extend(predictions)
    all_refs.extend(references)

cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")

cer = cer_metric.compute(predictions=all_preds, references=all_refs)
wer = wer_metric.compute(predictions=all_preds, references=all_refs)

print(f"\nBASELINE RESULTS  ({NUM_SAMPLES} test samples)")
print(f"   CER : {cer: .4f}  ({cer*100:.2f})%")
print(f"   WER : {wer: .4f}  ({wer*100:.2f})%")

#samplee prediction
print(f"\nSAMPLE PREDICTIONS (10 examples)")
for i in range(10):
    ref = all_refs[i]
    pred = all_preds[i]
    match = "✅" if ref.lower() == pred.lower() else "❌"
    print(f"\n  {match} [{i+1}]")
    print(f"  REF  : {ref}")
    print(f"  PRED : {pred}")

import os, json
os.makedirs("reports", exist_ok=True)
results = {
    "model"       : MODEL_NAME,
    "num_samples" : NUM_SAMPLES,
    "cer"         : round(cer, 4),
    "wer"         : round(wer, 4),
    "samples"     : [
        {"ref": all_refs[i], "pred": all_preds[i]}
        for i in range(min(20, len(all_preds)))
    ]
}

with open("reports/baseline_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Results saved → reports/baseline_results.json")


