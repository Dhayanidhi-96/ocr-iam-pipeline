import os
import torch
from datasets import load_dataset
from transformers import(
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator
)

#config
MODEL_NAME = "microsoft/trocr-small-handwritten"
OUTPUT_DIR = "./outputs"
EPOCHS     = 3
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device          : {DEVICE}")
print(f"Model           :{MODEL_NAME}")
print(f"Epochs          :{EPOCHS}")
print(f"Batch Size      :{BATCH_SIZE}")
print(f"Learning rate   : {LEARNING_RATE}")

#filter and normalize
def filter_sample(example):
    text = example["text"].strip()
    if len(text) <= 2:
        return False
    if not any(c.isalnum() for c in text):
        return False
    return True

def normalize_text(text : str) -> str:
    return text.strip()

#loading dataset
print("\nLoading IAM dataset...")
ds = load_dataset("Teklia/IAM-line")

train_ds = ds["train"].filter(filter_sample)
val_ds = ds["validation"].filter(filter_sample)

print(f"Training sample : {len(train_ds):,}")
print(f"Validation sample : {len(val_ds):,}")


#required model confif for TrOCR
# ── Load processor & model ───────────────────────────────────
print("\nLoading processor and model...")
processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
model     = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

# ── Required model config for TrOCR ──────────────────────────
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id           = processor.tokenizer.pad_token_id
model.config.vocab_size             = model.config.decoder.vocab_size
model.config.eos_token_id           = processor.tokenizer.sep_token_id
model.config.max_new_tokens         = 64


#preprocessing
def preprocess(batch):
    images = [img.convert("RGB") for img in batch["image"]]
    texts  = [normalize_text(t) for t in batch["text"]]

    pixel_values = processor(
        images=images,
        return_tensors = 'pt',
    ).pixel_values

    labels = processor.tokenizer(
        texts,
        padding="max_length",
        truncation = True,
        max_length = 64,
        return_tensors = 'pt'
    ).input_ids
    labels[labels == processor.tokenizer.pad_token_id] = -100

    return {
        "pixel_values" : pixel_values,
        "labels"       : labels
    }
print("\nPreprocessing train set...")
train_ds = train_ds.map(
    preprocess,
    batched=True,
    batch_size=16,
    remove_columns= [ "image","text"],
    desc = "Processing train"
    )

print("Preprocessing validation set...")
val_ds = val_ds.map(
    preprocess,
    batched=True,
    batch_size=16,
    remove_columns=["image", "text"],
    desc="Processing val"
)

train_ds.set_format("torch")
val_ds.set_format("torch")

#training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir                  = OUTPUT_DIR,
    num_train_epochs            = EPOCHS,
    per_device_eval_batch_size  = BATCH_SIZE,
    per_device_train_batch_size = BATCH_SIZE,
    learning_rate               = LEARNING_RATE,
    warmup_ratio                = 0.1,          
    weight_decay                = 0.01,
    logging_steps               = 50,
    evaluation_strategy         = "epoch",
    save_strategy               = "epoch",
    predict_with_generate       = True,
    fp16                        = False,         # fp16=True causes grad explosion on GTX 1650
    dataloader_num_workers      = 0,
    report_to                   = "none",
    load_best_model_at_end      = True
    )

#trainer
trainer = Seq2SeqTrainer(
    model=model,
    args = training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator= default_data_collator,
    
)

#train
print("\nStarting training...")
trainer.train()


print("\nSaving model and processor...")
os.makedirs(f"{OUTPUT_DIR}/model", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/processor", exist_ok=True)
model.save_pretrained(f"{OUTPUT_DIR}/model")
processor.save_pretrained(f"{OUTPUT_DIR}/processor")

print("\n✅ Training complete!")
print(f"   Model saved → {OUTPUT_DIR}/model")
print(f"   Processor saved → {OUTPUT_DIR}/processor")
