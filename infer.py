import argparse
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

MODEL_PATH     = "./outputs/models"
PROCESSOR_NAME = "microsoft/trocr-small-handwritten"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    parser = argparse.ArgumentParser(description="Run OCR inference on a single image")
    parser.add_argument("--image", required=True, help="Path to input image")
    args = parser.parse_args()

    print(f"Device : {DEVICE}")
    print(f"Loading model from {MODEL_PATH}...")

    processor = TrOCRProcessor.from_pretrained(PROCESSOR_NAME)
    model     = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH)
    model     = model.to(DEVICE)
    model.eval()

    image        = Image.open(args.image).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(DEVICE)

    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_new_tokens=64)

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"\nPredicted text: {text}")

if __name__ == "__main__":
    main()
