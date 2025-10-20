"""
Save Attention Data for Offline Analysis

This script runs inference on N samples and saves all attention weights,
model outputs, and metadata for later analysis without GPU.

Usage:
    python mechanistic/experiments/save_attention_data.py \
        --num-samples 10 \
        --output-file data/attention_cache.pt
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from dataset_zoo.aro_datasets import get_controlled_images_b


def parse_args():
    parser = argparse.ArgumentParser(description="Save attention data for offline analysis")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/leonardo_work/EUHPC_D27_102/compmech/models/Qwen3-VL-4B-Instruct",
        help="Path to Qwen3-VL model"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/leonardo_work/EUHPC_D27_102/compmech/whatsup_vlms_data",
        help="Root directory for dataset"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to process"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="data/attention_cache/qwen3vl_attention_data.pt",
        help="Output file path for saved data"
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Also save images (increases file size)"
    )
    return parser.parse_args()


def extract_all_data(model, processor, image, caption, question, object1, object2, save_image=False):
    """
    Extract all attention weights and metadata for a single sample.

    Returns:
        dict with all data needed for offline analysis
    """
    # Prepare messages in Qwen format
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]

    # Apply chat template
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Process inputs
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Forward pass with attention
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
            return_dict=True
        )

    # Check if attention was captured
    if not hasattr(outputs, 'attentions') or outputs.attentions is None:
        raise RuntimeError("No attention weights found in outputs!")

    attentions = outputs.attentions

    # Decode tokens
    token_ids = inputs['input_ids'][0]
    tokens_list = [processor.tokenizer.decode(t) for t in token_ids]

    # Find vision token boundaries
    vision_start_idx = None
    vision_end_idx = None
    vision_start_token = "<|vision_start|>"
    vision_end_token = "<|vision_end|>"

    for i, token in enumerate(tokens_list):
        if vision_start_token in token:
            vision_start_idx = i
        if vision_end_token in token:
            vision_end_idx = i
            break

    if vision_start_idx is None or vision_end_idx is None:
        raise RuntimeError(f"Could not find vision tokens!")

    num_image_tokens = vision_end_idx - vision_start_idx - 1

    # Calculate grid dimensions
    img_height, img_width = image.size[1], image.size[0]
    img_aspect_ratio = img_width / img_height

    factors = []
    for i in range(1, int(np.sqrt(num_image_tokens)) + 1):
        if num_image_tokens % i == 0:
            factors.append((i, num_image_tokens // i))

    best_factors = min(factors, key=lambda f: abs(f[1]/f[0] - img_aspect_ratio))
    grid_height, grid_width = best_factors

    # Extract cross-attention from all layers
    text_start_idx = vision_end_idx + 1
    all_cross_attn = []

    for layer_idx, layer_attn in enumerate(attentions):
        avg_attn = layer_attn[0].mean(0)  # Average over heads
        cross_attn_layer = avg_attn[text_start_idx:, vision_start_idx+1:vision_end_idx]
        all_cross_attn.append(cross_attn_layer.cpu().float())  # Convert to float32 for numpy compatibility

    text_tokens = tokens_list[text_start_idx:]

    # Get model prediction (generate output)
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False
        )

    # Decode prediction
    generated_text = processor.batch_decode(
        generated_ids[:, inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )[0]

    # Prepare data dict
    data = {
        'caption': caption,
        'question': question,
        'object1': object1,
        'object2': object2,
        'prediction': generated_text,
        'tokens_list': tokens_list,
        'text_tokens': text_tokens,
        'vision_start_idx': vision_start_idx,
        'vision_end_idx': vision_end_idx,
        'text_start_idx': text_start_idx,
        'num_image_tokens': num_image_tokens,
        'grid_height': grid_height,
        'grid_width': grid_width,
        'img_width': img_width,
        'img_height': img_height,
        'all_cross_attn': all_cross_attn,  # List of tensors, one per layer
        'num_layers': len(attentions),
    }

    # Optionally save image
    if save_image:
        data['image'] = np.array(image)

    return data


def main():
    args = parse_args()

    # Create output directory
    output_dir = os.path.dirname(args.output_file)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading Qwen3-VL from: {args.model_path}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager"
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    print(f"Model loaded on {model.device}")

    # Load dataset
    print(f"\nLoading dataset from {args.data_dir}")
    dataset = get_controlled_images_b(
        image_preprocess=None,
        download=False,
        root_dir=args.data_dir
    )
    print(f"Dataset size: {len(dataset)}")

    # Process samples
    all_data = []
    num_samples = min(args.num_samples, len(dataset))

    print(f"\nProcessing {num_samples} samples...")

    for idx in range(num_samples):
        print(f"\n{'='*60}")
        print(f"Processing sample {idx}/{num_samples}")
        print(f"{'='*60}")

        # Get sample
        sample = dataset[idx]
        image = sample['image_options'][0]
        caption = sample['caption_options'][0]

        # Extract objects
        words = caption.split()
        object1 = words[1]
        object2 = words[-1]

        # Create question
        question = f"Where is the {object1} in relation to the {object2}? Answer with left, right, front or behind."

        print(f"Caption: {caption}")
        print(f"Question: {question}")

        # Extract all data
        data = extract_all_data(
            model, processor, image, caption, question,
            object1, object2, save_image=args.save_images
        )

        data['sample_idx'] = idx
        all_data.append(data)

        print(f"Prediction: {data['prediction']}")
        print(f"Extracted {data['num_layers']} layers of attention")
        print(f"Grid: {data['grid_width']}x{data['grid_height']}")

    # Save all data
    print(f"\nSaving data to {args.output_file}")
    torch.save(all_data, args.output_file)

    # Print summary
    file_size_mb = os.path.getsize(args.output_file) / (1024 * 1024)
    print(f"\nSuccess!")
    print(f"Saved {len(all_data)} samples")
    print(f"File size: {file_size_mb:.2f} MB")
    print(f"\nTo load this data later:")
    print(f"  data = torch.load('{args.output_file}')")
    print(f"  sample_0 = data[0]")
    print(f"  # Access cross-attention from layer 5:")
    print(f"  # cross_attn = sample_0['all_cross_attn'][5]")


if __name__ == "__main__":
    main()
