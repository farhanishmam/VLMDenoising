#!/usr/bin/env python3
"""
VLM Inference Script
Runs inference on VQA datasets with various VLM models and corruption configurations.

Usage:
    python vlm_inference.py --model gemini --category count --image_type clean --text_type noisy --data_path data.csv --api_key YOUR_KEY
"""

import argparse
import pandas as pd
import os
import json
from PIL import Image
from collections import defaultdict
import importlib


# ==================== MODEL WRAPPERS ====================

class GeminiVLM:
    def __init__(self, api_key):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash")
    
    def generate_content(self, prompt, image):
        response = self.model.generate_content([prompt, image])
        return response.text.strip()


class HuggingFaceVLM:
    """Base class for HuggingFace VLMs with common device handling"""
    def __init__(self, model_class, processor_class, model_name):
        import torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = processor_class.from_pretrained(model_name)
        self.model = model_class.from_pretrained(
            model_name, torch_dtype=torch.float16
        ).to(self.device)
    
    def generate_content(self, prompt, image):
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=10)
        return self.processor.batch_decode(outputs, skip_special_tokens=True)[0]


class Idefics2VLM(HuggingFaceVLM):
    def __init__(self, model_name="HuggingFaceM4/idefics2-8b"):
        from transformers import Idefics2ForConditionalGeneration, AutoProcessor
        super().__init__(Idefics2ForConditionalGeneration, AutoProcessor, model_name)


class InstructBLIPVLM(HuggingFaceVLM):
    def __init__(self, model_name="Salesforce/instructblip-vicuna-7b"):
        from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
        super().__init__(InstructBlipForConditionalGeneration, InstructBlipProcessor, model_name)


class TextualDenoiser:
    def __init__(self, api_key):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash")
    
    def denoise(self, text):
        prompt = f"You are a text denoising assistant. Fix any typos, grammatical errors, word swaps, or OCR errors. Return ONLY the corrected text.\n\nText to denoise: {text}"
        return self.model.generate_content(prompt).text.strip()


# ==================== INFERENCE FUNCTIONS ====================

def get_vlm_model(model_name, api_key=None):
    model_map = {
        'gemini': lambda: GeminiVLM(api_key),
        'idefics2': lambda: Idefics2VLM(),
        'instructblip': lambda: InstructBLIPVLM(),
    }
    if model_name.lower() not in model_map:
        raise ValueError(f"Unsupported model: {model_name}")
    if model_name.lower() == 'gemini' and not api_key:
        raise ValueError("API key required for Gemini")
    return model_map[model_name.lower()]()


def load_image(image_path, image_type, corruption_type=None):
    if image_type == 'clean':
        return Image.open(image_path)
    
    base_dir = os.path.dirname(image_path)
    image_name = os.path.basename(image_path)
    folder_map = {'noisy': 'Noisy', 'denoised': 'Denoised'}
    
    if image_type not in folder_map:
        raise ValueError(f"Invalid image type: {image_type}")
    
    path = os.path.join(base_dir, '..', folder_map[image_type], corruption_type, image_name)
    return Image.open(path)


def get_text(row, text_type, text_denoiser=None):
    text_map = {
        'clean': lambda: row['original_question'],
        'noisy': lambda: row['question'],
        'denoised': lambda: text_denoiser.denoise(row['question']) if text_denoiser else row.get('denoised_question', row['question'])
    }
    if text_type not in text_map:
        raise ValueError(f"Invalid text type: {text_type}")
    return text_map[text_type]()


def create_prompt(question, options, category):
    opts = '\n'.join([f"{k}. {v}" for k, v in options.items()])
    return f"The following are multiple-choice questions about {category}. Answer by choosing the correct option. Give only the letter (e.g., 'A').\nQuestion: {question}\nOptions:\n{opts}\nAnswer:"


def run_inference(args):
    """Main inference function"""
    
    # Load dataset
    data = pd.read_csv(args.data_path)
    
    # Initialize VLM model
    vlm_model = get_vlm_model(args.model, args.api_key)
    
    # Initialize text denoiser if needed
    text_denoiser = None
    if args.text_type == 'denoised' and args.api_key:
        text_denoiser = TextualDenoiser(args.api_key)
    
    # Load or initialize checkpoint
    checkpoint_path = args.checkpoint or f"checkpoint_{args.model}_{args.category}_{args.image_type}_{args.text_type}.json"
    
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            checkpoint_data = json.load(f)
        processed_questions = defaultdict(list, checkpoint_data.get("processed_questions", {}))
        noise_type_accuracy = defaultdict(lambda: {"correct": 0, "total": 0}, 
                                         checkpoint_data.get("accuracy", {}))
        total_correct = checkpoint_data.get("total_correct", 0)
        total_predictions = checkpoint_data.get("total_predictions", 0)
    else:
        processed_questions = defaultdict(list)
        noise_type_accuracy = defaultdict(lambda: {"correct": 0, "total": 0})
        total_correct = 0
        total_predictions = 0
    
    # Filter by category if specified
    if args.category != 'all':
        data = data[data["category"] == args.category]
    
    print(f"Running inference on {len(data)} questions...")
    
    # Iterate over dataset
    for index, row in data.iterrows():
        question_id = row["id"]
        corruption_type = row.get("modified_question_function_name", "none")
        
        # Skip if already processed
        if corruption_type in processed_questions[question_id]:
            continue
        
        try:
            # Get image path and load image
            image_path_rel = row.get("path", "")
            if args.image_dir:
                image_path = os.path.join(args.image_dir, image_path_rel)
            else:
                image_path = image_path_rel
            
            image = load_image(image_path, args.image_type, corruption_type)
            
            # Get question text
            question = get_text(row, args.text_type, text_denoiser)
            
            # Prepare options
            options = {key: row[key] for key in ["A", "B", "C", "D"] if key in row}
            
            # Create prompt
            prompt = create_prompt(question, options, row["category"])
            
            # Run inference
            response = vlm_model.generate_content(prompt, image)
            predicted_answer = response.split()[-1] if response else "?"
            
            # Evaluate accuracy
            actual_answer = row["answer"]
            if isinstance(actual_answer, str) and len(actual_answer) > 2:
                actual_answer = actual_answer[2]  # Extract letter from "A)" format
            
            if predicted_answer == actual_answer:
                noise_type_accuracy[corruption_type]["correct"] += 1
                total_correct += 1
            
            noise_type_accuracy[corruption_type]["total"] += 1
            total_predictions += 1
            
            # Mark as processed
            processed_questions[question_id].append(corruption_type)
            
            # Save checkpoint periodically
            if total_predictions % args.checkpoint_freq == 0:
                checkpoint_data = {
                    "processed_questions": dict(processed_questions),
                    "accuracy": dict(noise_type_accuracy),
                    "total_correct": total_correct,
                    "total_predictions": total_predictions,
                }
                with open(checkpoint_path, "w") as f:
                    json.dump(checkpoint_data, f)
                print(f"Checkpoint saved at {checkpoint_path}")
            
            # Print progress
            if total_predictions % 10 == 0:
                print(f"Progress: {total_correct}/{total_predictions} = "
                      f"{100*total_correct/total_predictions:.2f}%")
        
        except Exception as e:
            print(f"Error processing question {question_id}: {e}")
            continue
    
    # Final checkpoint save
    checkpoint_data = {
        "processed_questions": dict(processed_questions),
        "accuracy": dict(noise_type_accuracy),
        "total_correct": total_correct,
        "total_predictions": total_predictions,
    }
    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint_data, f)
    
    # Print final results
    print(f"\n{'='*60}")
    print(f"Final Results: {total_correct}/{total_predictions} = "
          f"{100*total_correct/total_predictions:.2f}% accuracy")
    print(f"\nAccuracy by corruption type:")
    for corruption, counts in sorted(noise_type_accuracy.items()):
        acc = 100 * counts['correct'] / counts['total'] if counts['total'] > 0 else 0
        print(f"  {corruption:20s}: {counts['correct']:3d}/{counts['total']:3d} = {acc:5.2f}%")
    print(f"{'='*60}\n")
    print(f"Checkpoint saved to: {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Run VLM inference with flexible image/text configurations'
    )
    
    # Model selection
    parser.add_argument('--model', type=str, required=True,
                       choices=['gemini', 'idefics2', 'instructblip', 'llava', 'janus'],
                       help='VLM model to use')
    parser.add_argument('--api_key', type=str, default=None,
                       help='API key for cloud-based models (Gemini)')
    
    # Data configuration
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to CSV file with questions')
    parser.add_argument('--image_dir', type=str, default=None,
                       help='Base directory for images')
    parser.add_argument('--category', type=str, default='all',
                       choices=['all', 'count', 'order', 'trick', 'vcr'],
                       help='Question category to process')
    
    # Image/Text configuration
    parser.add_argument('--image_type', type=str, required=True,
                       choices=['clean', 'noisy', 'denoised'],
                       help='Type of image to use')
    parser.add_argument('--text_type', type=str, required=True,
                       choices=['clean', 'noisy', 'denoised'],
                       help='Type of text to use')
    
    # Checkpoint configuration
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint file (will be created if not exists)')
    parser.add_argument('--checkpoint_freq', type=int, default=50,
                       help='Save checkpoint every N predictions')
    
    args = parser.parse_args()
    
    # Validate API key for Gemini
    if args.model == 'gemini' and not args.api_key:
        parser.error("--api_key is required for Gemini model")
    
    run_inference(args)


if __name__ == "__main__":
    main()

