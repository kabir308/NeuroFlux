from flask import Flask, render_template, request
import torch
import torch.nn as nn # For TinyBERT classifier head
from transformers import BertTokenizer
import sys
import os
import json
import requests # For fetching imagenet labels
from PIL import Image # For image manipulation
import torchvision.transforms as T # For image transforms
import io # For byte streams
import base64 # For encoding image for HTML display

# Adjust path to import from src and models
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)

from src.models.emotion_detector import EmotionDetector
from models.mobilenet.model import MobileNetV2
from models.tinybert.model import TinyBERT # Import TinyBERT

app = Flask(__name__)

# --- Tokenizer (shared for Emotion Detector and TinyBERT) ---
tokenizer = None
try:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
except Exception as e:
    print(f"Error loading BertTokenizer: {e}")

# --- Emotion Detector Setup ---
EMOTION_LABELS = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'neutral']
emotion_model = None
if tokenizer: # Only if tokenizer loaded successfully
    try:
        emotion_model = EmotionDetector(vocab_size=1000, num_classes=len(EMOTION_LABELS))
        emotion_model.eval()
    except Exception as e:
        print(f"Error loading Emotion Detector model: {e}")

# --- MobileNet Setup ---
IMAGENET_LABELS_URL = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
IMAGENET_LABELS_FILE = "imagenet_class_index.json"
IMAGENET_LABELS = {}
mobilenet_model = None
try:
    if os.path.exists(IMAGENET_LABELS_FILE):
        with open(IMAGENET_LABELS_FILE, 'r') as f: IMAGENET_LABELS = json.load(f)
        print("Loaded ImageNet labels from local file.")
    else:
        response = requests.get(IMAGENET_LABELS_URL, timeout=10)
        response.raise_for_status()
        IMAGENET_LABELS = response.json()
        with open(IMAGENET_LABELS_FILE, 'w') as f: json.dump(IMAGENET_LABELS, f)
        print(f"Saved ImageNet labels to {IMAGENET_LABELS_FILE}")
except Exception as e:
    print(f"Failed to fetch/load ImageNet labels: {e}. Using placeholder.")
    IMAGENET_LABELS = {"0": ["n01440764", "tench"], "1": ["n01443537", "goldfish"], "294": ["n02123045", "tabby_cat"]}

try:
    mobilenet_model = MobileNetV2(num_classes=1000)
    mobilenet_model.eval()
except Exception as e:
    print(f"Error loading MobileNet model: {e}")

image_transforms = T.Compose([
    T.Resize(256), T.CenterCrop(224), T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- TinyBERT Setup (for Sentiment Analysis) ---
SENTIMENT_LABELS = ['negative', 'positive']
tinybert_base_model = None
tinybert_classifier_head = None
if tokenizer: # TinyBERT also depends on the tokenizer
    try:
        tinybert_base_model = TinyBERT() # Assuming it loads pretrained weights by default or is for feature extraction
        tinybert_base_model.eval()
        # Simple linear classifier head
        tinybert_classifier_head = nn.Linear(tinybert_base_model.bert.config.hidden_size, len(SENTIMENT_LABELS))
        tinybert_classifier_head.eval()
        # Note: For a real scenario, tinybert_base_model would need to be loaded with pre-trained weights,
        # and tinybert_classifier_head would need to be trained on a sentiment analysis task.
        # Here, we are setting it up structurally. Predictions will be random if not trained.
    except Exception as e:
        print(f"Error loading TinyBERT model or creating classifier head: {e}")


# --- General App Data ---
models_data = [
    {
        'id': 'emotion-detector',
        'name': 'Emotion Detector',
        'description': 'Detects emotions from text. Uses a simplified vocabulary & custom LSTM model.'
    },
    {
        'id': 'mobilenet',
        'name': 'MobileNet',
        'description': 'Classifies images using a lightweight MobileNetV2 model (ImageNet based).'
    },
    {
        'id': 'tinybert',
        'name': 'TinyBERT Sentiment',
        'description': 'Analyzes sentiment of text using TinyBERT base and a simple classifier head (demo setup).'
    }
]

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/gallery')
def gallery_page():
    return render_template('gallery.html', models=models_data)

@app.route('/demo/emotion-detector', methods=['GET', 'POST'])
def demo_emotion_detector():
    if request.method == 'POST':
        if not tokenizer or not emotion_model:
            return render_template('demo_emotion_detector.html', error="Emotion Detector Model/Tokenizer not available.")
        text_input = request.form.get('text_input', '')
        predicted_label = "N/A"; confidence_score = 0.0
        if text_input:
            try:
                inputs = tokenizer(text_input, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
                hacked_input_ids = inputs['input_ids'] % 1000
                with torch.no_grad(): outputs = emotion_model(hacked_input_ids)
                probabilities = torch.softmax(outputs, dim=1)
                confidence_score_tensor, predicted_idx_tensor = torch.max(probabilities, dim=1)
                predicted_idx, confidence_score = predicted_idx_tensor.item(), confidence_score_tensor.item()
                predicted_label = EMOTION_LABELS[predicted_idx] if 0 <= predicted_idx < len(EMOTION_LABELS) else "Unknown"
            except Exception as e: print(f"Emotion Detector error: {e}"); predicted_label = "Error"
        return render_template('demo_emotion_detector.html', input_text=text_input, predicted_label=predicted_label, confidence=confidence_score)
    return render_template('demo_emotion_detector.html')

@app.route('/demo/mobilenet', methods=['GET', 'POST'])
def demo_mobilenet():
    if request.method == 'POST':
        if not mobilenet_model: return render_template('demo_mobilenet.html', error="MobileNet Model not available.")
        if 'image_file' not in request.files or request.files['image_file'].filename == '':
            return render_template('demo_mobilenet.html', error="No image file selected.")
        file = request.files['image_file']
        try:
            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            img_tensor = image_transforms(img).unsqueeze(0)
            with torch.no_grad(): outputs = mobilenet_model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            top5_prob, top5_catid_tensor = torch.topk(probabilities, 5)
            results = []
            for i in range(top5_prob.size(1)):
                prob = top5_prob[0, i].item()
                cat_id = str(top5_catid_tensor[0, i].item())
                label_info = IMAGENET_LABELS.get(cat_id, ["unknown", "Unknown Class"])
                results.append({"label": label_info[1], "score": prob})
            base64_image_string = base64.b64encode(img_bytes).decode('utf-8')
            return render_template('demo_mobilenet.html', results=results, base64_image_string=base64_image_string)
        except Exception as e: print(f"MobileNet error: {e}"); return render_template('demo_mobilenet.html', error=f"Error: {e}")
    return render_template('demo_mobilenet.html')

@app.route('/demo/tinybert', methods=['GET', 'POST'])
def demo_tinybert():
    if request.method == 'POST':
        if not tokenizer or not tinybert_base_model or not tinybert_classifier_head:
            return render_template('demo_tinybert.html', error="TinyBERT model or tokenizer not available.")

        text_input = request.form.get('text_input', '')
        predicted_label = "N/A"
        confidence_score = 0.0

        if text_input:
            try:
                inputs = tokenizer(text_input, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
                input_ids = inputs['input_ids']
                attention_mask = inputs['attention_mask']

                with torch.no_grad():
                    base_model_outputs = tinybert_base_model(input_ids=input_ids, attention_mask=attention_mask)
                    pooler_output = base_model_outputs['pooler_output'] # Or base_model_outputs.pooler_output
                    logits = tinybert_classifier_head(pooler_output)

                probabilities = torch.softmax(logits, dim=1)
                confidence_score_tensor, predicted_idx_tensor = torch.max(probabilities, dim=1)

                predicted_idx = predicted_idx_tensor.item()
                confidence_score = confidence_score_tensor.item()

                if 0 <= predicted_idx < len(SENTIMENT_LABELS):
                    predicted_label = SENTIMENT_LABELS[predicted_idx]
                else:
                    predicted_label = "Unknown sentiment"
            except Exception as e:
                print(f"Error during TinyBERT prediction: {e}")
                predicted_label = "Error processing request"

        return render_template('demo_tinybert.html',
                               input_text=text_input,
                               predicted_label=predicted_label,
                               confidence=confidence_score)

    return render_template('demo_tinybert.html')

if __name__ == '__main__':
    if not tokenizer: print("CRITICAL: Tokenizer failed to load.")
    if not emotion_model: print("CRITICAL: Emotion Model failed to load.")
    if not mobilenet_model: print("CRITICAL: MobileNet Model failed to load.")
    if not IMAGENET_LABELS: print("WARNING: ImageNet labels are empty/placeholders.")
    if not tinybert_base_model or not tinybert_classifier_head: print("CRITICAL: TinyBERT model/head failed to load.")
    app.run(debug=True)
