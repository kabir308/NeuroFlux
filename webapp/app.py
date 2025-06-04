from flask import Flask, render_template, request
import torch
import torch.nn as nn # For TinyBERT classifier head
from transformers import BertTokenizer
import sys
import os
import json
import numpy as np
try:
    from tflite_runtime.interpreter import Interpreter
    try:
        from tflite_runtime.interpreter import load_delegate
        print("Successfully imported load_delegate from tflite_runtime.interpreter")
    except ImportError:
        load_delegate = None
        print("Failed to import load_delegate from tflite_runtime.interpreter. GPU delegation might not be available or might need a different import method.")
except ImportError:
    print("tflite_runtime.interpreter not found, trying tensorflow.lite.Interpreter")
    try:
        import tensorflow.lite as tflite
        Interpreter = tflite.Interpreter
    except ImportError:
        print("Neither tflite_runtime nor tensorflow.lite could be imported. TFLite functionality will be unavailable.")
        Interpreter = None # Placeholder to avoid further errors if Interpreter is None

import requests # For fetching imagenet labels
from PIL import Image, ImageDraw # For image manipulation, ImageDraw for drawing boxes
import torchvision.transforms as T # For image transforms
import io # For byte streams
import base64 # For encoding image for HTML display

from webapp.llm_client import query_llm
# Ensure llm_config is also accessible if not already via llm_client
from webapp.llm_config import LLM_API_ENDPOINT
from webapp.ncnn_object_detector import NanoDetPlusNCNN, NCNN_AVAILABLE, COCO_CLASSES

# Adjust path to import from src and models
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)

from src.models.emotion_detector import EmotionDetector
# from models.mobilenet.model import MobileNetV2 # Comment this out
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

# --- MobileNet Setup --- # This section is replaced for TFLite
IMAGENET_LABELS_URL = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
IMAGENET_LABELS_FILE = "imagenet_class_index.json"
IMAGENET_LABELS = {}
# mobilenet_model = None # Comment this out
# try: # Comment out this whole try-except block for PyTorch MobileNetV2
#     mobilenet_model = MobileNetV2(num_classes=1000)
#     mobilenet_model.eval()
# except Exception as e:
#     print(f"Error loading MobileNet model: {e}")

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

# --- EfficientNet-Lite4 (TFLite) Setup ---
TFLITE_MODEL_PATH = "webapp/models/tflite/efficientnet_lite4_classification_2.tflite"
tflite_interpreter = None
tflite_input_details = None
tflite_output_details = None
model_loaded_successfully = False
delegates_list = []

if Interpreter: # Proceed only if Interpreter was successfully imported
    if os.path.exists(TFLITE_MODEL_PATH):
        # Attempt to load GPU delegate
        if load_delegate:
            try:
                gpu_delegate_name = None
                if os.name == 'posix': # Linux/macOS
                    gpu_delegate_name = 'libtensorflowlite_gpu_delegate.so'
                elif os.name == 'nt': # Windows
                    gpu_delegate_name = 'tensorflowlite_gpu_delegate.dll'

                if gpu_delegate_name:
                    try:
                        delegates_list.append(load_delegate(gpu_delegate_name))
                        print(f"Attempting to use GPU delegate: {gpu_delegate_name}")
                    except Exception as e_delegate_load:
                        print(f"Failed to load delegate '{gpu_delegate_name}': {e_delegate_load}. Will try CPU.")
                        delegates_list = []
                else:
                    print("GPU delegate name not determined for this OS. Using CPU.")

            except Exception as e_delegate_general:
                print(f"Error during GPU delegate setup: {e_delegate_general}. Using CPU.")
                delegates_list = []
        else:
            print("load_delegate function not available. Using CPU for TFLite.")

        # Initialize interpreter (with or without delegate)
        try:
            tflite_interpreter = Interpreter(model_path=TFLITE_MODEL_PATH, experimental_delegates=delegates_list if delegates_list else None)
            tflite_interpreter.allocate_tensors()
            tflite_input_details = tflite_interpreter.get_input_details()
            tflite_output_details = tflite_interpreter.get_output_details()
            model_loaded_successfully = True
            if delegates_list:
                print(f"TFLite model loaded successfully from {TFLITE_MODEL_PATH} WITH GPU delegate.")
            else:
                print(f"TFLite model loaded successfully from {TFLITE_MODEL_PATH} using CPU.")
            print("Input details:", tflite_input_details)
            print("Output details:", tflite_output_details)
        except Exception as e_interpreter_init:
            print(f"Error loading TFLite model or allocating tensors (even with CPU fallback attempt): {e_interpreter_init}")
            model_loaded_successfully = False

    else: # File not found
        print(f"TFLite model file not found at {TFLITE_MODEL_PATH}. Please download it.")
        model_loaded_successfully = False
else:
    print("TFLite Interpreter not available. Skipping TFLite model loading.")


image_transforms = T.Compose([
    T.Resize([224, 224]), # EfficientNet-Lite models often take exact size
    T.ToTensor() # This converts PIL image to [0,1] FloatTensor
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

# --- NanoDet-Plus NCNN Object Detector Setup ---
nanodet_detector = None
NANODET_PARAM_PATH = "webapp/models/ncnn/nanodet-plus-m_320.param"
NANODET_BIN_PATH = "webapp/models/ncnn/nanodet-plus-m_320.bin"
object_detector_loaded_successfully = False

if NCNN_AVAILABLE:
    if os.path.exists(NANODET_PARAM_PATH) and os.path.exists(NANODET_BIN_PATH):
        try:
            nanodet_detector = NanoDetPlusNCNN(
                param_path=NANODET_PARAM_PATH,
                bin_path=NANODET_BIN_PATH,
            )
            object_detector_loaded_successfully = True
            print("NanoDet-Plus NCNN detector initialized successfully.")
        except Exception as e:
            print(f"Error initializing NanoDet-Plus NCNN detector: {e}")
            nanodet_detector = None # Ensure it's None if init fails
    else:
        print(f"NanoDet-Plus NCNN model files not found. Searched for:\n{NANODET_PARAM_PATH}\n{NANODET_BIN_PATH}")
        print("Object detection demo will be unavailable.")
else:
    print("NCNN is not available. NanoDet-Plus object detection demo will be unavailable.")

# --- General App Data ---
models_data = [
    {
        'id': 'emotion-detector',
        'name': 'Emotion Detector',
        'description': 'Detects emotions from text. Uses a simplified vocabulary & custom LSTM model.'
    },
    {
        'id': 'mobilenet', # Route /demo/mobilenet still uses this id
        'name': 'EfficientNet-Lite4 (TFLite Image Classification)',
        'description': 'Classifies images using the EfficientNet-Lite4 model via TensorFlow Lite. Demonstrates optimized edge inference, with experimental GPU delegate support.'
    },
    {
        'id': 'tinybert',
        'name': 'TinyBERT Sentiment',
        'description': 'Analyzes sentiment of text using TinyBERT base and a simple classifier head (demo setup).'
    },
    {
        'id': 'llm',
        'name': 'Interactive LLM Demo',
        'description': 'Interact with a configurable Large Language Model. Requires a local LLM server (e.g., text-generation-webui or Ollama) with an OpenAI-compatible API. Configure the endpoint in webapp/llm_config.py.'
    },
    {
        'id': 'object-detection', # Used to build URL /demo/object-detection
        'name': 'Object Detection (NanoDet-Plus with NCNN)',
        'description': 'Detects objects in images using the NanoDet-Plus model with the NCNN inference engine. This demo highlights fast CPU-based object detection. Note: NCNN setup and full post-processing logic can be complex.'
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
        if not model_loaded_successfully or not tflite_interpreter:
            return render_template('demo_mobilenet.html', error="EfficientNet-Lite TFLite Model not available or failed to load. Check server logs and ensure model file is present at webapp/models/tflite/.")

        if 'image_file' not in request.files or request.files['image_file'].filename == '':
            return render_template('demo_mobilenet.html', error="No image file selected.")

        file = request.files['image_file']
        try:
            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

            img_tensor_chw = image_transforms(img) # Output: [C, H, W], e.g., [3, 224, 224]

            # TFLite models usually expect NHWC: [Batch, Height, Width, Channels]
            img_tensor_nhwc = img_tensor_chw.permute(1, 2, 0).numpy() # H, W, C (NumPy)
            img_tensor_nhwc = np.expand_dims(img_tensor_nhwc, axis=0) # 1, H, W, C (NumPy)

            # Ensure it's float32, which ToTensor() should provide
            if img_tensor_nhwc.dtype != np.float32:
                img_tensor_nhwc = img_tensor_nhwc.astype(np.float32)

            # Check input tensor shape and type compatibility (optional, for debugging)
            # print(f"Input tensor shape: {img_tensor_nhwc.shape}, dtype: {img_tensor_nhwc.dtype}")
            # print(f"Expected input shape: {tflite_input_details[0]['shape']}, type: {tflite_input_details[0]['dtype']}")

            tflite_interpreter.set_tensor(tflite_input_details[0]['index'], img_tensor_nhwc)
            tflite_interpreter.invoke()
            tflite_outputs = tflite_interpreter.get_tensor(tflite_output_details[0]['index'])

            # EfficientNet-Lite output on TF Hub is typically logits. Apply softmax.
            probabilities_tensor = torch.softmax(torch.from_numpy(tflite_outputs), dim=1)

            top5_prob_tensor, top5_catid_tensor = torch.topk(probabilities_tensor, 5)

            results = []
            for i in range(top5_prob_tensor.size(1)):
                prob = top5_prob_tensor[0, i].item()
                cat_id = str(top5_catid_tensor[0, i].item())
                label_info = IMAGENET_LABELS.get(cat_id, ["unknown", "Unknown Class"])
                results.append({"label": label_info[1], "score": prob})

            base64_image_string = base64.b64encode(img_bytes).decode('utf-8')
            return render_template('demo_mobilenet.html', results=results, base64_image_string=base64_image_string)
        except Exception as e:
            print(f"EfficientNet-Lite TFLite error: {e}")
            return render_template('demo_mobilenet.html', error=f"Error processing image with TFLite model: {e}")

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

@app.route('/demo/llm', methods=['GET', 'POST'])
def demo_llm():
    llm_response = None
    error_message = None
    input_prompt = ""

    if request.method == 'POST':
        input_prompt = request.form.get('prompt_input', '')
        if not input_prompt:
            error_message = "Please enter a prompt."
        else:
            # Inform the user that the query is being made
            print(f"Sending prompt to LLM: '{input_prompt}' via endpoint {LLM_API_ENDPOINT}")
            raw_response = query_llm(input_prompt) # This function now handles error string returns

            # Check if the response from query_llm indicates an error
            if raw_response.startswith("Error:"):
                error_message = raw_response
                llm_response = None
            else:
                llm_response = raw_response

    return render_template('demo_llm.html',
                           input_prompt=input_prompt,
                           llm_response=llm_response,
                           error_message=error_message,
                           llm_endpoint=LLM_API_ENDPOINT) # Pass endpoint for display/debug

@app.route('/demo/object-detection', methods=['GET', 'POST'])
def demo_object_detection():
    detection_results_img_b64 = None
    detections_list = []
    error_message = None
    original_img_b64 = None # To display the uploaded image even if detection fails

    if not NCNN_AVAILABLE:
        error_message = "NCNN runtime is not available or failed to load. Object detection demo is disabled."
        return render_template('demo_object_detection.html', error_message=error_message)

    if not object_detector_loaded_successfully or not nanodet_detector:
        error_message = "NanoDet-Plus NCNN Model not available or failed to load. Check server logs."
        return render_template('demo_object_detection.html', error_message=error_message)

    if request.method == 'POST':
        if 'image_file' not in request.files or request.files['image_file'].filename == '':
            error_message = "No image file selected."
            return render_template('demo_object_detection.html', error_message=error_message)

        file = request.files['image_file']
        try:
            img_bytes = file.read()
            img_pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')

            buffered_original = io.BytesIO()
            img_pil.save(buffered_original, format="PNG")
            original_img_b64 = base64.b64encode(buffered_original.getvalue()).decode('utf-8')

            detections_list = nanodet_detector.detect(img_pil.copy())

            img_with_boxes = img_pil.copy()
            draw = ImageDraw.Draw(img_with_boxes)

            for det in detections_list:
                box = det['box']
                label = det['label']
                score = det['score']

                draw.rectangle(box, outline="red", width=2)
                text = f"{label} ({score:.2f})"
                text_position = (box[0], box[1] - 10 if box[1] - 10 > 0 else box[1])
                draw.text(text_position, text, fill="red")

            buffered_processed = io.BytesIO()
            img_with_boxes.save(buffered_processed, format="PNG")
            detection_results_img_b64 = base64.b64encode(buffered_processed.getvalue()).decode('utf-8')

        except Exception as e:
            print(f"Error during object detection: {e}")
            error_message = f"Error during object detection: {e}"
            if not original_img_b64 and 'img_pil' in locals():
                 buffered_original_fallback = io.BytesIO()
                 img_pil.save(buffered_original_fallback, format="PNG")
                 original_img_b64 = base64.b64encode(buffered_original_fallback.getvalue()).decode('utf-8')

    return render_template('demo_object_detection.html',
                           error_message=error_message,
                           original_image_b64=original_img_b64,
                           detection_results_image_b64=detection_results_img_b64,
                           detections=detections_list)

if __name__ == '__main__':
    if not tokenizer: print("CRITICAL: Tokenizer failed to load.")
    if not emotion_model: print("CRITICAL: Emotion Model failed to load.")
    if not model_loaded_successfully: print("CRITICAL: EfficientNet-Lite TFLite Model failed to load or not found. Check logs and model path.")
    if not IMAGENET_LABELS: print("WARNING: ImageNet labels are empty/placeholders.")
    if not tinybert_base_model or not tinybert_classifier_head: print("CRITICAL: TinyBERT model/head failed to load.")
    if NCNN_AVAILABLE and not object_detector_loaded_successfully:
        print("WARNING: NanoDet-Plus NCNN Model failed to load or NCNN files missing. Object detection demo may be unavailable.")
    elif not NCNN_AVAILABLE:
        print("INFO: NCNN runtime not available, object detection demo is disabled.")
    app.run(debug=True)
