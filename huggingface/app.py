import gradio as gr
from transformers import pipeline
import torch
from datasets import load_dataset
import pandas as pd

class NeuroFluxSpace:
    def __init__(self):
        """
        Initialize the NeuroFlux Space with all models.
        """
        # Load dataset
        self.dataset = load_dataset('csv', data_files={'train': 'data/models.csv'})['train']
        
        # Initialize pipelines for each model
        self.tinybert_pipeline = pipeline(
            "text-classification",
            model="neuroflux/tinybert",
            device=0 if torch.cuda.is_available() else -1
        )
        
        self.emotion_pipeline = pipeline(
            "text-classification",
            model="neuroflux/emotion-detector",
            device=0 if torch.cuda.is_available() else -1
        )
        
        self.mobilenet_pipeline = pipeline(
            "image-classification",
            model="neuroflux/mobilenet",
            device=0 if torch.cuda.is_available() else -1
        )
    
    def analyze_text(self, text):
        """
        Analyze text using both TinyBERT and Emotion Detector.
        """
        # Get TinyBERT analysis
        tinybert_result = self.tinybert_pipeline(text)[0]
        
        # Get Emotion analysis
        emotion_result = self.emotion_pipeline(text)[0]
        
        return {
            "TinyBERT Analysis": {
                "Label": tinybert_result["label"],
                "Score": f"{tinybert_result['score']:0.2f}"
            },
            "Emotion Analysis": {
                "Label": emotion_result["label"],
                "Score": f"{emotion_result['score']:0.2f}"
            }
        }
    
    def analyze_image(self, image):
        """
        Analyze image using MobileNet.
        """
        # Get MobileNet analysis
        mobilenet_result = self.mobilenet_pipeline(image)[0]
        
        return {
            "MobileNet Analysis": {
                "Label": mobilenet_result["label"],
                "Score": f"{mobilenet_result['score']:0.2f}"
            }
        }
    
    def get_model_info(self, model_name):
        """
        Get information about a specific model.
        """
        model_info = self.dataset.filter(lambda x: x["model_name"] == model_name)
        if len(model_info) > 0:
            return model_info[0]
        return None

def create_interface():
    """
    Create the Gradio interface for the NeuroFlux Space.
    """
    space = NeuroFluxSpace()
    
    # Text analysis interface
    text_interface = gr.Interface(
        fn=space.analyze_text,
        inputs=[
            gr.Textbox(
                label="Enter text to analyze",
                placeholder="Type or paste text here...",
                lines=5
            )
        ],
        outputs=[
            gr.JSON(
                label="Analysis Results",
                show_label=True
            )
        ],
        title="Text Analysis",
        description="Analyze text using TinyBERT and Emotion Detector"
    )
    
    # Image analysis interface
    image_interface = gr.Interface(
        fn=space.analyze_image,
        inputs=[
            gr.Image(
                label="Upload an image to analyze",
                type="pil"
            )
        ],
        outputs=[
            gr.JSON(
                label="Analysis Results",
                show_label=True
            )
        ],
        title="Image Analysis",
        description="Analyze images using MobileNet"
    )
    
    # Model information interface
    model_info_interface = gr.Interface(
        fn=space.get_model_info,
        inputs=[
            gr.Dropdown(
                choices=["tinybert", "mobilenet", "emotion-detector"],
                label="Select a model"
            )
        ],
        outputs=[
            gr.JSON(
                label="Model Information",
                show_label=True
            )
        ],
        title="Model Information",
        description="View detailed information about each model"
    )
    
    return gr.TabbedInterface(
        [text_interface, image_interface, model_info_interface],
        ["Text Analysis", "Image Analysis", "Model Info"]
    )

if __name__ == "__main__":
    interface = create_interface()
    interface.launch()
