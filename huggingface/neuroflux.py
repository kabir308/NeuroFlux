from datasets import DatasetBuilder, DatasetInfo
from dataclasses import dataclass
from typing import Dict, Any
import torch
import os

@dataclass
class NeuroFluxConfig:
    """Configuration for the NeuroFlux dataset."""
    name: str = "neuroflux"
    version: str = "1.0.0"
    description: str = "Microscopic AI models for edge devices"
    model_types: list = ("tinybert", "mobilenet", "emotion-detector")

class NeuroFluxDataset(DatasetBuilder):
    BUILDER_CONFIGS = [
        NeuroFluxConfig(
            name="tinybert",
            description="TinyBERT model for text classification",
            model_types=["tinybert"]
        ),
        NeuroFluxConfig(
            name="mobilenet",
            description="MobileNet model for image classification",
            model_types=["mobilenet"]
        ),
        NeuroFluxConfig(
            name="emotion-detector",
            description="Emotion detection model",
            model_types=["emotion-detector"]
        )
    ]

    def _info(self) -> DatasetInfo:
        """Return the dataset metadata."""
        return DatasetInfo(
            description="Microscopic AI models for edge devices",
            features={
                "model_name": "string",
                "model_type": "string",
                "pipeline_tag": "string",
                "model_size": "int32",
                "description": "string",
                "target_devices": ["string"],
                "performance": {
                    "inference_time": "string",
                    "memory_usage": "string",
                    "accuracy": "string"
                }
            }
        )

    def _split_generators(self, dl_manager):
        """Return SplitGenerators."""
        return [
            self._generate_examples(
                os.path.join("models", model_type)
            )
            for model_type in self.config.model_types
        ]

    def _generate_examples(self, model_path):
        """Yield examples as (key, example) tuples."""
        for model_type in os.listdir(model_path):
            model_dir = os.path.join(model_path, model_type)
            if os.path.isdir(model_dir):
                # Load model metadata
                with open(os.path.join(model_dir, "README.md"), "r") as f:
                    metadata = self._parse_readme(f.read())
                
                # Convert model to TFLite (if not already done)
                tflite_path = os.path.join(model_dir, f"{model_type}_4bit.tflite")
                if not os.path.exists(tflite_path):
                    self._convert_to_tflite(model_dir, tflite_path)
                
                yield model_type, {
                    "model_name": model_type,
                    "model_type": metadata["model_type"],
                    "pipeline_tag": metadata["pipeline_tag"],
                    "model_size": metadata["model_size"],
                    "description": metadata["description"],
                    "target_devices": metadata["target_devices"],
                    "performance": {
                        "inference_time": metadata["performance"]["inference_time"],
                        "memory_usage": metadata["performance"]["memory_usage"],
                        "accuracy": metadata["performance"]["accuracy"]
                    }
                }

    def _parse_readme(self, content: str) -> Dict[str, Any]:
        """Parse YAML metadata from README.md."""
        import yaml
        
        # Extract YAML metadata
        metadata = yaml.safe_load(content.split("---\n")[1])
        return metadata

    def _convert_to_tflite(self, model_dir: str, output_path: str):
        """Convert PyTorch model to TFLite."""
        import torch
        import torch.quantization
        from torch.quantization import QuantStub, DeQuantStub
        
        # Load PyTorch model
        model = torch.load(os.path.join(model_dir, "model.pth"))
        
        # Quantize the model
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(model, inplace=True)
        torch.quantization.convert(model, inplace=True)
        
        # Save TFLite model
        torch.save(model.state_dict(), output_path)

if __name__ == "__main__":
    dataset = NeuroFluxDataset()
    ds = dataset.download_and_prepare()
    ds.save_to_disk("./neuroflux_dataset")
