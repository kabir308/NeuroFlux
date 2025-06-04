---
model_name: pp-ocrv3-tiny
model_type: pp-ocrv3-tiny # Matches NeuroFluxConfig
original_format: paddle # PP-OCR is from PaddlePaddle
pipeline_tag: optical-character-recognition
model_size: "N/A MB"
description: "PP-OCRv3 Tiny model for lightweight optical character recognition."
target_devices: ["cpu"]
performance:
  inference_time: "N/A ms"
  memory_usage: "N/A MB"
  accuracy: "N/A (e.g., text recognition accuracy)"
conversion_options:
  gpu_delegate: false # OCR models often CPU bound for pre/post processing
  quantization_type: "post_training_int8_weights"
inference_solutions:
  mnn_compatible: false # Potentially true if converted from Paddle
  ncnn_compatible: false # Potentially true if converted from Paddle
  onnx_runtime_mobile_compatible: true # Assuming TFLite/ONNX conversion
hardware_specific_operators: []
# Model file (e.g., model.pdmodel) to be added here.
# A dummy model.txt file will be created as a placeholder.
---

# PP-OCRv3 Tiny

This document describes the PP-OCRv3 Tiny model.
It includes both detection and recognition models, optimized for edge deployment.
Further details about the model architecture, training, and use cases will be added here.
Model files to be added to this directory.
