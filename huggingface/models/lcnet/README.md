---
model_name: lcnet
model_type: lcnet
original_format: pytorch # Placeholder: LCNet is from PaddlePaddle, often converted via ONNX/PyTorch
pipeline_tag: image-classification
model_size: "N/A MB"
description: "LCNet (Lightweight CPU Network) model for image classification, designed for speed on CPU."
target_devices: ["cpu"]
performance:
  inference_time: "N/A ms"
  memory_usage: "N/A MB"
  accuracy: "N/A (e.g., Top-1 ImageNet)"
conversion_options:
  gpu_delegate: false # Optimized for CPU
  quantization_type: "post_training_int8_weights"
inference_solutions:
  mnn_compatible: false # Potentially true if converted from Paddle
  ncnn_compatible: false # Potentially true if converted from Paddle
  onnx_runtime_mobile_compatible: true # Assuming TFLite/ONNX conversion
hardware_specific_operators: []
# Model file (e.g., model.pdmodel, model.onnx, model.pth) to be added here.
# A dummy model.txt file will be created as a placeholder.
---

# LCNet

This document describes the LCNet model.
Further details about the model architecture, training, and use cases will be added here.
Model file to be added to this directory.
