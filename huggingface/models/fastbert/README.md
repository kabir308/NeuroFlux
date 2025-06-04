---
model_name: fastbert
model_type: fastbert
original_format: pytorch # Placeholder: BERT variants are commonly PyTorch
pipeline_tag: text-classification # Or question-answering, etc.
model_size: "N/A MB"
description: "FastBERT model for efficient NLP tasks on edge devices."
target_devices: ["cpu"]
performance:
  inference_time: "N/A ms"
  memory_usage: "N/A MB"
  accuracy: "N/A (e.g., GLUE score)"
conversion_options:
  gpu_delegate: false # Typically BERT-like models run better on CPU for edge
  quantization_type: "post_training_int8_weights"
inference_solutions:
  mnn_compatible: false
  ncnn_compatible: false
  onnx_runtime_mobile_compatible: true # Assuming TFLite/ONNX conversion
hardware_specific_operators: []
# Model file (e.g., model.pth, model.onnx) to be added here.
# A dummy model.txt file will be created as a placeholder.
---

# FastBERT

This document describes the FastBERT model.
Further details about the model architecture, training, and use cases will be added here.
Model file to be added to this directory.
