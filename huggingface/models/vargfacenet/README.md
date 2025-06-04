---
model_name: vargfacenet
model_type: vargfacenet
original_format: pytorch # Placeholder
pipeline_tag: face-recognition
model_size: "N/A MB"
description: "VarGFaceNet model for lightweight face recognition."
target_devices: ["cpu", "gpu"]
performance:
  inference_time: "N/A ms"
  memory_usage: "N/A MB"
  accuracy: "N/A (e.g., LFW accuracy)"
conversion_options:
  gpu_delegate: true
  quantization_type: "post_training_int8_weights"
inference_solutions:
  mnn_compatible: false
  ncnn_compatible: false
  onnx_runtime_mobile_compatible: true # Assuming TFLite/ONNX conversion
hardware_specific_operators: []
# Model file (e.g., model.pth, model.onnx) to be added here.
# A dummy model.txt file will be created as a placeholder.
---

# VarGFaceNet

This document describes the VarGFaceNet model.
Further details about the model architecture, training, and use cases will be added here.
Model file to be added to this directory.
