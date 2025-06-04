---
model_name: efficientnet-lite
model_type: efficientnet-lite
original_format: tensorflow # Placeholder: EfficientNets are often from TensorFlow
pipeline_tag: image-classification
model_size: "N/A MB"
description: "EfficientNet-Lite model for image classification, optimized for edge devices."
target_devices: ["cpu", "gpu", "npu"]
performance:
  inference_time: "N/A ms"
  memory_usage: "N/A MB"
  accuracy: "N/A (e.g., Top-1 ImageNet)"
conversion_options:
  gpu_delegate: true
  quantization_type: "hybrid_fp16_int8"
inference_solutions:
  mnn_compatible: false
  ncnn_compatible: false
  onnx_runtime_mobile_compatible: true # Assuming TFLite conversion will make it compatible
hardware_specific_operators: []
# Model file (e.g., model.pb, model.onnx, model.pth) to be added here.
# A dummy model.txt file will be created as a placeholder.
---

# EfficientNet-Lite

This document describes the EfficientNet-Lite model.
Further details about the model architecture, training, and use cases will be added here.
Model file to be added to this directory.
