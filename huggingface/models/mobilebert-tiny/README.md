---
model_name: mobilebert-tiny
model_type: mobilebert-tiny # Matches NeuroFluxConfig
original_format: tensorflow # MobileBERT is often from TensorFlow
pipeline_tag: question-answering # Or other NLP tasks
model_size: "N/A MB"
description: "MobileBERT-Tiny, an even smaller version of MobileBERT for NLP on the edge."
target_devices: ["cpu"]
performance:
  inference_time: "N/A ms"
  memory_usage: "N/A MB"
  accuracy: "N/A (e.g., SQuAD F1 score)"
conversion_options:
  gpu_delegate: false
  quantization_type: "post_training_int8_weights"
inference_solutions:
  mnn_compatible: false
  ncnn_compatible: false
  onnx_runtime_mobile_compatible: true # Assuming TFLite conversion
hardware_specific_operators: []
# Model file (e.g., model.pb, model.onnx) to be added here.
# A dummy model.txt file will be created as a placeholder.
---

# MobileBERT-Tiny

This document describes the MobileBERT-Tiny model.
Further details about the model architecture, training, and use cases will be added here.
Model file to be added to this directory.
