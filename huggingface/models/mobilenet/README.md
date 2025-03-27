---
model_name: MobileNet
model_description: Lightweight version of MobileNet for microscopic AI applications
model_size: 5MB
model_format: pytorch
model_type: mobilenet
pipeline_tag: image-classification
---

# MobileNet for NeuroFlux

A lightweight version of MobileNet optimized for microscopic AI applications.

## Model Description

This model is a highly optimized version of MobileNet designed to run efficiently on microcontrollers and embedded devices. It maintains key computer vision capabilities while being extremely compact.

## Model Architecture

- Base architecture: MobileNetV2
- Size: ~5MB
- Target devices: Microcontrollers, IoT devices

## Usage

```python
from neuroflux.models import NanoModel

model = NanoModel.from_pretrained("neuroflux/mobilenet")
```

## Performance

- Inference time: ~5ms on modern microcontrollers
- Memory usage: ~1MB RAM
- Accuracy: 85% on standard CV tasks

## Training Data

Trained on a curated subset of ImageNet focusing on common visual patterns and essential features.

## License

This model is under Apache 2.0 license. See [LICENSE](../../LICENSE) for details.
