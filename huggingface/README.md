---
dataset_name: NeuroFlux Models
dataset_description: Collection of lightweight AI models for microscopic applications
dataset_size: 18MB
dataset_format: pytorch
tags:
  - microcontroller
  - ai-models
  - lightweight
  - edge-computing
---

# NeuroFlux Models

This repository contains the trained models and configurations for the NeuroFlux framework.

## Available Models

- `neuroflux/tinybert`: A tiny version of BERT optimized for microscopic AI
- `neuroflux/mobilenet`: A lightweight MobileNet for computer vision tasks
- `neuroflux/emotion-detector`: A specialized model for emotion detection
- `neuroflux/efficientnet-lite`: EfficientNet-Lite model for image classification
- `neuroflux/nanodet-plus`: NanoDet-Plus model for object detection
- `neuroflux/fastbert`: FastBERT model for text classification
- `neuroflux/lcnet`: LCNet model for image classification
- `neuroflux/vargfacenet`: VarGFaceNet model for face recognition
- `neuroflux/pp-ocrv3-tiny`: PP-OCRv3 Tiny model for optical character recognition
- `neuroflux/mobilebert-tiny`: MobileBERT-Tiny model for question answering

## Usage

To use these models with the NeuroFlux framework:

```python
from neuroflux.models import NanoModel

model = NanoModel.from_pretrained("neuroflux/tinybert")
```

## Model Cards

Each model has its own model card with detailed information:

- [TinyBERT Model Card](models/tinybert/README.md)
- [MobileNet Model Card](models/mobilenet/README.md)
- [Emotion Detector Model Card](models/emotion-detector/README.md)
- [EfficientNet-Lite Model Card](models/efficientnet-lite/README.md)
- [NanoDet-Plus Model Card](models/nanodet-plus/README.md)
- [FastBERT Model Card](models/fastbert/README.md)
- [LCNet Model Card](models/lcnet/README.md)
- [VarGFaceNet Model Card](models/vargfacenet/README.md)
- [PP-OCRv3 Tiny Model Card](models/pp-ocrv3-tiny/README.md)
- [MobileBERT-Tiny Model Card](models/mobilebert-tiny/README.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to contribute.

## License

This project is under Apache 2.0 license. See [LICENSE](LICENSE) for details.
