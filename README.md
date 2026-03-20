# Power Encoding: 20x Neural Network Compression

Compress neural networks by 20x using mathematical power transformations.

## Results

| Model | Parameters | Accuracy | Compression |
|-------|-----------|----------|-------------|
| Traditional | 109,386 | 96.97% | 1x |
| **Power Encoding** | **5,587** | **78.88%** | **19.6x** |

## Quick Start

pip install torch torchvision
python train_improved.py

## How It Works

Store small seed matrix + Apply power transformations = Generate full weights

## License

MIT License
