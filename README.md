[![LICENSE](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/sebasmos/quantaudio/blob/main/LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://github.com/sebasmos/quantaudio)

# QuantAudio: Optimized Pre-Trained Vector Embeddings for Resource-Efficient Audio Classification

> 🚧 This repository is under development.
>
> 📩 **Code and models will be made publicly available upon preprint upload or journal submission.**

- 📂 **GitHub Repository**: [quantaudio](https://github.com/sebasmos/quantaudio)

## Project Structure

- 📁 `data/` – Links and scripts to download UrbanSound8K, ESC-50, and other datasets
- 📁 `src/`
  - `preprocessing/` – Audio loading, Mel spectrogram generation
  - `models/` – Embedding extractor, MLP classifier
  - `quantization/` – Post-training quantization scripts
  - `evaluation/` – Metrics and logging tools
- 📁 `experiments/` – Configs and logs for reproducible experiments
- 📁 `notebooks/` – Visualizations and exploratory analyses
- 📁 `scripts/` – End-to-end training, testing, and quantization pipelines

## Setting Up Your Environment

1. **Create a Conda Environment:**
   ```bash
   conda create -n quantaudio python=3.11 -y
   conda activate quantaudio
   ```

2. **Install Dependencies:**
   ```bash
   git clone https://github.com/sebasmos/quantaudio.git
   cd quantaudio
   pip install -r requirements.txt
   ```

## Running the Pipeline

To train and evaluate the quantized MLP classifier:
```bash
python scripts/train.py --config configs/urban8k_base.yaml
```

To apply post-training quantization:
```bash
python scripts/quantize.py --model-checkpoint checkpoints/best_model.pth
```

## Contributing to QuantAudio

We welcome community contributions! Fork the [QuantAudio repository](https://github.com/sebasmos/QuantAudio), make your improvements, and open a pull request. Contributors will be acknowledged in the release.

Feel free to report bugs, suggest features, or share your use cases.

## License

QuantAudio is **free** and **open source**, released under the [MIT License](https://github.com/sebasmos/QuantAudio/blob/main/LICENSE).

## Citation

```bibtex
@software{Cajas2025_QuantAudio,
  author = {Cajas Ord\'o\~nez, Sebasti\'an Andr\'es and Torres Torres, Luis Fernando and Bosch, Cristian and Lai, Yuan and Duran Paredes, Carlos Andr\'es and Celi, Leo Anthony and Simon Carbajo, Ricardo},
  title = {QuantAudio: Optimized Pre-Trained Vector Embeddings for Resource-Efficient Audio Classification},
  year = {2025},
  url = {https://github.com/sebasmos/QuantAudio},
  license = {MIT}
}
```
