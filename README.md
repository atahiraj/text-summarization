# Text Summarization

## About

This repository is an early experimental pipeline for abstractive text
summarization from article-title pairs. It contains preprocessing for paired
text files, TorchText dataset and iterator construction, pretrained GloVe
vocabulary loading, and an initial recurrent encoder model.

The project is an incomplete archival prototype. `train.py` currently builds
the vocabulary and batches and prints diagnostic samples; it does not run a
complete optimization loop. The code uses legacy TorchText and FastAI APIs,
and exact dependency versions were not recorded.

## Setup

Create and activate a Python environment, then install the imported
dependencies using mutually compatible legacy versions:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install pandas torch torchtext fastai spacy
```

The expected raw data consists of one article and one title per line:

```text
data/
├── raw_data/
│   ├── train/
│   │   ├── train.article.txt
│   │   └── train.title.txt
│   └── valid/
│       ├── valid.article.txt
│       └── valid.title.txt
└── processed_data/
```

Because the repository has no lockfile or environment specification, current
package releases may require changes to the legacy imports.

## Usage

Create sampled training and validation CSV files:

```sh
python preprocess.py \
  --source_articles data/raw_data/train/train.article.txt \
  --source_titles data/raw_data/train/train.title.txt \
  --destination data/processed_data/train_sample.csv \
  --sample 10000

python preprocess.py \
  --source_articles data/raw_data/valid/valid.article.txt \
  --source_titles data/raw_data/valid/valid.title.txt \
  --destination data/processed_data/valid_sample.csv \
  --sample 1000
```

Exercise the current vocabulary and batching pipeline with:

```sh
python train.py
```

`train.py` accepts `--gpu`, but its current iterator is fixed to the CPU and
the selected device is not connected to a training loop. Extending the encoder
in `model.py` and adding a decoder, loss, and optimization loop are still
required for end-to-end summarization training.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for the
complete terms.
