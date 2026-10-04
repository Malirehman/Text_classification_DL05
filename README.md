# BERT-Based Sarcasm Detection

A text-classification project that detects sarcasm in news headlines using a **frozen BERT encoder** (`google-bert/bert-base-uncased`) and a small neural-network classification head implemented in PyTorch.

## Project overview

The notebook follows this pipeline:

```text
Kaggle dataset
      |
      v
Data cleaning
      |
      v
70 / 15 / 15 train-validation-test split
      |
      v
BERT tokenizer
      |
      v
Frozen BERT encoder
      |
      v
[CLS] 768-d representation
      |
      v
Linear(768 -> 384) + Dropout + Linear(384 -> 1)
      |
      v
Sigmoid probability
      |
      v
BCELoss + Adam training
```

## Dataset

The notebook downloads the **News Headlines Dataset for Sarcasm Detection** from Kaggle and reads `Sarcasm_Headlines_Dataset.json` as JSON Lines.

Source used in the notebook:
`https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection`

The dataset is cleaned with:

- `dropna()`
- `drop_duplicates()`
- removal of `article_link`

After cleaning, the notebook reports **26,708 rows and 2 columns** (`headline`, `is_sarcastic`).

## Model

### BERT encoder

The notebook uses:

```python
AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
AutoModel.from_pretrained("google-bert/bert-base-uncased")
```

The BERT parameters are frozen:

```python
for param in bert_model.parameters():
    param.requires_grad = False
```

The model takes the first-token / `[CLS]` representation:

```python
pooled_output = self.bert(
    input_ids=input_ids,
    attention_mask=attention_mask,
    return_dict=False
)[0][:, 0, :]
```

### Classification head

```text
768 -> 384 -> 1
```

with `Dropout(0.25)` between the linear layers and a sigmoid output for binary classification.

## Training configuration

| Setting | Value |
|---|---:|
| Batch size | 32 |
| Epochs | 10 |
| Learning rate | 1e-4 |
| Loss | Binary Cross Entropy (`BCELoss`) |
| Optimizer | Adam |
| BERT weights | Frozen |
| Max token length | 100 |
| Train / validation / test | 70% / 15% / 15% |

## Results from the notebook run

The recorded validation accuracy reached **87.47%** at epoch 10. Training accuracy at epoch 10 was **83.83%**.

> **Important evaluation note:** the final test section calculates `total_acc_test`, but the printed line uses `total_acc_train`. Therefore, the displayed `83.8299%` is **training accuracy**, not the true test accuracy. This should be fixed before reporting a final test score.

The notebook also stores training and validation loss/accuracy history and plots them across epochs.

## How to run

### Google Colab (recommended for the notebook)

The original notebook was executed with CUDA available. For a laptop without a suitable GPU, BERT inference/training can be much slower.

1. Open `text_classification_BERT_Sarcasm.ipynb` in Google Colab.
2. Enable a GPU runtime when available.
3. Install the dependencies.
4. Run the Kaggle download cell and provide Kaggle credentials when prompted.
5. Run the remaining cells in order.

### Local environment

```bash
pip install -r requirements.txt
```

You will also need access to the Kaggle dataset referenced above and enough memory/compute for BERT.

## Repository files

```text
.
├── text_classification_BERT_Sarcasm.ipynb
├── text_classification_BERT_Sarcasm_original.ipynb
├── README.md
├── requirements.txt
├── assets/
│   └── bert_sarcasm_pipeline.png
└── docs/
    └── BERT_Sarcasm_Detection_Code_Explanation.pdf
```

## Code notes / improvement opportunities

This repository documents the notebook as it currently exists; it does not hide its implementation details. Before using the project as a polished portfolio piece, consider:

1. Fix the final test accuracy print statement to use `total_acc_test` and `testing_data.__len__()`.
2. Report loss as the mean across batches instead of dividing by a hard-coded `1000`.
3. Use `shuffle=False` for validation and test dataloaders for cleaner evaluation practice (shuffling does not change aggregate accuracy here, but is unnecessary).
4. Add precision, recall, F1-score, and a confusion matrix.
5. Set a random seed for reproducible train/validation/test splits.
6. Save the trained classifier weights and tokenizer configuration if you want to reuse the model outside the notebook.
7. Add a small inference script that accepts a headline and returns `sarcastic` / `not sarcastic` with a probability.

## Learning value

This project demonstrates several practical ML-engineering ideas in one notebook:

- dataset ingestion and cleanup
- train/validation/test splitting
- tokenization and attention masks
- transfer learning with a frozen pretrained model
- custom PyTorch `Dataset` and `DataLoader`
- model architecture design
- loss functions and optimization
- validation during training
- metric tracking and visualization
- identifying an evaluation/reporting bug

## License

Add the license that matches how you want to publish the code and the dataset terms. The dataset itself is hosted on Kaggle, so review its dataset/license page before redistributing the raw data.
