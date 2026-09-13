# Fine-Tune a Support Ticket Router

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vaish222/Fine-Tune-a-Support-Ticket-Router/blob/main/Finetune_Support_Ticket_Classifier_Qwen3.ipynb)

Fine-tune `Qwen/Qwen3-1.7B-Base` with LoRA to classify IT support tickets into the correct service queue. The included notebook provides an end-to-end Google Colab workflow using [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory): prepare the data, train through the LLaMA Board UI, merge the adapter, run inference, and compare the result with the base model.

## Routing categories

The classifier returns exactly one of seven labels:

| Label | Typical requests |
| --- | --- |
| `Active Directory` | Accounts, identity, access, and login workflows |
| `Fileservice` | Shared folders, file shares, and network-drive permissions |
| `O365` | Outlook, Teams, OneDrive, Skype, and mailbox issues |
| `EOL` | Server retirement and decommissioning tasks |
| `Software` | Application installation, updates, and application access |
| `Computer-Services` | Printers, scanners, drivers, and device support |
| `Support general` | Requests that do not clearly belong to another queue |

## Project contents

```text
.
├── Finetune_Support_Ticket_Classifier_Qwen3.ipynb  # Training and evaluation workflow
├── support_tickets.csv                             # Example labeled dataset
└── README.md
```

The dataset contains 585 tickets with these columns:

| Column | Description |
| --- | --- |
| `category_truth` | Ground-truth routing category |
| `text` | Support ticket text |

The notebook renames `category_truth` to `label`, filters unsupported categories, shuffles the records, and creates a reproducible stratified 80/20 training-validation split.

## Requirements

- A Google account with access to Google Colab
- A GPU-enabled Colab runtime; a free NVIDIA T4 is sufficient
- Internet access from Colab to clone LLaMA-Factory and download the Qwen model
- The included `support_tickets.csv`, or a compatible CSV using the schema above

No local installation is required for the intended workflow. The notebook installs LLaMA-Factory and its PyTorch and bitsandbytes dependencies inside the Colab runtime.

## Quick start

1. Click **Open in Colab** above, or upload `Finetune_Support_Ticket_Classifier_Qwen3.ipynb` to Colab.
2. In Colab, choose **Runtime → Change runtime type → T4 GPU**.
3. Run the dependency-installation and GPU-check cells.
4. Run the dataset-preparation cell and upload `support_tickets.csv` when prompted.
5. Run the LLaMA Board cell and open the public Gradio URL printed in its output.
6. In the **Train** tab, select:
   - Model: `Qwen/Qwen3-1.7B-Base`
   - Dataset: `support_tickets`
   - Fine-tuning method: `LoRA`
   - Compute type: `fp16` for a T4 GPU
7. Start training. When it finishes, copy the output directory shown by LLaMA Board, then manually stop the notebook cell running the UI server.
8. Set `ADAPTER_DIR` in the loss-curve cell to that output directory. For example:

   ```python
   ADAPTER_DIR = "/content/LLaMA-Factory/saves/Qwen3-1.7B-Base/lora/train_YYYY-MM-DD-HH-MM-SS"
   ```

9. Run the remaining cells in order to inspect training loss, evaluate the untouched base model, merge the LoRA adapter, smoke-test the classifier, and evaluate the fine-tuned model.

Training commonly takes around 30–60 minutes on a T4, depending on the selected hyperparameters and runtime conditions.

## Workflow

```text
support_tickets.csv
        │
        ├── 80% training split → ShareGPT JSON → LoRA fine-tuning
        │                                          │
        │                                          ▼
        │                              merged Qwen3 checkpoint
        │                                          │
        └── 20% held-out validation ────────────────┤
                                                   ▼
                                  reports, plots, and predictions
```

The notebook creates the following runtime artifacts:

- `/content/LLaMA-Factory/data/TRAIN.json` — training records in ShareGPT format
- `/content/val_split.csv` — held-out validation records
- `/content/training_curve.png` — training loss plot
- `/content/qwen3_merged` — standalone model with the LoRA weights merged
- `/content/confusion_matrix.png` — fine-tuned model confusion matrix
- `/content/baseline_vs_finetuned.png` — per-class F1 and overall accuracy comparison

Colab storage is temporary. Download any trained model or generated report that you want to keep before the runtime is disconnected.

## Dataset format

To use a different dataset, provide a CSV with the same two columns and one of the seven supported labels:

```csv
category_truth,text
Active Directory,"Please create an account for our new employee."
Fileservice,"I get Access Denied when opening the shared drive."
O365,"Outlook keeps asking me to sign in."
```

Each category needs enough examples for a stratified split. If you add or rename categories, update `LABEL2ID`, `SYSTEM_PROMPT`, `LABEL_TOKENS`, and the baseline choice mapping in the notebook so training and evaluation remain consistent.

## Inference

After the adapter is merged, the notebook exposes:

```python
label, confidence = classify(
    "Outlook will not sync my mailbox and Teams keeps signing me out."
)

print(label)       # O365
print(confidence)  # Relative score among the seven label-start tokens
```

`classify()` uses deterministic generation and falls back to `Support general` if the generated text does not begin with a known label. The reported confidence is a relative probability over the first token of each candidate label; it is useful for comparison and triage, but it is not a calibrated probability of correctness.

## Evaluation

The notebook evaluates the router only on the held-out 20% split and produces:

- Precision, recall, and F1 score for every routing category
- Overall accuracy
- A confusion matrix to reveal commonly confused queues
- A base-model versus fine-tuned comparison
- Example predictions and confidence scores

For production use, also test on a separate, recent dataset that reflects real ticket traffic. Pay particular attention to recall for high-impact queues, ambiguous or multi-intent tickets, out-of-scope requests, and confidence thresholds used for human review.

## Troubleshooting

- **CUDA is unavailable:** confirm that the Colab runtime uses a GPU, then restart the runtime and rerun the notebook.
- **`support_tickets` is missing in LLaMA Board:** rerun the dataset-preparation cell; it creates `TRAIN.json` and registers the dataset in `dataset_info.json`.
- **Adapter directory is not found:** copy the exact output directory from the completed LLaMA Board training run into `ADAPTER_DIR`.
- **Out-of-memory error on a T4:** keep `fp16`, reduce the per-device batch size, reduce the cutoff length, or increase gradient accumulation to preserve the effective batch size.
- **Training loss stays flat or unstable:** verify the dataset and prompt format, then try a lower learning rate or more epochs.
- **Gradio remains active after training:** stop the LLaMA Board cell manually; it runs a server and does not exit when training completes.

## Built with

- [Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base)
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [Transformers](https://github.com/huggingface/transformers)
- [PEFT](https://github.com/huggingface/peft)
- [scikit-learn](https://scikit-learn.org/)

## Notes

- The notebook downloads model weights and dependencies from third-party services whose terms and access requirements may change.
- Review and sanitize support-ticket data before uploading it to a hosted notebook, especially if it may contain personal, confidential, or security-sensitive information.
- Model outputs should be treated as routing suggestions until performance and fallback behavior have been validated for the target environment.
