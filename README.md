# Dialogue Text Summarizer

Fine-tunes [`google/flan-t5-base`](https://huggingface.co/google/flan-t5-base) on the
[SAMSum](https://huggingface.co/datasets/knkarthick/samsum) dialogue-summarization dataset and serves it
through a small FastAPI web app.

## Project structure

```
text_summarizer.ipynb      # data loading, fine-tuning and saving the model
TextSummarizerApp/
  app.py                   # FastAPI backend (GET / for the UI, POST /summarize)
  index.html               # simple web UI
  requirements.txt
```

## Getting the data and model

The dataset CSVs and the fine-tuned weights (~1 GB) are **not** in this repo because of GitHub's file size
limit.

1. **Data** – download SAMSum (`samsum-train.csv`, `samsum-validation.csv`, `samsum-test.csv`) from the link
   above and place the files next to the notebook.
2. **Model** – run `text_summarizer.ipynb` to fine-tune and save the model to `saved_summary_model/`, then
   copy that folder into `TextSummarizerApp/`.

## Run the app

```bash
cd TextSummarizerApp
pip install -r requirements.txt
uvicorn app:app --reload
```

Open http://127.0.0.1:8000, paste a dialogue, and get a summary.

API example:

```bash
curl -X POST http://127.0.0.1:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{"dialogue": "Anna: Are we meeting at 5? Tom: Yes, at the cafe."}'
```

## Tech stack

PyTorch, Hugging Face Transformers (T5), FastAPI, Uvicorn.
