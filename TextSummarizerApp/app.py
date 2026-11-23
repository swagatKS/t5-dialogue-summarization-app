from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re
from fastapi.templating import Jinja2Templates # UI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# initialize our fastapi app
app = FastAPI(title="Text Summarizer App", description="Text Summarization using T5", version="1.0")

# model & tokenizer
model = T5ForConditionalGeneration.from_pretrained("../saved_summary_model")
tokenizer = T5Tokenizer.from_pretrained("../saved_summary_model")

# device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)

# templating
templates = Jinja2Templates(directory=".")


# Input schema for dialogue => string
class DialogueInput(BaseModel):
    dialogue: str


def clean_data(text):
    text = re.sub(r"<.*?>", " ", text) # html tags <p> <h1>
    text = re.sub(r"\s+", " ", text)     # collapse spaces/newlines
    text = text.strip()
    return text


def summarize_dialogue(dialogue: str) -> str:
    dialogue = "summarize: " + clean_data(dialogue) # clean

    # tokenize
    inputs = tokenizer(
        dialogue,
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    # generate the summary => token ids
    targets = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=128,
        num_beams=4,
        early_stopping=True
    )

    # token ids convert to summary => decoding
    summary = tokenizer.decode(targets[0], skip_special_tokens=True)
    return summary


# API endpoints
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/summarize")
async def summarize(dialogue: DialogueInput):
    summary = summarize_dialogue(dialogue.dialogue)
    return {"summary": summary}