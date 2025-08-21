# Local Command-Line Chatbot (Hugging Face)

A tiny local chatbot you run in the terminal using the Hugging Face `transformers` pipeline.
It keeps a short conversation memory (sliding window) and exits with `/exit`.

---

## 1) Quick Start

### A) Open the project in Cursor
1. Download the ZIP and unzip it somewhere easy (e.g., Desktop).
2. In Cursor, click **File → Open...** and select the unzipped folder `hf_cli_chatbot`.

### B) Create a virtual environment (Windows PowerShell)

```powershell
py -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

**macOS / Linux (bash/zsh):**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> The first run will download a small model. It may take a minute.

### C) Run the chatbot
```bash
python interface.py
```
Then type messages. Use `/exit` to quit, `/reset` to clear memory, `/help` for help.

---

## 2) Change the model (optional)
By default this uses `google/flan-t5-small` (good and lightweight). You can switch with a flag:

```bash
python interface.py --model distilgpt2
```
Other small options to try: `distilgpt2` (casual LM), `google/flan-t5-base` (slightly bigger).

> Tip: Very large models may be slow on CPU. GPU is optional; pass `--use-gpu` if you have one.

---

## 3) Sample run

```
$ python interface.py
👋 Hello! I'm your local CLI chatbot.
Type anything to chat. Commands: /exit, /reset, /help

You: What is the capital of France?
Bot: The capital of France is Paris.

You: And what about Italy?
Bot: The capital of Italy is Rome.

You: /exit
Exiting chatbot. Goodbye!
```

---

## 4) Files you will see

- `model_loader.py` — loads a Hugging Face model & pipeline.
- `chat_memory.py` — small sliding-window chat memory helper.
- `interface.py` — the CLI loop that ties everything together.

---

## 5) Notes

- On the first run, the model files will download automatically to your Hugging Face cache.
- If you ever get a long or messy answer, try `--temperature 0.7 --top-p 0.9 --max-new-tokens 128`
- If the model refuses to follow instructions well, try `google/flan-t5-base` or another small instruct model.

Enjoy! 😊