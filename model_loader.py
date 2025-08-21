from typing import Optional
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    pipeline,
)
import torch


def load_pipeline(
    model_name: str = "google/flan-t5-large",
    use_gpu: bool = False,
    torch_dtype: Optional[str] = None,
):
    """Create and return a Hugging Face generation pipeline.
    
    - Automatically detects whether the model is encoder-decoder (e.g., T5)
      or causal LM (e.g., GPT-2) and builds the correct pipeline.
    - Sets a reasonable pad token if missing.
    """
    config = AutoConfig.from_pretrained(model_name)
    is_encoder_decoder = getattr(config, "is_encoder_decoder", False)

    # Choose task + model class
    if is_encoder_decoder:
        task = "text2text-generation"
        model_cls = AutoModelForSeq2SeqLM
    else:
        task = "text-generation"
        model_cls = AutoModelForCausalLM

    # Set dtype if provided (e.g., 'float16' on GPU)
    dtype = None
    if torch_dtype:
        torch_dtype = torch_dtype.lower().strip()
        if torch_dtype == "float16":
            dtype = torch.float16
        elif torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        elif torch_dtype == "float32":
            dtype = torch.float32

    # Device map
    device_map = "auto" if use_gpu else None

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = model_cls.from_pretrained(model_name, torch_dtype=dtype, device_map=device_map)

    # Ensure we have a pad token
    if tokenizer.pad_token is None:
        # Best effort: fall back to EOS if available
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # create a new pad token
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
            model.resize_token_embeddings(len(tokenizer))

    gen = pipeline(task=task, model=model, tokenizer=tokenizer, device=0 if use_gpu else -1)
    return gen, task, tokenizer