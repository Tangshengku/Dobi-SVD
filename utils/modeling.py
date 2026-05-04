import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_no_svd_layers(model_id_or_name):
    """Return substrings for linear layers that should stay dense."""
    lower_name = model_id_or_name.split("/")[-1].lower()
    if any(model_type in lower_name for model_type in ("llama", "qwen", "mistral")):
        return ["lm_head"]
    if "opt" in lower_name:
        return ["project_out", "project_in"]
    return ["lm_head"]


def should_decompose_linear(name, module, no_svd_layers):
    return isinstance(module, nn.Linear) and all(x not in name for x in no_svd_layers)


def load_causal_lm_and_tokenizer(model_id, torch_dtype, trust_remote_code=False):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
    )
    return model, tokenizer
