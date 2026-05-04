import logging
import json
import os
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from modules.remapping import DOBI_dequantize
from modules.module import *
from utils.modeling import get_no_svd_layers, should_decompose_linear


def _load_hf_state_dict(model_id):
    safetensors_path = os.path.join(model_id, "model.safetensors")
    bin_path = os.path.join(model_id, "pytorch_model.bin")
    safetensors_index = os.path.join(model_id, "model.safetensors.index.json")
    bin_index = os.path.join(model_id, "pytorch_model.bin.index.json")

    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        return load_file(safetensors_path, device="cpu")
    if os.path.exists(bin_path):
        return torch.load(bin_path, map_location="cpu")

    state_dict = {}
    if os.path.exists(safetensors_index):
        from safetensors.torch import load_file
        with open(safetensors_index, "r") as handle:
            index = json.load(handle)
        for shard in sorted(set(index["weight_map"].values())):
            state_dict.update(load_file(os.path.join(model_id, shard), device="cpu"))
        return state_dict
    if os.path.exists(bin_index):
        with open(bin_index, "r") as handle:
            index = json.load(handle)
        for shard in sorted(set(index["weight_map"].values())):
            state_dict.update(torch.load(os.path.join(model_id, shard), map_location="cpu"))
        return state_dict

    raise FileNotFoundError(f"No HuggingFace model weights found in {model_id}")


def load_remapping_model(updated_model_path, trust_remote_code=False):
    logging.getLogger("transformers").setLevel(logging.ERROR)
        
    model_id = updated_model_path
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=trust_remote_code)
    state_dict = _load_hf_state_dict(model_id)
    model.load_state_dict(state_dict, strict=False)
    model.to(torch.float16) 
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)

    mapping_info = torch.load(f"{model_id}/remapping_weight.pt", map_location="cpu")
    dobi_svd_config = getattr(config, "dobi_svd", {}) or {}
    no_svd_layers = dobi_svd_config.get("no_svd_layers", get_no_svd_layers(model_id))
    for name, module in tqdm(model.named_modules(), desc="Dequantize the model after remaping."):
        if should_decompose_linear(name, module, no_svd_layers):
            us_quan = mapping_info[name]["us_quan"]
            vt_quan = mapping_info[name]["vt_quan"]
            us_absmax = mapping_info[name]["us_absmax"]
            vt_absmax = mapping_info[name]["vt_absmax"]
            tuple_info = mapping_info[name]["tuple_info"]
            dequan_us, dequan_vt = DOBI_dequantize(us_quan, vt_quan, us_absmax, vt_absmax, tuple_info, code = None)

            compress_size = dequan_vt.size(0)* dequan_vt.size(1) + dequan_us.size(0)*dequan_us.size(1)
            ori_size = module.in_features * module.out_features
            if ori_size> compress_size:
                parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                attr_name = name.rsplit('.', 1)[-1]
                if parent_name != '':
                    parent = dict(model.named_modules())[parent_name]
                else:
                    parent = model
                NewLayer = SVDTransformLayer_remapping(weight1 = dequan_vt.T, weight2 = dequan_us.T,
                                             bias = module.bias, name = name, device = "cpu")
                setattr(parent, attr_name, NewLayer)
                del module
            else:
                new_weight = dequan_us @ dequan_vt
                module.weight.data = new_weight.detach()
                
            mapping_info[name] = {}
            
    return model, tokenizer 



def load_unremapping_model(model_id, trust_remote_code=False):
    legacy_path = f"{model_id}/DobiSVD_Model.pt"
    if os.path.exists(legacy_path):
        pruned_dict = torch.load(legacy_path) #, map_location='cuda'
        tokenizer, model = pruned_dict['tokenizer'], pruned_dict['model']
        return model, tokenizer

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    dobi_svd_config = getattr(config, "dobi_svd", {}) or {}
    base_model_id = dobi_svd_config.get("base_model_id", model_id)
    no_svd_layers = dobi_svd_config.get("no_svd_layers", get_no_svd_layers(base_model_id))
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=trust_remote_code)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    state_dict = _load_hf_state_dict(model_id)

    module_dict = dict(model.named_modules())
    for name, module in tqdm(list(model.named_modules()), desc="Rebuild Dobi-SVD layers"):
        if not should_decompose_linear(name, module, no_svd_layers):
            continue
        a_key = f"{name}.ALinear.weight"
        b_key = f"{name}.BLinear.weight"
        if a_key not in state_dict or b_key not in state_dict:
            continue
        parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
        attr_name = name.rsplit('.', 1)[-1]
        parent = module_dict[parent_name] if parent_name else model
        bias = state_dict.get(f"{name}.BLinear.bias")
        new_layer = SVDTransformLayer_remapping(
            weight1=state_dict[a_key].T,
            weight2=state_dict[b_key].T,
            bias=bias,
            name=name,
            device="cpu",
        )
        setattr(parent, attr_name, new_layer)

    model.load_state_dict(state_dict, strict=False)
    model.to(torch.float16)
    return model, tokenizer
