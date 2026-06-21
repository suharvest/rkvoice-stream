#!/usr/bin/env python3
"""Write MOSS-specific HuggingFace custom code for the RKLLM scaffold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CONFIGURATION_CUSTOM = '''\
from transformers import PretrainedConfig


class CustomConfig(PretrainedConfig):
    model_type = "moss_rkllm_custom"

    def __init__(
        self,
        vocab_size=16384,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_key_value_heads=None,
        max_position_embeddings=2048,
        rms_norm_eps=1e-5,
        hidden_act="gelu",
        position_embedding_type="rope",
        rope_base=10000.0,
        moss_rkllm=None,
        use_cache=True,
        tie_word_embeddings=False,
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.hidden_act = hidden_act
        self.position_embedding_type = position_embedding_type
        self.rope_base = rope_base
        self.moss_rkllm = moss_rkllm or {}
        self.use_cache = use_cache
'''


MODELING_CUSTOM = '''\
import math
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from .configuration_custom import CustomConfig


class MossSelfAttention(nn.Module):
    def __init__(self, config: CustomConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        inv_freq = 1.0 / (
            float(getattr(config, "rope_base", 10000.0))
            ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.num_heads, self.head_dim)

    @staticmethod
    def _rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
        neg_odd = (-hidden_states[..., 1::2]).unsqueeze(-1)
        even = hidden_states[..., ::2].unsqueeze(-1)
        return torch.cat((neg_odd, even), dim=-1).reshape_as(hidden_states)

    def _apply_rope(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        freqs = torch.einsum("bt,d->btd", position_ids.to(hidden_states.device, torch.float32), self.inv_freq)
        emb = torch.repeat_interleave(freqs, 2, dim=-1).to(dtype=hidden_states.dtype)
        cos = emb.cos().unsqueeze(2)
        sin = emb.sin().unsqueeze(2)
        return (hidden_states * cos) + (self._rotate_half(hidden_states) * sin)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_valid_lengths: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ):
        bsz, seq_len, _ = hidden_states.shape
        q = self._shape(self.q_proj(hidden_states))
        k = self._shape(self.k_proj(hidden_states))
        v = self._shape(self.v_proj(hidden_states))
        q = self._apply_rope(q, position_ids)
        k = self._apply_rope(k, position_ids)
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k.to(device=k.device, dtype=k.dtype), k], dim=1)
            v = torch.cat([past_v.to(device=v.device, dtype=v.dtype), v], dim=1)
        present = (k, v) if use_cache else None
        q_states = q.permute(0, 2, 1, 3)
        k_states = k.permute(0, 2, 3, 1)
        v_states = v.permute(0, 2, 1, 3)
        scores = torch.matmul(q_states, k_states) / math.sqrt(self.head_dim)
        key_len = k.shape[1]
        key_positions = torch.arange(key_len, device=hidden_states.device, dtype=torch.long).view(1, 1, 1, key_len)
        query_positions = position_ids.to(device=hidden_states.device, dtype=torch.long).view(bsz, 1, seq_len, 1)
        causal_mask = key_positions <= query_positions
        if attention_mask is not None:
            mask = attention_mask.to(dtype=torch.bool, device=hidden_states.device)
            key_mask = mask.view(bsz, 1, 1, key_len)
        elif past_valid_lengths is not None:
            valid_key_lengths = past_valid_lengths.to(device=hidden_states.device, dtype=torch.long).view(bsz, 1, 1, 1) + seq_len
            key_mask = key_positions < valid_key_lengths
        else:
            key_mask = torch.ones((bsz, 1, 1, key_len), dtype=torch.bool, device=hidden_states.device)
        scores = scores.masked_fill(~(causal_mask & key_mask), torch.finfo(scores.dtype).min)
        probs = F.softmax(scores, dim=-1)
        out = torch.matmul(probs, v_states).transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        return self.o_proj(out), present


class MossMLP(nn.Module):
    def __init__(self, config: CustomConfig):
        super().__init__()
        self.fc_in = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc_out = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_out(F.gelu(self.fc_in(x), approximate="tanh"))


class MossDecoderLayer(nn.Module):
    def __init__(self, config: CustomConfig):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps, elementwise_affine=True)
        self.self_attn = MossSelfAttention(config)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps, elementwise_affine=True)
        self.mlp = MossMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_valid_lengths: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ):
        attn_output, present = self.self_attn(
            self.input_layernorm(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            past_valid_lengths=past_valid_lengths,
            use_cache=use_cache,
        )
        hidden_states = hidden_states + attn_output
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present


class CustomPreTrainedModel(PreTrainedModel):
    config_class = CustomConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["MossDecoderLayer"]

    def _init_weights(self, module):
        return


class CustomModel(CustomPreTrainedModel):
    def __init__(self, config: CustomConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        audio_sizes = config.moss_rkllm.get("audio_codebook_sizes") or [1024] * 16
        self.audio_embeddings = nn.ModuleList(
            [nn.Embedding(int(size), config.hidden_size) for size in audio_sizes]
        )
        self.layers = nn.ModuleList([MossDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps, elementwise_affine=True)
        self.post_init()

    def _embed_rows(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.dim() == 2:
            return self.embed_tokens(input_ids)
        if input_ids.dim() != 3:
            raise ValueError(f"expected input_ids rank 2 or 3, got {tuple(input_ids.shape)}")
        text = input_ids[..., 0]
        hidden = self.embed_tokens(text)
        audio_pad = int(self.config.moss_rkllm.get("audio_pad_token_id", 1024))
        for idx, emb in enumerate(self.audio_embeddings):
            codes = input_ids[..., idx + 1]
            mask = codes.ne(audio_pad)
            safe_codes = codes.clamp(min=0, max=emb.num_embeddings - 1)
            hidden = hidden + emb(safe_codes) * mask.unsqueeze(-1).to(hidden.dtype)
        return hidden

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        past_valid_lengths: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        return_dict = True if return_dict is None else return_dict
        use_cache = self.config.use_cache if use_cache is None else bool(use_cache)
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds is required")
            hidden_states = self._embed_rows(input_ids)
        else:
            hidden_states = inputs_embeds
        is_decode = past_key_values is not None
        if is_decode:
            if past_valid_lengths is None:
                past_valid_lengths = torch.tensor(
                    [past_key_values[0][0].shape[1]] * hidden_states.shape[0],
                    dtype=torch.long,
                    device=hidden_states.device,
                )
            else:
                past_valid_lengths = past_valid_lengths.to(device=hidden_states.device, dtype=torch.long)
            position_ids = past_valid_lengths.unsqueeze(1) + torch.arange(
                hidden_states.shape[1],
                dtype=torch.long,
                device=hidden_states.device,
            ).view(1, -1)
            layer_attention_mask = None
            query_mask = torch.ones(hidden_states.shape[:2] + (1,), dtype=hidden_states.dtype, device=hidden_states.device)
        else:
            if attention_mask is None:
                attention_mask = torch.ones(hidden_states.shape[:2], dtype=torch.long, device=hidden_states.device)
            attention_mask = attention_mask.to(device=hidden_states.device)
            position_ids = attention_mask.to(dtype=torch.long).cumsum(dim=-1) - 1
            position_ids = position_ids.masked_fill(~attention_mask.to(dtype=torch.bool), 0)
            layer_attention_mask = attention_mask
            query_mask = attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
        hidden_states = hidden_states * query_mask
        all_hidden = [] if output_hidden_states else None
        next_past = [] if use_cache else None
        for layer_index, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden.append(hidden_states)
            hidden_states, present = layer(
                hidden_states,
                attention_mask=layer_attention_mask,
                position_ids=position_ids,
                past_key_value=None if past_key_values is None else past_key_values[layer_index],
                past_valid_lengths=past_valid_lengths,
                use_cache=use_cache,
            )
            hidden_states = hidden_states * query_mask
            if next_past is not None:
                next_past.append(present)
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states * query_mask
        if output_hidden_states:
            all_hidden.append(hidden_states)
        next_past_tuple = tuple(next_past) if next_past is not None else None
        if not return_dict:
            return (hidden_states, next_past_tuple, tuple(all_hidden) if all_hidden is not None else None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_past_tuple,
            hidden_states=tuple(all_hidden) if all_hidden is not None else None,
        )

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value


class CustomForCausalLM(CustomPreTrainedModel):
    def __init__(self, config: CustomConfig):
        super().__init__(config)
        self.model = CustomModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        past_valid_lengths: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            past_valid_lengths=past_valid_lengths,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        logits = self.lm_head(outputs.last_hidden_state)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[..., :-1, :].reshape(-1, logits.size(-1)), labels[..., 1:].reshape(-1))
        if return_dict is False:
            return (logits, outputs.past_key_values, outputs.hidden_states)
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=None,
        )
'''


def write_custom_code(scaffold_dir: Path) -> dict:
    scaffold_dir = scaffold_dir.resolve()
    config_path = scaffold_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"missing config.json: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["auto_map"] = {
        "AutoConfig": "configuration_custom.CustomConfig",
        "AutoModel": "modeling_custom.CustomModel",
        "AutoModelForCausalLM": "modeling_custom.CustomForCausalLM",
    }
    config["architectures"] = ["CustomForCausalLM"]
    config["model_type"] = "moss_rkllm_custom"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (scaffold_dir / "configuration_custom.py").write_text(CONFIGURATION_CUSTOM, encoding="utf-8")
    (scaffold_dir / "modeling_custom.py").write_text(MODELING_CUSTOM, encoding="utf-8")
    report = {
        "scaffold_dir": str(scaffold_dir),
        "configuration_custom": str(scaffold_dir / "configuration_custom.py"),
        "modeling_custom": str(scaffold_dir / "modeling_custom.py"),
        "supports_row_width": config.get("moss_rkllm", {}).get("row_width"),
        "supports_audio_embeddings": len(config.get("moss_rkllm", {}).get("audio_codebook_sizes", [])),
        "ready_for_hf_load_probe": True,
        "ready_for_rkllm_export": False,
        "why_not_ready": [
            "hidden parity against ONNX prefill/decode has not been proven",
            "RKLLM toolkit export has not been run",
        ],
    }
    (scaffold_dir / "moss_rkllm_custom_code_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scaffold-dir", required=True, type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    report = write_custom_code(args.scaffold_dir)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
