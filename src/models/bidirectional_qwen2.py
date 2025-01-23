from typing import List, Optional, Tuple, Union
import torch

from transformers import Qwen2Model, Qwen2ForCausalLM, Qwen2PreTrainedModel, Qwen2Config
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2DecoderLayer,
    Qwen2RMSNorm,
    Qwen2Attention,
    Qwen2FlashAttention2,
    Qwen2SdpaAttention,
    Qwen2MLP,
)
from torch import nn
from transformers.utils import logging
from .attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)

from peft import PeftModel

logger = logging.get_logger(__name__)



# 定制注意力模块，所有模型都相同，继承自基础模型的注意力类，将 self.is_causal 设置为 False，以支持双向注意力，而非传统的自回归模型的单向注意力机制。这些类被用于双向解码器层中
class ModifiedQwen2Attention(Qwen2Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False

class ModifiedQwen2FlashAttention2(Qwen2FlashAttention2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False

class ModifiedQwen2SdpaAttention(Qwen2SdpaAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False

QWEN2_ATTENTION_CLASSES = {
    "eager": ModifiedQwen2Attention,
    "flash_attention_2": ModifiedQwen2FlashAttention2,
    "sdpa": ModifiedQwen2SdpaAttention,
}



# 定制解码器层，用于在后面 Qwen2BiModel 中进行双向编码，所有模型都相同
class ModifiedQwen2DecoderLayer(Qwen2DecoderLayer):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        nn.Module.__init__(self)
        # 隐藏层维度
        self.hidden_size = config.hidden_size

        # 自注意力层，_attn_implementation 指定选择哪种注意力机制
        self.self_attn = QWEN2_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        # 多层感知机层，将自注意力的输出通过非线性变换，以增加模型的表达能力
        self.mlp = Qwen2MLP(config)
        # 输入层归一化层，标准化自注意力层输入的特征，config.hidden_size 作为层大小，config.rms_norm_eps 作为归一化的小数精度（epsilon），避免数值不稳定
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 后注意力层的归一化层，标准化自注意力输出的特征，以稳定模型的训练过程
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)



class Qwen2BiModel(Qwen2Model):
    _no_split_modules = ["ModifiedQwen2DecoderLayer"]

    def __init__(self, config: Qwen2Config):
        Qwen2PreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                ModifiedQwen2DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()



# 创建用于条件语言模型任务的双向模型，所有模型都相同
class Qwen2BiForMNTP(Qwen2ForCausalLM):
    def __init__(self, config):
        Qwen2PreTrainedModel.__init__(self, config)
        self.model = Qwen2BiModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    # getter for PEFT model
    def get_model_for_peft(self):
        return self.model

    # setter for PEFT model
    def set_model_for_peft(self, model: PeftModel):
        self.model = model

    # save the PEFT model
    def save_peft_model(self, path):
        self.model.save_pretrained(path)
