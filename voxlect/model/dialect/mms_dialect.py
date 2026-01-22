import torch
import loralib as lora
import transformers.models.wav2vec2.modeling_wav2vec2 as w2v2

from torch import nn
from torch.nn import functional as F
from transformers import Wav2Vec2Model
from transformers import AutoFeatureExtractor
from huggingface_hub import PyTorchModelHubMixin

class Wav2Vec2EncoderLayer(nn.Module):
    def __init__(
        self, 
        config, 
        i
    ):
        super().__init__()
        self.attention = w2v2.Wav2Vec2Attention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
        )
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = w2v2.Wav2Vec2FeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.config = config
        
        if self.config.finetune_method == "embedding_prompt" or self.config.finetune_method == "combined":
            self.embed_prompt = nn.Parameter(torch.randn([1, self.config.embedding_prompt_dim, 768]))
            nn.init.xavier_uniform_(self.embed_prompt)
        if self.config.finetune_method == "lora" or self.config.finetune_method == "combined":
            self.feed_forward.intermediate_dense    = lora.Linear(config.hidden_size, config.intermediate_size, r=config.lora_rank)
            self.feed_forward.output_dense          = lora.Linear(config.intermediate_size, config.hidden_size, r=config.lora_rank)
            
        if self.config.finetune_method == "adapter" or self.config.finetune_method == "adapter_l" or self.config.finetune_method == "combined":
            self.adapter = Adapter(
                config, 
                dropout=0.1, 
                bottleneck=config.adapter_hidden_dim, 
                adapter_scalar=0.1
            )
        self.i = i

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        if self.config.finetune_method == "embedding_prompt" or self.config.finetune_method == "combined":
            hidden_states = torch.cat((self.embed_prompt.repeat(hidden_states.size(0), 1, 1), hidden_states), dim=1)
        attn_residual = hidden_states
        
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states

        # Adapter
        if self.config.finetune_method == "adapter":
            adapt_h = self.adapter(hidden_states)

        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states + self.feed_forward(hidden_states) 
        # Adapter
        if self.config.finetune_method == "adapter": 
            hidden_states = hidden_states+ adapt_h
        if self.config.finetune_method == "adapter_l" or self.config.finetune_method == "combined": 
            hidden_states = hidden_states + self.adapter(hidden_states)
            
        hidden_states = self.final_layer_norm(hidden_states)
        if self.config.finetune_method == "embedding_prompt" or self.config.finetune_method == "combined":
            hidden_states = hidden_states[:, self.config.embedding_prompt_dim:, :]

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)
        return outputs
    

class Wav2Vec2EncoderLayerStableLayerNorm(nn.Module):
    def __init__(self, layer_idx, config, has_relative_position_bias: bool = True):
        super().__init__()
        self.attention = w2v2.Wav2Vec2Attention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
        )
        self.config = config
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = w2v2.Wav2Vec2FeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        if self.config.finetune_method == "lora":
            if layer_idx > config.num_hidden_layers // 4 * 3:
                self.feed_forward.intermediate_dense    = lora.Linear(config.hidden_size, config.intermediate_size, r=config.lora_rank)
                self.feed_forward.output_dense          = lora.Linear(config.intermediate_size, config.hidden_size, r=config.lora_rank)

        if getattr(config, "adapter_attn_dim", None) is not None:
            self.adapter_layer = w2v2.Wav2Vec2AttnAdapterLayer(config)
        else:
            self.adapter_layer = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        output_attentions: bool = False,
    ):
        attn_residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states
        hidden_states = hidden_states + self.feed_forward(self.final_layer_norm(hidden_states))

        if self.adapter_layer is not None:
            hidden_states = hidden_states + self.adapter_layer(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs

from typing import Any, Dict, Optional

class MMSWrapper(
    nn.Module,
    PyTorchModelHubMixin,
    # repo_url="https://github.com/tiantiaf0627/voxlect",
):
    """
    Wrapper that (a) can be loaded from HF Hub via PyTorchModelHubMixin and
    (b) internally loads a Transformers backbone + feature extractor.

    Key fix: make sure HF hub kwargs (cache_dir, token, revision, local_files_only, etc.)
    get forwarded to the internal Transformers .from_pretrained() calls.
    """

    def __init__(
        self,
        pretrain_model: str = "mms-lid-256",
        hidden_dim: int = 256,
        finetune_method: str = "lora",
        lora_rank: int = 16,
        freeze_params: bool = True,
        output_class_num: int = 4,
        use_conv_output: bool = True,
        backbone_from_pretrained_kwargs: Optional[Dict[str, Any]] = None,
        **_ignored: Any,  # swallow any extra kwargs safely
    ):
        super().__init__()

        # These kwargs will be forwarded to BOTH AutoFeatureExtractor and Wav2Vec2Model downloads.
        hf_kwargs: Dict[str, Any] = dict(backbone_from_pretrained_kwargs or {})
        # Avoid accidental conflicts / duplicates.
        hf_kwargs.pop("output_hidden_states", None)

        # 1) Load backbone + processor with forwarded kwargs
        if pretrain_model == "mms-lid-256":
            model_id = "facebook/mms-lid-256"
        elif pretrain_model == "mms-300m":
            model_id = "facebook/mms-300m"
        else:
            raise ValueError(f"Unknown pretrain_model={pretrain_model!r}")

        self.processor = AutoFeatureExtractor.from_pretrained(model_id, **hf_kwargs)
        self.backbone_model = Wav2Vec2Model.from_pretrained(
            model_id,
            output_hidden_states=True,
            **hf_kwargs,
        )

        self.pretrain_model = pretrain_model
        self.finetune_method = finetune_method
        self.use_conv_output = use_conv_output

        state_dict = self.backbone_model.state_dict()

        # 2) Read + extend config
        self.model_config = self.backbone_model.config
        self.model_config.finetune_method = finetune_method
        self.model_config.lora_rank = lora_rank

        # 3) Swap encoder layers
        self.backbone_model.encoder.layers = nn.ModuleList(
            [
                Wav2Vec2EncoderLayerStableLayerNorm(
                    i, self.model_config, has_relative_position_bias=(i == 0)
                )
                for i in range(self.model_config.num_hidden_layers)
            ]
        )

        # 4) Reload weights
        msg = self.backbone_model.load_state_dict(state_dict, strict=False)

        # 5) Freeze weights
        self.freeze_params = freeze_params
        if self.freeze_params and self.finetune_method != "lora":
            for _, p in self.backbone_model.named_parameters():
                p.requires_grad = False
        elif self.freeze_params and self.finetune_method == "lora":
            for name, p in self.backbone_model.named_parameters():
                p.requires_grad = name in msg.missing_keys
        else:
            for _, p in self.backbone_model.named_parameters():
                p.requires_grad = True

        # 6) Downstream
        self.model_seq = nn.Sequential(
            nn.Conv1d(self.model_config.hidden_size, hidden_dim, 1, padding=0),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv1d(hidden_dim, hidden_dim, 1, padding=0),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv1d(hidden_dim, hidden_dim, 1, padding=0),
        )

        if self.use_conv_output:
            num_layers = self.model_config.num_hidden_layers + 1
            self.weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        else:
            num_layers = self.model_config.num_hidden_layers
            self.weights = nn.Parameter(torch.zeros(num_layers))

        self.out_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_class_num),
        )

    @classmethod
    def _from_pretrained(
        cls,
        model_id: str,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        force_download: bool = False,
        proxies: Optional[Dict[str, str]] = None,
        resume_download: bool = False,
        local_files_only: bool = False,
        token: Optional[str] = None,
        map_location: str = "cpu",
        strict: bool = True,
        **model_kwargs: Any,
    ):
        """
        Bridge HF hub kwargs (cache_dir/token/revision/...) into the constructor so the
        internal Transformers downloads use the same cache.
        """
        hub_kwargs: Dict[str, Any] = {
            "cache_dir": cache_dir,
            "revision": revision,
            "force_download": force_download,
            "proxies": proxies,
            "resume_download": resume_download,
            "local_files_only": local_files_only,
            "token": token,
        }
        # Keep False/None out except local_files_only which is meaningful either way.
        hub_kwargs = {k: v for k, v in hub_kwargs.items() if v is not None}
        if local_files_only:
            hub_kwargs["local_files_only"] = True

        existing = dict(model_kwargs.get("backbone_from_pretrained_kwargs") or {})
        # User-provided backbone kwargs win over injected hub kwargs if they collide
        merged = {**hub_kwargs, **existing}
        model_kwargs["backbone_from_pretrained_kwargs"] = merged

        return super()._from_pretrained(
            model_id=model_id,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,
            token=token,
            map_location=map_location,
            strict=strict,
            **model_kwargs,
        )

    def forward(self, x, length=None, return_feature=False):
        #=========================== Speech Modeling ============================#
        with torch.no_grad():
            signal = list()
            for idx in range(len(x)):
                input = self.processor(x[idx], sampling_rate=16_000, return_tensors="pt", padding=True)
                signal.append(input["input_values"][0].to(x.device))
            signal = torch.stack(signal)
        
        # 2. get length and mask
        if length is not None:
            length = self._get_feat_extract_output_lengths(length.detach().cpu())
            length = length.cuda()
            
        # 3. transformer encoding features
        x = self.backbone_model(
                signal,
                output_hidden_states=True
            ).hidden_states
        
        # 4. stacked feature
        if self.use_conv_output: stacked_feature = torch.stack(x, dim=0)
        else: stacked_feature = torch.stack(x, dim=0)[1:]
        
        # 5. Weighted sum
        _, *origin_shape = stacked_feature.shape
        # Return transformer enc outputs [num_enc_layers, B, T, D]
        if self.use_conv_output:
            stacked_feature = stacked_feature.view(self.backbone_model.config.num_hidden_layers+1, -1)
        else:
            stacked_feature = stacked_feature.view(self.backbone_model.config.num_hidden_layers, -1)
        norm_weights = F.softmax(self.weights, dim=-1)
        
        # Perform weighted average
        weighted_feature = (norm_weights.unsqueeze(-1) * stacked_feature).sum(dim=0)
        features = weighted_feature.view(*origin_shape)
        
        # 6. Pass the weighted average to point-wise 1D Conv
        # B x T x D
        features = features.transpose(1, 2)
        features = self.model_seq(features)
        features = features.transpose(1, 2)
        
        # 7. Pooling
        if length is not None:
            mean, std = list(), list()
            for snt_id in range(features.shape[0]):
                # Avoiding padded time steps
                actual_size = length[snt_id]
                mean.append(torch.mean(features[snt_id, 0:actual_size, ...], dim=0))
            features = torch.stack(mean)
        else:
            features = torch.mean(features, dim=1)

        # 8. Output predictions
        # B x D
        predicted = self.out_layer(features)
        if return_feature: return predicted, features
        return predicted
    
    # From huggingface
    def _get_feat_extract_output_lengths(self, input_length):
        """
        Computes the output length of the convolutional layers
        """
        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return (input_length - kernel_size) // stride + 1
        for kernel_size, stride in zip(self.backbone_model.config.conv_kernel, self.backbone_model.config.conv_stride):
            input_length = _conv_out_length(input_length, kernel_size, stride)
        return input_length
    
    def _get_feature_vector_attention_mask(
        self, feature_vector_length: int, attention_mask: torch.LongTensor, add_adapter=None
    ):
        # Effectively attention_mask.sum(-1), but not inplace to be able to run
        # on inference mode.
        non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]

        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths)
        output_lengths = output_lengths.to(torch.long)

        batch_size = attention_mask.shape[0]

        attention_mask = torch.zeros(
            (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
        )
        # these two operations makes sure that all values before the output lengths idxs are attended to
        attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return attention_mask
