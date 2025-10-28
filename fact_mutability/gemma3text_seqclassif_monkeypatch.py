"""
Universal Gemma-3 sequence-classification heads.
Works with
  • google/gemma-3-1b-(base|it)  (text only)
  • google/gemma-3-4b-* and larger (multimodal)
Import once, early, then call
    AutoModelForSequenceClassification.from_pretrained(...)
"""
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification
from transformers.models.gemma3.configuration_gemma3 import (
    Gemma3Config, Gemma3TextConfig,
)
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3Model, Gemma3TextModel, Gemma3PreTrainedModel,
)
from transformers.modeling_outputs import SequenceClassifierOutputWithPast


# ───────────────────── helpers ─────────────────────────────────────────
def _txt(cfg):                              # text sub-config
    return getattr(cfg, "text_config", cfg)


def _pool_last(token_logits, input_ids, pad_id):
    if input_ids is None or pad_id is None:
        return token_logits[:, -1]
    ends = (~input_ids.eq(pad_id)).cumsum(-1).argmax(-1)
    return token_logits[torch.arange(token_logits.size(0)), ends]


def _compute_loss(config, logits, labels, num_labels):
    if labels is None:
        return None
    if config.problem_type is None:
        if num_labels == 1:
            config.problem_type = "regression"
        elif labels.dtype in (torch.long, torch.int):
            config.problem_type = "single_label_classification"
        else:
            config.problem_type = "multi_label_classification"

    if config.problem_type == "regression":
        return nn.MSELoss()(logits.squeeze(), labels.squeeze())
    if config.problem_type == "single_label_classification":
        return nn.CrossEntropyLoss()(logits, labels)
    return nn.BCEWithLogitsLoss()(logits, labels)


# ───────────────────── multimodal variant (4 B+) ──────────────────────
# class Gemma3ForSequenceClassification(Gemma3Model):
#     config_class = Gemma3Config
#     _no_split_modules = ["GemmaBlock"]
#     keys_to_ignore_at_inference = ["past_key_values"]

#     def __init__(self, config):
#         super().__init__(config)  # backbone
#         self.num_labels = config.num_labels
#         self.score = nn.Linear(_txt(config).hidden_size,
#                                self.num_labels, bias=False)
#         self.post_init()

#     def forward(
#         self,
#         input_ids=None, attention_mask=None, position_ids=None,
#         inputs_embeds=None, pixel_values=None, past_key_values=None,
#         labels=None, use_cache=None, output_attentions=None,
#         output_hidden_states=None, return_dict=None,
#     ):
#         return_dict = (return_dict if return_dict is not None
#                        else self.config.use_return_dict)

#         outputs = super().forward(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             inputs_embeds=inputs_embeds,
#             pixel_values=pixel_values,
#             past_key_values=past_key_values,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=True,
#         )
#         token_logits = self.score(outputs.last_hidden_state)
#         pad_id = getattr(self.config, "pad_token_id",
#                          getattr(_txt(self.config), "pad_token_id", None))
#         pooled = _pool_last(token_logits, input_ids, pad_id)
#         loss = _compute_loss(self.config, pooled, labels, self.num_labels)

#         if not return_dict:
#             out = (pooled,) + outputs[1:]
#             return ((loss,) + out) if loss is not None else out

#         return SequenceClassifierOutputWithPast(
#             loss=loss, logits=pooled,
#             past_key_values=outputs.past_key_values,
#             hidden_states=outputs.hidden_states
#                 if output_hidden_states else None,
#             attentions=outputs.attentions
#                 if output_attentions else None,
#         )


# ───────────────────── text-only variant (1 B) ────────────────────────
class Gemma3TextForSequenceClassification(Gemma3PreTrainedModel):
    """
    **Wraps** Gemma3TextModel in `self.model` – keeps `model.*`
    prefixes so every pretrained weight loads.
    """
    config_class = Gemma3TextConfig
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = Gemma3TextModel(config)              # <- wrapper
        self.score = nn.Linear(config.hidden_size,
                               self.num_labels, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids=None, attention_mask=None, position_ids=None,
        inputs_embeds=None, past_key_values=None, labels=None,
        use_cache=None, output_attentions=None,
        output_hidden_states=None, return_dict=None,
    ):
        return_dict = (return_dict if return_dict is not None
                       else self.config.use_return_dict)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        token_logits = self.score(outputs.last_hidden_state)
        pooled = _pool_last(token_logits, input_ids, self.config.pad_token_id)
        loss = _compute_loss(self.config, pooled, labels, self.num_labels)

        if not return_dict:
            out = (pooled,) + outputs[1:]
            return ((loss,) + out) if loss is not None else out

        return SequenceClassifierOutputWithPast(
            loss=loss, logits=pooled,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states
                if output_hidden_states else None,
            attentions=outputs.attentions
                if output_attentions else None,
        )


# ───────────────────── register with HF factory ───────────────────────
# AutoModelForSequenceClassification.register(
#     Gemma3Config, Gemma3ForSequenceClassification)
AutoModelForSequenceClassification.register(
    Gemma3TextConfig, Gemma3TextForSequenceClassification)