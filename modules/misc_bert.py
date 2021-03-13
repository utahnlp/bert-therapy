# Time-stamp: <2021-03-06>
# --------------------------------------------------------------------
# File Name          : misc-bert.py
# Original Author    : jiessie.cao@gmail.com
# Description        : Auto version for using <CLS> and other tags for classification head
# --------------------------------------------------------------------

import collections
import logging
import re
import os
import numpy as np
import torch
import torch.nn as nn
from misc_config import MISCConfig
import misc_constants
from transformers.modeling_utils import PreTrainedModel

from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer
)
from transformers.modeling_outputs import SequenceClassifierOutput

logger = logging.getLogger(__name__)

# Now we use the same json config
CLS_PRETRAINED_MODEL_ARCHIVE_MAP = {

}

class MergedEmbedding(nn.Embedding):
    def __init__(self, embedding1layer, misc_special_embedding_layer):
        dim1 = embedding1layer.weight.size()[1]
        dim2 = misc_special_embedding_layer.weight.size()[1]
        assert dim1 == dim2, "misc_special_embedding_layer weight is no consistent{} with dim1 = {}, dim2={}".format(dim1,dim2)
        num = embedding1layer.weight.size()[0]+misc_special_embedding_layer.weight.size()[0]
        super(MergedEmbedding, self).__init__(num_embeddings=num, embedding_dim=dim1)
        self.embedding1 = embedding1layer
        self.misc_special_embedding = misc_special_embedding_layer
        self.merged_weight = torch.cat([self.embedding1.weight, self.misc_special_embedding.weight], 0)

    def foward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(
            input, self.merged_weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)



class MISCBERTModel(PreTrainedModel):
    """
    Main entry of MISC BERT model, this model is based on transformer BERT, but only different in classfier design .
    """
    config_class=MISCConfig
    base_model_prefix = ""

    pretrained_model_archieve_map = CLS_PRETRAINED_MODEL_ARCHIVE_MAP
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config=None, tokenizer=None):
        super(MISCBERTModel, self).__init__(config=config)
        self.num_labels = config.num_labels
        self.config = config
        self.misc_special_embeddings = None

        # config is the configuration for pretrained model
        self.encoder = AutoModel.from_pretrained(config.encoder_model_name_or_path,
            cache_dir=config.cache_dir,
            revision=config.revision,
            use_auth_token=True if config.use_auth_token else None,
        )
        if tokenizer:
            self.tokenizer = tokenizer
            if 'special_token_lr' not in config.__dict__ or not config.special_token_lr:
                self.encoder.resize_token_embeddings(len(tokenizer)) # doesn't mess with existing tokens
            else:
                # nn.Module
                old_embeddings = self.encoder.get_input_embeddings()
                self.misc_special_embeddings = nn.Embedding(len(misc_constants.special_speaker_tags), old_embeddings.weight.size()[-1])
                self.encoder._init_weights(self.misc_special_embeddings)
                merged_embeddings = MergedEmbedding(old_embeddings, self.misc_special_embeddings)
                self.encoder.set_input_embeddings(merged_embeddings)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.tokenizer_name if config.tokenizer_name else config.encoder_model_name_or_path,
                cache_dir=config.cache_dir,
                use_fast=config.use_fast_tokenizer,
                revision=config.revision,
                use_auth_token=True if config.use_auth_token else None,
            )
            # still load original tokenizer, add readd the new special tokens
            num_added_tokens = self.tokenizer.add_special_tokens({'additional_special_tokens': misc_constants.special_speaker_tags})
            if 'special_token_lr' not in config.__dict__ or not config.special_token_lr:
                self.encoder.resize_token_embeddings(len(tokenizer)) # doesn't mess with existing tokens
            else:
                # nn.Module
                old_embeddings = self.encoder.get_input_embeddings()
                self.misc_special_embeddings = nn.Embedding(len(misc_constants.special_speaker_tags), old_embeddings.weight.size()[-1])
                self.encoder._init_weights(self.misc_special_embeddings)
                merged_embeddings = MergedEmbedding(old_embeddings, self.misc_special_embeddings)
                self.encoder.set_input_embeddings(merged_embeddings)

        # not use base for BERT
        setattr(self, self.base_model_prefix, torch.nn.Sequential())
        self.classifier = MISCClassificationHead(config)
        # we always use the pretrained BERT, no need for init_weights.

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # adding the start and index position for special U tags, the first maybe the context or the last, it depends on the order
        U_start_tag_id = self.tokenizer.convert_tokens_to_ids(misc_constants.UTTER_START_TAG)
        U_end_tag_id = self.tokenizer.convert_tokens_to_ids(misc_constants.UTTER_END_TAG)
        head_indices = []
        if self.config.use_CLS:
            cls_indices = torch.zeros(input_ids.size()[0], 1).long().to(input_ids.device)
            logger.info("cls_indices:{}".format(cls_indices))
            head_indices.append(cls_indices)

        if self.config.use_start_U:
            # for the second seq dim
            start_U_indices = (input_ids == U_start_tag_id).nonzero(as_tuple=True)[1].long().unsqueeze(-1)
            logger.info("start_U_indices:{}".format(start_U_indices))
            head_indices.append(start_U_indices)

        if self.config.use_end_U:
            end_U_indices = (input_ids == U_end_tag_id).nonzero(as_tuple=True)[1].long().unsqueeze(-1)
            logger.info("end_U_indices:{}".format(end_U_indices))
            head_indices.append(end_U_indices)

        # this input will be used in the forward in the new pretrained BERT model

        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # only use the sequence output, not the pooled_output
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output, cls_head_indices=head_indices)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class MISCClassificationHead(nn.Module):
    """Head for sentence-level classification tasks. Adopted from the Roberta"""

    def __init__(self, config):
        super().__init__()
        acc = 0
        if config.use_CLS:
            acc = acc + 1
        if config.use_start_U:
            acc = acc + 1
        if config.use_end_U:
            acc = acc + 1
        self.acc = acc
        self.dense = nn.Linear(self.acc * config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.tanh = nn.Tanh()
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        if 'cls_head_indices' in kwargs.keys():
            assert self.acc == len(kwargs['cls_head_indices']), "length not match of indices:{}".format(kwargs['cls_head_indices'])
            # [batch_size, self.acc, hidden_size]
            indices_tensor = torch.cat(kwargs['cls_head_indices'], 1).unsqueeze(-1).expand(-1, -1, features.size()[-1])
            # [batch_size, self.acc, hidden_size]
            pre_reshape_x = features.gather(1, indices_tensor)
            x = pre_reshape_x.reshape(features.size()[0], len(kwargs['cls_head_indices'])*features.size()[-1])
            logger.info('using indices_tensor as {}, pre_reshape_x:{}, x:{}'.format(indices_tensor, pre_reshape_x, x))
        else:
            # most of the BERT are just using the first subtoken. maybe adapted for different BERT.
            x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
