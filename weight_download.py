from huggingface_hub import snapshot_download
downloads = ["LanguageBind/Video-LLaVA-7B"]
for download in downloads:
    snapshot_download(download,cache_dir='/mnt/bn/cq-mllm-data/chenxiuyuan/weight_hf/'+download)

# # from datasets import load_dataset
# # download = "liuhaotian/llava-bench-in-the-wild"
# # dataset = load_dataset(download,cache_dir='/mnt/bn/cq-mllm-data/chenxiuyuan/data_hf/'+download)


# import torch
# from PIL import Image
# import numpy as np
# from transformers import Blip2Processor, Blip2Model, Blip2QFormerModel, Blip2Config

# def cross_attention_mask(query_total_len, query_each_group_len,
#                          key_total_len, key_first_group_len, key_each_group_len):
#         mask = torch.ones((query_total_len, key_total_len))

#         assert query_total_len % query_each_group_len == 0 and (key_total_len - key_first_group_len) % key_each_group_len == 0
#         query_num_groups = (query_total_len) // query_each_group_len
#         key_num_groups = (key_total_len - key_first_group_len) // key_each_group_len
#         assert query_num_groups == key_num_groups

#         for i in range(1, query_num_groups + 1):
#             query_start = (i - 1) * query_each_group_len
#             query_end = query_start + query_each_group_len

#             key_start = key_first_group_len + i * key_each_group_len
#             key_end = key_total_len

#             mask[query_start:query_end, key_start:key_end] = 0

#         return mask
        
# class myqformer(Blip2QFormerModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.frame_query_length = 32
#         self.frame_embedding_length = 60

#     def get_extended_attention_mask(
#         self,
#         attention_mask: torch.Tensor,
#         device: torch.device,
#     ) -> torch.Tensor:
#         """
#         Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

#         Arguments:
#             attention_mask (`torch.Tensor`):
#                 Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
#             input_shape (`Tuple[int]`):
#                 The shape of the input to the model.
#             device: (`torch.device`):
#                 The device of the input to the model.

#         Returns:
#             `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
#         """
#         # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
#         # ourselves in which case we just need to make it broadcastable to all heads.
#         # if attention_mask.dim() == 3:
#         #     extended_attention_mask = attention_mask[:, None, :, :]
#         # if attention_mask.dim() == 2:
#         #     # Provided a padding mask of dimensions [batch_size, seq_length]
#         #     # - the model is an encoder, so make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
#         #     extended_attention_mask = attention_mask.repeat()
#         # else:
#         #     raise ValueError(
#         #         "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
#         #             input_shape, attention_mask.shape
#         #         )
#         #     )

#         # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
#         # masked positions, this operation will create a tensor which is 0.0 for
#         # positions we want to attend and -10000.0 for masked positions.
#         # Since we are adding it to the raw scores before the softmax, this is
#         # effectively the same as removing these entirely.
#         extended_attention_mask = attention_mask.to(dtype=self.dtype).to(device)  # fp16 compatibility
#         extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
#         return extended_attention_mask

#     def forward(
#         self,
#         query_embeds,
#         attention_mask=None,
#         head_mask=None,
#         encoder_hidden_states=None,
#         encoder_attention_mask=None,
#         past_key_values=None,
#         use_cache=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#     ):
#         r"""
#         encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, `optional`):
#             Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
#             the model is configured as a decoder.
#         encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, `optional`):
#             Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
#             the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
#             - 1 for tokens that are **not masked**,
#             - 0 for tokens that are **masked**.
#         past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of:
#             shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`): Contains precomputed key and
#             value hidden states of the attention blocks. Can be used to speed up decoding. If `past_key_values` are
#             used, the user can optionally input only the last `decoder_input_ids` (those that don't have their past key
#             value states given to this model) of shape `(batch_size, 1)` instead of all `decoder_input_ids` of shape
#             `(batch_size, sequence_length)`.
#         use_cache (`bool`, `optional`):
#             If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
#             `past_key_values`).
#         """
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         # past_key_values_length
#         past_key_values_length = (
#             past_key_values[0][0].shape[2] - self.config.query_length if past_key_values is not None else 0
#         )

#         query_length = query_embeds.shape[1] if query_embeds is not None else 0

#         embedding_output = self.layernorm(query_embeds)
#         embedding_output = self.dropout(embedding_output)

#         input_shape = embedding_output.size()[:-1]
#         batch_size, seq_length = input_shape
#         device = embedding_output.device

#         frame_query_length = self.frame_query_length
#         frame_embedding_length = self.frame_embedding_length
#         attention_mask = cross_attention_mask(query_length, frame_query_length, 
#                                             query_length, 0, frame_query_length
#                                             )
#         attention_mask = attention_mask.unsqueeze(0).unsqueeze(0).to(device).repeat(batch_size, self.config.num_attention_heads ,1,1)
#         extended_attention_mask = self.get_extended_attention_mask(attention_mask, device)
#         encoder_extended_attention_mask = cross_attention_mask(query_length, frame_query_length,
#                                             encoder_hidden_states.shape[1], frame_embedding_length, frame_embedding_length
#                                             )
#         encoder_extended_attention_mask = encoder_extended_attention_mask.unsqueeze(0).unsqueeze(0).to(device).repeat(batch_size, self.config.num_attention_heads ,1,1)
#         encoder_extended_attention_mask = self.get_extended_attention_mask(encoder_extended_attention_mask, device)
        


#         # If a 2D or 3D attention mask is provided for the cross-attention
#         # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
#         # if encoder_hidden_states is not None:
#         #     if type(encoder_hidden_states) == list:
#         #         encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states[0].size()
#         #     else:
#         #         encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
#         #     encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)

#         #     if type(encoder_attention_mask) == list:
#         #         encoder_extended_attention_mask = [self.invert_attention_mask(mask) for mask in encoder_attention_mask]
#         #     elif encoder_attention_mask is None:
#         #         encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
#         #         encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
#         #     else:
#         #         encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
#         # else:
#         #     encoder_extended_attention_mask = None

#         # Prepare head mask if needed
#         # 1.0 in head_mask indicate we keep the head
#         # attention_probs has shape bsz x n_heads x N x N
#         # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
#         # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
#         head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

#         encoder_outputs = self.encoder(
#             embedding_output,
#             attention_mask=extended_attention_mask,
#             head_mask=head_mask,
#             encoder_hidden_states=encoder_hidden_states,
#             encoder_attention_mask=encoder_extended_attention_mask,
#             past_key_values=past_key_values,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#             query_length=query_length,
#         )
#         sequence_output = encoder_outputs[0]
#         pooled_output = sequence_output[:, 0, :]

#         # if not return_dict:
#         return (sequence_output, pooled_output) + encoder_outputs[1:]

#         # return BaseModelOutputWithPoolingAndCrossAttentions(
#         #     last_hidden_state=sequence_output,
#         #     pooler_output=pooled_output,
#         #     past_key_values=encoder_outputs.past_key_values,
#         #     hidden_states=encoder_outputs.hidden_states,
#         #     attentions=encoder_outputs.attentions,
#         #     cross_attentions=encoder_outputs.cross_attentions,
#         # )

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model_dir = "/mnt/bn/cq-mllm-data/chenxiuyuan/weight_hf/Salesforce/blip2-opt-2.7b/models--Salesforce--blip2-opt-2.7b/snapshots/235c75ea3861136b9dd202c6edc6a7ba285c35e3"
# # processor = Blip2Processor.from_pretrained(model_dir)
# config = Blip2Config.from_pretrained(model_dir)
# # model.to(device)  # doctest: +IGNORE_RESULT
# # url = "/mnt/bn/arnold-labcq-llm/cxy/Untitled.png"
# # image = Image.open(url)
# # inputs = processor(images=image, return_tensors="pt").to(device)
# # image_embeds = model.get_image_features(**inputs)[0]

# frames = 5
# image_embeds = torch.randn((2, 60*frames, 1408)).to(device)
# query_tokens = torch.randn((1, 32, 768)).repeat(image_embeds.shape[0], frames-1, 1).to(device)
# qformer = myqformer(config.qformer_config).to(device)
# print(query_tokens.shape, image_embeds.shape)
# query_outputs = qformer(
#     query_embeds=query_tokens,
#     encoder_hidden_states=image_embeds,
# )[0]