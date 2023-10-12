""" 
Python version: 3.9
Description: Contains utility functions and helper classes for different layers.
"""

# %% Importing libraries
import torch.nn as nn

from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        else:
            raise NotImplementedError

def cl_init(cls, pooler_type_, config, temp_):
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = pooler_type_
    cls.pooler = Pooler(pooler_type_)
    cls.mlp = MLPLayer(config)
    cls.sim = Similarity(temp=temp_)
    cls.init_weights()

def cl_forward(cls, encoder, pooler_type, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, 
                labels=None, output_attentions=None, output_hidden_states=None, return_dict=None):
    #import ipdb; ipdb.set_trace();
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    batch_size = int(input_ids.size(0))
    
    # mlm_outputs = None
    # Flatten input for encoding
    # input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
    # attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent len)
    # if token_type_ids is not None:
    #     token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)

    # Get raw embeddings
    outputs = encoder(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds,
                        output_attentions=output_attentions, output_hidden_states=False if pooler_type == 'cls' else True, return_dict=True)

    # Pooling
    pooler_output = cls.pooler(attention_mask, outputs)
    pooler_output = pooler_output.view((batch_size, pooler_output.size(-1))) # (bs, num_sent, hidden)

    # If using "cls", we add an extra MLP layer
    # (same as BERT's original implementation) over the representation.
    if cls.pooler_type == "cls":
        pooler_output = cls.mlp(pooler_output)

    return pooler_output

def sentemb_forward(cls, encoder, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, 
                    inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None):

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    outputs = encoder(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask,
                        inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=True if cls.pooler_type in ['avg_top2', 'avg_first_last'] else False,
                        return_dict=True)

    pooler_output = cls.pooler(attention_mask, outputs)
    if cls.pooler_type == "cls":
        pooler_output = cls.mlp(pooler_output)

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(pooler_output=pooler_output, last_hidden_state=outputs.last_hidden_state, 
                                                        hidden_states=outputs.hidden_states)