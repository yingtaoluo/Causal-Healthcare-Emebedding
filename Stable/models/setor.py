# -*- coding: utf-8 -*-
import pdb
import torch
import warnings
from Normal.train import GPU
import torch.nn as nn
import torch
import math
import copy
import os
import logging
# from torchdiffeq import odeint, odeint_adjoint

warnings.filterwarnings('ignore')

device = torch.device("cuda:%d" % GPU if torch.cuda.is_available() and GPU >= 0 else "cpu")

VERY_BIG_NUMBER = 1e30
VERY_SMALL_NUMBER = 1e-30
VERY_POSITIVE_NUMBER = VERY_BIG_NUMBER
VERY_NEGATIVE_NUMBER = -VERY_BIG_NUMBER
logger = logging.getLogger(__name__)


MAX_NUM_STEPS = 1000  # Maximum number of steps for ODE solver


class ODEFunc(nn.Module):
    """MLP modeling the derivative of ODE system.
    Parameters
    ----------
    device : torch.device
    data_dim : int
        Dimension of data.
    hidden_dim : int
        Dimension of hidden layers.
    augment_dim: int
        Dimension of augmentation. If 0 does not augment ODE, otherwise augments
        it with augment_dim dimensions.
    time_dependent : bool
        If True adds time as input, making ODE time dependent.
    non_linearity : string
        One of 'relu' and 'softplus'
    """
    def __init__(self, device, data_dim, hidden_dim, augment_dim=0, time_dependent=False, non_linearity='relu'):
        super(ODEFunc, self).__init__()
        self.device = device
        self.augment_dim = augment_dim
        self.data_dim = data_dim
        self.input_dim = data_dim + augment_dim
        self.hidden_dim = hidden_dim
        self.nfe = 0  # Number of function evaluations
        self.time_dependent = time_dependent

        if time_dependent:
            self.fc1 = nn.Linear(self.input_dim + 1, hidden_dim)
        else:
            self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, self.input_dim)

        if non_linearity == 'relu':
            self.non_linearity = nn.ReLU(inplace=True)
        elif non_linearity == 'softplus':
            self.non_linearity = nn.Softplus()

    def forward(self, t, x):
        """
        Parameters
        ----------
        t : torch.Tensor
            Current time. Shape (1,).
        x : torch.Tensor
            Shape (batch_size, input_dim)
        """
        # Forward pass of model corresponds to one function evaluation, so
        # increment counter
        self.nfe += 1
        if self.time_dependent:
            # Shape (batch_size, 1)
            t_vec = torch.ones(x.shape[0], 1).to(self.device) * t
            # Shape (batch_size, data_dim + 1)
            t_and_x = torch.cat([t_vec, x], 1)
            # Shape (batch_size, hidden_dim)
            out = self.fc1(t_and_x)
        else:
            out = self.fc1(x)
        out = self.non_linearity(out)
        out = self.fc2(out)
        out = self.non_linearity(out)
        out = self.fc3(out)
        return out


# class ODEBlock(nn.Module):
#     """
#     Solves ODE defined by odefunc.
#     Parameters
#     ----------
#     device : torch.device
#     odefunc : ODEFunc instance or anode.conv_models.ConvODEFunc instance
#         Function defining dynamics of system.
#     is_conv : bool
#         If True, treats odefunc as a convolutional model.
#     tol : float
#         Error tolerance.
#     adjoint : bool
#         If True calculates gradient with adjoint method, otherwise
#         backpropagates directly through operations of ODE solver.
#     """
#     def __init__(self, device, odefunc, is_conv=False, tol=1e-3, adjoint=False):
#         super(ODEBlock, self).__init__()
#         self.adjoint = adjoint
#         self.device = device
#         self.is_conv = is_conv
#         self.odefunc = odefunc
#         self.tol = tol
#
#     def forward(self, x, eval_times=None):
#         """Solves ODE starting from x.
#         Parameters
#         ----------
#         x : torch.Tensor
#             Shape (batch_size, self.odefunc.data_dim)
#         eval_times : None or torch.Tensor
#             If None, returns solution of ODE at final time t=1. If torch.Tensor
#             then returns full ODE trajectory evaluated at points in eval_times.
#         """
#         # Forward pass corresponds to solving ODE, so reset number of function
#         # evaluations counter
#         self.odefunc.nfe = 0
#
#         if eval_times is None:
#             integration_time = torch.tensor([0, 1]).float().type_as(x)
#         else:
#             integration_time = eval_times.type_as(x)
#
#         if self.odefunc.augment_dim > 0:
#             if self.is_conv:
#                 # Add augmentation
#                 batch_size, channels, height, width = x.shape
#                 aug = torch.zeros(batch_size, self.odefunc.augment_dim, height, width).to(self.device)
#                 # Shape (batch_size, channels + augment_dim, height, width)
#                 x_aug = torch.cat([x, aug], 1)
#             else:
#                 # Add augmentation
#                 aug = torch.zeros(x.shape[0], self.odefunc.augment_dim).to(self.device)
#                 # Shape (batch_size, data_dim + augment_dim)
#                 x_aug = torch.cat([x, aug], 1)
#         else:
#             x_aug = x
#
#         if self.adjoint:
#             out = odeint_adjoint(self.odefunc, x_aug, integration_time,
#                                  rtol=self.tol, atol=self.tol, method='euler',
#                                  options={'max_num_steps': MAX_NUM_STEPS})
#         else:
#             out = odeint(self.odefunc, x_aug, integration_time,
#                          rtol=self.tol, atol=self.tol, method='euler',
#                          options={'max_num_steps': MAX_NUM_STEPS})
#
#         if eval_times is None:
#             return out[1]  # Return only final time
#         else:
#             return out


# class ODENet(nn.Module):
#     """An ODEBlock followed by a Linear layer.
#     Parameters
#     ----------
#     device : torch.device
#     data_dim : int
#         Dimension of data.
#     hidden_dim : int
#         Dimension of hidden layers.
#     output_dim : int
#         Dimension of output after hidden layer. Should be 1 for regression or
#         num_classes for classification.
#     augment_dim: int
#         Dimension of augmentation. If 0 does not augment ODE, otherwise augments
#         it with augment_dim dimensions.
#     time_dependent : bool
#         If True adds time as input, making ODE time dependent.
#     non_linearity : string
#         One of 'relu' and 'softplus'
#     tol : float
#         Error tolerance.
#     adjoint : bool
#         If True calculates gradient with adjoint method, otherwise
#         backpropagates directly through operations of ODE solver.
#     """
#     def __init__(self, device, data_dim, hidden_dim, output_dim=1,
#                  augment_dim=0, time_dependent=False, non_linearity='relu',
#                  tol=1e-3, adjoint=False):
#         super(ODENet, self).__init__()
#         self.device = device
#         self.data_dim = data_dim
#         self.hidden_dim = hidden_dim
#         self.augment_dim = augment_dim
#         self.output_dim = output_dim
#         self.time_dependent = time_dependent
#         self.tol = tol
#
#         odefunc = ODEFunc(device, data_dim, hidden_dim, augment_dim, time_dependent, non_linearity)
#
#         self.odeblock = ODEBlock(device, odefunc, tol=tol, adjoint=adjoint)
#         self.linear_layer = nn.Linear(self.odeblock.odefunc.input_dim, self.output_dim)
#
#     def forward(self, x, eval_times=None):
#         features = self.odeblock(x, eval_times)
#         pred = self.linear_layer(features)
#         return pred


# class ODEEmbeddings(nn.Module):
#     def __init__(self, config):
#         super(ODEEmbeddings, self).__init__()
#         self.odeNet_los = ODENet(config.device, config.hidden_size, config.hidden_size)
#         self.odeNet_interval = ODENet(config.device, config.hidden_size, config.hidden_size,
#                                       output_dim=config.hidden_size, augment_dim=10, time_dependent=True)
#         self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#
#     def forward(self, input_states):
#
#         seq_length = input_states.size(1)
#         integration_time = torch.linspace(0., 1., seq_length)
#         y0 = input_states[:, 0, :]
#         interval_embeddings = self.odeNet_interval(y0, integration_time).permute(1, 0, 2)
#
#         los_embeddings = self.odeNet_los(input_states)
#         embeddings = input_states + interval_embeddings + los_embeddings
#
#         embeddings = self.dropout(embeddings)
#         embeddings = self.LayerNorm(embeddings)
#         return embeddings


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class AttentionPooling(nn.Module):
    def __init__(self, config):
        super(AttentionPooling, self).__init__()
        self.config = config
        self.linear1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear2 = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, input_tensor, pooling_mask):
        pooling_score = self.linear1(input_tensor)
        # pooling_score = ACT2FN['relu'](pooling_score)
        pooling_score = ACT2FN[self.config.hidden_act](pooling_score)

        pooling_score = self.linear2(pooling_score)

        pooling_score += pooling_mask
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=1)(pooling_score)

        attention_output = (attention_probs * input_tensor).sum(dim=1)
        return attention_output


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class BertOutput(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = BertLayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.code_size, config.hidden_size)
        if config.position_embedding == 'True':
            self.is_position_embedding = True
        else:
            self.is_position_embedding = False
        if self.is_position_embedding:
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        if self.is_position_embedding:
            seq_length = input_ids.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
            position_embeddings = self.position_embeddings(position_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        if self.is_position_embedding:
            embeddings = words_embeddings + position_embeddings + token_type_embeddings
        else:
            embeddings = words_embeddings + token_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class PositionEmbeddings(nn.Module):
    def __init__(self, max_position, hidden_size, dropout):
        super(PositionEmbeddings, self).__init__()
        self.position_embeddings = nn.Embedding(max_position, hidden_size)
        self.LayerNorm = BertLayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_states):

        seq_length = input_states.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_states.device)
        position_ids = position_ids.unsqueeze(0).repeat(input_states.size(0), 1)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = input_states + position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, alpha, hidden_size, num_attention_heads, dropout):
        super(BertSelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.alpha = alpha
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.cnn = nn.Conv2d(self.num_attention_heads, self.num_attention_heads, 1, stride=1, padding=0)

        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, previous_attention, output_attentions=False):

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask
        # print(attention_scores[0][0][0][0])
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # residual connect the previous attentions.
        if previous_attention is not None and self.alpha > 0:
            # 1*1 CNN for previous attentions
            attention_probs = self.alpha * self.cnn(previous_attention) + (1 - self.alpha) * attention_probs
            # # residual connection with previous attention
            # attention_probs = self.alpha * previous_attention + (1 - self.alpha) * attention_probs
            attention_probs = nn.Softmax(dim=-1)(attention_probs)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, hidden_size, dropout):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = BertLayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, alpha, hidden_size, num_attention_heads, dropout):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(alpha, hidden_size, num_attention_heads, dropout)
        self.output = BertSelfOutput(hidden_size, dropout)

    def forward(self, input_tensor, attention_mask, previous_attention, output_attentions=False):
        self_output = self.self(input_tensor, attention_mask, previous_attention, output_attentions)
        attention_output = self.output(self_output[0], input_tensor)
        outputs = (attention_output,) + self_output[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_act):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = ACT2FN[hidden_act] \
            if isinstance(hidden_act, str) else hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, alpha, hidden_size, intermediate_size, hidden_act,
                 num_attention_heads, dropout):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(alpha, hidden_size, num_attention_heads, dropout)
        self.intermediate = BertIntermediate(hidden_size, intermediate_size, hidden_act)
        self.output = BertOutput(hidden_size, intermediate_size, dropout)

    def forward(self, hidden_states, attention_mask, previous_attention, output_attentions=False):
        attention_output = self.attention(hidden_states, attention_mask, previous_attention, output_attentions)
        outputs = attention_output[1:]  # add self attentions if we output attention weights
        intermediate_output = self.intermediate(attention_output[0])
        layer_output = self.output(intermediate_output, attention_output[0])

        outputs = (layer_output,) + outputs
        return outputs


class BertEncoder(nn.Module):
    def __init__(self, alpha, hidden_size, intermediate_size, hidden_act,
                 num_attention_heads, num_layers, dropout):
        super(BertEncoder, self).__init__()
        layer = BertLayer(alpha, hidden_size, intermediate_size, hidden_act,
                 num_attention_heads, dropout)
        layers = []
        for _ in range(num_layers):
            layers.append(copy.deepcopy(layer))
        self.layer = nn.ModuleList(layers)

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True,
                output_attentions=False, previous_attention=None):
        all_encoder_layers = []
        all_attentions = () if output_attentions else None
        for layer_module in self.layer:
            # print(hidden_states[0][0][0])
            layer_outputs = layer_module(hidden_states, attention_mask, previous_attention, output_attentions)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)

            hidden_states = layer_outputs[0]
            previous_attention = layer_outputs[1] if len(layer_outputs) == 2 else layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers, all_attentions


class PreTrainedBertModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pre-trained models.
    """
    def __init__(self, config,  *inputs, **kwargs):
        super(PreTrainedBertModel, self).__init__()
        self.config = config

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=1.0)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name, config, state_dict=None, *inputs, **kwargs):
        print('parameters in inputs: ', *inputs)

        # Instantiate model.
        model = cls(config, *inputs, **kwargs)

        if state_dict is None:
            weights_path = os.path.join(pretrained_model_name, 'pytorch_model.bin')
            state_dict = torch.load(weights_path).state_dict()

        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
        load(model)
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))

        return model, missing_keys


class SETOR(PreTrainedBertModel):
    def __init__(self, alpha, hidden_size, intermediate_size, hidden_act,
                 icd_size, pro_size, max_position, num_attention_heads, num_layers, dropout):
        super(SETOR, self).__init__(alpha, hidden_size, intermediate_size, hidden_act,
                                    icd_size, max_position, num_attention_heads, num_layers, dropout)
        self.dia_embedding = nn.Linear(icd_size, hidden_size)
        self.pro_embedding = nn.Linear(pro_size, hidden_size)
        self.encoder_patient = BertEncoder(alpha, hidden_size, intermediate_size, hidden_act,
                 num_attention_heads, num_layers, dropout)
        self.position_embedding = PositionEmbeddings(max_position, hidden_size, dropout)
        # self.position_embedding = ODEEmbeddings(config)

        self.classifier_patient = nn.Linear(hidden_size, icd_size)
        # self.apply(self.init_bert_weights)

    def forward(self,
                input_data,
                choice,
                mask=None,
                output_attentions=False
                ):
        if choice == 'dia':
            embed_data = self.dia_embedding(input_data)
        else:
            embed_data = self.pro_embedding(input_data)

        # add position embedding
        visit_outs = self.position_embedding(embed_data)

        if mask is None:
            # mask = torch.zeros(embed_data.shape[1], embed_data.shape[1]).float().to(device)
            mask = (torch.triu(torch.ones(embed_data.shape[1], embed_data.shape[1])) == 1).transpose(0, 1).to(device)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

        # extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        # extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        # extended_attention_mask = (1.0 - extended_attention_mask) * VERY_NEGATIVE_NUMBER

        patient_outputs, visit_attentions = self.encoder_patient(visit_outs, mask, output_all_encoded_layers=False,
                                                                 output_attentions=output_attentions)

        prediction_scores_patient = self.classifier_patient(patient_outputs[-1])
        # prediction_scores_patient = torch.sigmoid(prediction_scores_patient)

        return prediction_scores_patient, patient_outputs[-1]
