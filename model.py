from transformers import BertTokenizer
from transformers import BertModel, BertPreTrainedModel, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup

from transformers import DistilBertTokenizer
from transformers import DistilBertModel, DistilBertPreTrainedModel
import torch.nn as nn
import torch 
import math 
from sklearn.linear_model import LogisticRegression
from utils import platt_scale, gelu, make_bow_vector

CUDA = (torch.cuda.device_count() > 0)
MASK_IDX = 103
    
class CausalBert(DistilBertPreTrainedModel):
    """The model itself."""
    # very hacky implemenation of the causal bert model, but it works for now
    def __init__(self, config):
        # Adapt it to autotokenizers 
        super().__init__(config)

        self.num_labels = config.num_labels
        self.vocab_size = config.vocab_size

        self.distilbert = DistilBertModel(config)
        self.vocab_transform = nn.Linear(config.dim, config.dim)
        self.vocab_layer_norm = nn.LayerNorm(config.dim, eps=1e-12)
        self.vocab_projector = nn.Linear(config.dim, config.vocab_size)

        self.Q_cls = nn.ModuleDict()

        for T in range(2):
            # ModuleDict keys have to be strings..
            self.Q_cls['%d' % T] = nn.Sequential(
                nn.Linear(config.hidden_size + self.num_labels, 200),
                nn.ReLU(),
                nn.Linear(200, self.num_labels))
    
        self.g_cls = nn.Linear(config.hidden_size + self.num_labels, 
            self.config.num_labels)

        self.init_weights()


    def forward(self, W_ids, W_len, W_mask, C, T, Y=None, use_mlm=True):
        if use_mlm:
            W_len = W_len.unsqueeze(1) - 2 # -2 because of the +1 below
            mask_class = torch.cuda.FloatTensor if CUDA else torch.FloatTensor
            mask = (mask_class(W_len.shape).uniform_() * W_len.float()).long() + 1 # + 1 to avoid CLS
            target_words = torch.gather(W_ids, 1, mask)
            mlm_labels = torch.ones(W_ids.shape).long() * -100
            if CUDA:
                mlm_labels = mlm_labels.cuda()
            mlm_labels.scatter_(1, mask, target_words)
            W_ids.scatter_(1, mask, MASK_IDX)

        outputs = self.distilbert(W_ids, attention_mask=W_mask)
        seq_output = outputs[0]
        pooled_output = seq_output[:, 0]
        # seq_output, pooled_output = outputs[:2]
        # pooled_output = self.dropout(pooled_output)

        if use_mlm:
            prediction_logits = self.vocab_transform(seq_output)  # (bs, seq_length, dim)
            prediction_logits = gelu(prediction_logits)  # (bs, seq_length, dim)
            prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
            prediction_logits = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)
            mlm_loss = nn.CrossEntropyLoss()(
                prediction_logits.view(-1, self.vocab_size), mlm_labels.view(-1))
        else:
            mlm_loss = 0.0

        C_bow = make_bow_vector(C.unsqueeze(1), self.num_labels)
        inputs = torch.cat((pooled_output, C_bow), 1)

        # g logits
        g = self.g_cls(inputs)
        if Y is not None:  # TODO train/test mode, this is a lil hacky
            g_loss = nn.CrossEntropyLoss()(g.view(-1, self.num_labels), T.view(-1))
        else:
            g_loss = 0.0

        # conditional expected outcome logits: 
        # run each example through its corresponding T matrix
        # TODO this would be cleaner with sigmoid and BCELoss, but less general 
        #   (and I couldn't get it to work as well)
        Q_logits_T0 = self.Q_cls['0'](inputs)
        Q_logits_T1 = self.Q_cls['1'](inputs)

        if Y is not None:
            T0_indices = (T == 0).nonzero().squeeze()
            Y_T1_labels = Y.clone().scatter(0, T0_indices, -100)

            T1_indices = (T == 1).nonzero().squeeze()
            Y_T0_labels = Y.clone().scatter(0, T1_indices, -100)

            Q_loss_T1 = nn.CrossEntropyLoss()(
                Q_logits_T1.view(-1, self.num_labels), Y_T1_labels)

            Q_loss_T0 = nn.CrossEntropyLoss()(
                Q_logits_T0.view(-1, self.num_labels), Y_T0_labels)

            Q_loss = Q_loss_T0 + Q_loss_T1
        else:
            Q_loss = 0.0

        sm = nn.Softmax(dim=1)
        Q0 = sm(Q_logits_T0)[:, 1]
        Q1 = sm(Q_logits_T1)[:, 1]
        g = sm(g)[:, 1]

        return g, Q0, Q1, g_loss, Q_loss, mlm_loss
