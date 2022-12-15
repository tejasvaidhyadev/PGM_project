from transformers import BertModel, BertPreTrainedModel, AdamW, AutoModel, AutoConfig

import torch.nn as nn
import torch 
import math 
from sklearn.linear_model import LogisticRegression
from utils import platt_scale, gelu

CUDA = (torch.cuda.device_count() > 0)
MASK_IDX = 103
  
            
class CausalBert(nn.Module):
    """The model itself."""
    # very hacky implemenation of the causal bert model, but it works for now
    # now the current version support all the formate
    def __init__(self, 
            model_card,
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False):
        
        # Adapt it to autotokenizers 
        super(CausalBert, self).__init__()
        config = AutoConfig.from_pretrained(model_card)
        
        self.distilbert = AutoModel.from_config(config)
        print(self.distilbert)
        self.distilbert.config.output_hidden_states = output_hidden_states
        self.num_labels = num_labels

        self.vocab_size = config.vocab_size
        self.vocab_transform = nn.Linear(config.hidden_size, config.hidden_size)
        self.vocab_layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.vocab_projector = nn.Linear(config.hidden_size, config.vocab_size)

        self.Q_cls = nn.ModuleDict()

        for T in range(2):
            # ModuleDict keys have to be strings..
            self.Q_cls['%d' % T] = nn.Sequential(
                nn.Linear(config.hidden_size , 200),
                nn.ReLU(),
                nn.Linear(200, self.num_labels))
    
        self.g_cls = nn.Linear(config.hidden_size, 
            self.num_labels)

        #self.init_weights()


    def forward(self, W_ids, W_len, W_mask, T, Y=None, use_mlm=True):
        if use_mlm:
            W_len = W_len.unsqueeze(1) - 2 # -2 because of the +1 below
            mask_class = torch.cuda.FloatTensor if CUDA else torch.FloatTensor
            # uniformlly masking words
            mask = (mask_class(W_len.shape).uniform_() * W_len.float()).long() + 1 # + 1 to avoid CLS
            target_words = torch.gather(W_ids, 1, mask)
            mlm_labels = torch.ones(W_ids.shape).long() * -100
            if CUDA:
                mlm_labels = mlm_labels.cuda()
            mlm_labels.scatter_(1, mask, target_words)
            W_ids.scatter_(1, mask, MASK_IDX)

        outputs = self.distilbert(W_ids, attention_mask=W_mask)
        
        # rep of all the words 
        seq_output = outputs[0]
        
        # rep vec of CLS token
        pooled_output = seq_output[:, 0]
        # seq_output, pooled_output = outputs[:2]
        # pooled_output = self.dropout(pooled_output)
        # L(wi; ξ, γ) = yi − Q˜(ti, λi; γ)**2+ CrossEntti, g˜(λi; γ) + LU(wi; ξ, γ)
        if use_mlm:
            prediction_logits = self.vocab_transform(seq_output)  # (bs, seq_length, dim)
            prediction_logits = gelu(prediction_logits)  # (bs, seq_length, dim)
            prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
            prediction_logits = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)
            mlm_loss = nn.CrossEntropyLoss()(
                prediction_logits.view(-1, self.vocab_size), mlm_labels.view(-1))
        else:
            mlm_loss = 0.0

        inputs = pooled_output
        
        # g logits
        g = self.g_cls(inputs)
        if Y is not None:  
            # Compute cross-entropy loss between generator output and target labels
            g_loss = nn.CrossEntropyLoss()(g.view(-1, self.num_labels), T.view(-1))
        else:
            g_loss = 0.0

        Q_logits_T0 = self.Q_cls['0'](inputs)
        Q_logits_T1 = self.Q_cls['1'](inputs)

        if Y is not None:
            # Cross entropy for T =1  and Y when T = 1
            Y_T1_labels = Y.clone()
            T0_indices = (T == 0).nonzero().squeeze()
            
            mask_T0 = torch.zeros_like(Y_T1_labels).scatter_(0, T0_indices, 1)
            # mask indices where T = 1
            mask_T1 = 1 - mask_T0
            # cross entropy for T = 1 and Y when T = 1
            # apply mask to Y_T1_labels

            Y_T1_labels = Y_T1_labels * mask_T1

            #Y_T1_labels = Y.clone().scatter(0, T0_indices, -100)
            Y_T1_labels = Y * mask_T1
            Q_loss_T1 = nn.CrossEntropyLoss()(Q_logits_T1.view(-1, self.num_labels), Y_T1_labels.view(-1))
            Q_loss_T1 = Q_loss_T1 * mask_T1.sum() / mask_T1.shape[0]

            # cross entropy for T = 0 and Y when T = 0
            # apply mask to Y_T0_labels
            Y_T0_labels = Y * mask_T0
            Q_loss_T0 = nn.CrossEntropyLoss()(Q_logits_T0.view(-1, self.num_labels), Y_T0_labels.view(-1))
            Q_loss_T0 = Q_loss_T0 * mask_T0.sum() / mask_T0.shape[0]

        # The above is a bit hacky, but it works for now
            Q_loss = Q_loss_T0 + Q_loss_T1
        else:
            Q_loss = 0.0

        #sm = nn.Sigmoid()
        sm = nn.Softmax(dim=1)
        Q0 = sm(Q_logits_T0)[:, 1]
        Q1 = sm(Q_logits_T1)[:, 1]
        g = sm(g)[:, 1]

        return g, Q0, Q1, g_loss, Q_loss, mlm_loss
