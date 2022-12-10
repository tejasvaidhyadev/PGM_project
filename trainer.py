from dataloader import build_dataloader
from model import CausalBert
import torch 
from torch import optim
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import argparse
import os
from utils import *
import pandas as pd
import numpy as np
CUDA = (torch.cuda.device_count() > 0)

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='experiment')
parser.add_argument("--data_dir", default="./PeerRead/proc/beta0_0.25beta1_5.0gamma_0.0.csv", type=str, required=False,
                    help="The input training data file (a csv file).")
parser.add_argument("--model_card", default="bert-base-uncased", type=str, required=False,
                    help="The model card to use.")
parser.add_argument("--batch_size", default=32, type=int, required=False,
                    help="Batch size for training.")

parser.add_argument("--learning_rate", default=2e-5, type=float, required=False,
                    help="The initial learning rate for Adam.")
parser.add_argument("--num_train_epochs", default=1, type=float, required=False,
                    help="Total number of training epochs to perform.")
parser.add_argument("--warmup_steps", default=0, type=int, required=False,
                    help="Linear warmup over warmup_steps.")    
parser.add_argument("--output_dir", default="output", type=str, required=False,
                    help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--Q_weigth", default=0.1, type=float, required=False,
                    help="Weigthing of Q term in loss")
parser.add_argument("--g_weight", default=1.0, type=float, required=False,
                    help="Weigthing of g term in loss")
parser.add_argument("--mlm_weight", default=1.0, type=float, required=False,
                    help="Weigthing of g term in loss")

args = parser.parse_args()

MASK_IDX = 103

class CausalBertWrapper:
    """Model wrapper in charge of training and inference."""

    def __init__(self, g_weight=1.0, Q_weight=0.1, mlm_weight=1.0,
        batch_size=32):
        self.model = CausalBert(
            args.model_card,
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False)
        if CUDA:
            self.model = self.model.cuda()

        self.loss_weights = {
            'g': g_weight,
            'Q': Q_weight,
            'mlm': mlm_weight
        }
        self.batch_size = batch_size


    def train(self, texts, treatments, outcomes,
            learning_rate=2e-5, epochs=5):
        dataloader = build_dataloader( args.model_card, self.batch_size,
            texts, treatments, outcomes)

        self.model.train()
        optimizer = optim.Adam( self.model.parameters(), lr=learning_rate, eps=1e-8)
        total_steps = len(dataloader) * epochs
        warmup_steps = total_steps * 0.1
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        for epoch in range(epochs):
            losses = []
            self.model.train()
            for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
                    logging.info(f'Epoch {epoch}')
                    if CUDA: 
                        batch = (x.cuda() for x in batch)
                    W_ids, W_len, W_mask, C, T, Y = batch
                    # while True:
                    self.model.zero_grad()
                    g, Q0, Q1, g_loss, Q_loss, mlm_loss = self.model(W_ids, W_len, W_mask, T, Y)
                    loss = self.loss_weights['g'] * g_loss + \
                            self.loss_weights['Q'] * Q_loss + \
                            self.loss_weights['mlm'] * mlm_loss
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    losses.append(loss.detach().cpu().item())
            logging.info(np.mean(losses))
                    # if step > 5: continue
        return self.model


    def inference(self, texts, outcome=None):
        self.model.eval()
        dataloader = build_dataloader(args.model_card, self.batch_size, texts, outcomes=outcome,
            sampler='sequential')
        Q0s = []
        Q1s = []
        Ys = []
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            if CUDA: 
                batch = (x.cuda() for x in batch)
            W_ids, W_len, W_mask, C, T, Y = batch
            g, Q0, Q1, _, _, _ = self.model(W_ids, W_len, W_mask, T, use_mlm=False)
            Q0s += Q0.detach().cpu().numpy().tolist()
            Q1s += Q1.detach().cpu().numpy().tolist()
            Ys += Y.detach().cpu().numpy().tolist()
        probs = np.array(list(zip(Q0s, Q1s)))
        preds = np.argmax(probs, axis=1)

        return probs, preds, Ys, g

    def ATE(self, C, W, Y=None, platt_scaling=False):
        Q_probs, _, Ys, g = self.inference(W, C, outcome=Y)
        if platt_scaling and Y is not None:
            Q0 = platt_scale(Ys, Q_probs[:, 0])[:, 0]
            Q1 = platt_scale(Ys, Q_probs[:, 1])[:, 1]
        else:
            Q0 = Q_probs[:, 0]
            Q1 = Q_probs[:, 1]
        # q_t0, q_t1, g, t, y, prob_t
        # get treatment from dataloader
        
        treatments = []
        for batch in dataloader:
            treatments += batch[4].detach().cpu().numpy().tolist()
        df = pd.DataFrame(list(zip(Q0s, Q1s, g, treatments, Ys)), columns=['q_t0', 'q_t1', "g", 't', 'Y'])
        df.to_csv('logs/'+args.exp_name+'/inference.csv', index=False)
        
        return np.mean(Q1 - Q0)
    
    def gt(self, df):
        gt = df[df.treatment == 1].y1.mean() - df[df.treatment == 1].y0.mean()
    
    def unadjusted(self, df):
        df[df.treatment == 1].outcome.mean() - df[df.treatment == 0].outcome.mean()

if __name__ == '__main__':
    if not os.path.exists('logs'):
        os.makedirs('logs')
    # create folder with name args.exp_name
    if not os.path.exists('logs/'+args.exp_name):
        os.makedirs('logs/'+args.exp_name)
    exp_dir = 'logs/'+args.exp_name
    set_logger(os.path.join(exp_dir, 'log.txt'))
    
    logging.info('Experiment directory: %s', exp_dir)
    logging.info('Arguments: %s', args)
    logging.info('CUDA: %s', CUDA)
    logging.info('Mask index: %s', MASK_IDX)

    # load data
    logging.info('Loading data... at %s', args.data_dir)
    
    df = pd.read_csv(args.data_dir)
    cb = CausalBertWrapper(batch_size=args.batch_size,
        g_weight=args.g_weight, Q_weight=args.g_weight, mlm_weight=args.mlm_weight)
    logging.info(df.T)

    # This trainer sucks, but it's a start
    cb.train(df['title'], df['treatment'], df['outcome'], learning_rate=args.learning_rate, epochs=args.num_train_epochs)
    
    logging.info("ATE")
    logging.info(cb.ATE( df.title, platt_scaling=True))
    
    logging.info("Ground Truth")
    
    # very specific to preprocessing and dataformate 
    gt = cb.gt(df)
    logging.info("ground truth", gt)

    naive = cb.unadjusted(df)
    logging.info("Unadjusted", naive)