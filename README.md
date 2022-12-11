# PGM_project
Repo for PGM project

## Instruction to run the code
```
python trainer.py # use args parse to update params
```

Here are params list

-- exp_name: experiment
--data_dir", default="./PeerRead/process_data/beta0_0.25beta1_5.0gamma_0.0.csv", type=str, required=False, "The input training data file (a csv file)

--model_card", default="bert-base-uncased", type=str, required=False,
                    help="The model card to use.")
--batch_size", default=8, type=int, required=False,
                    help="Batch size for training.")

parser.add_argument("--learning_rate", default=2e-5, type=float, required=False,
                    help="The initial learning rate for Adam.")
parser.add_argument("--num_train_epochs", default=5, type=float, required=False,
                    help="Total number of training epochs to perform.")
parser.add_argument("--warmup_steps", default=0, type=int, required=False,
                    help="Linear warmup over warmup_steps.")    
parser.add_argument("--output_dir", default="output", type=str, required=False,
                    help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--Q_weigth", default=0.1, type=float, required=False,
                    help="Weigthing of Q term in loss")
parser.add_argument("--g_weight", default=1, type=float, required=False,
                    help="Weigthing of g term in loss")
parser.add_argument("--mlm_weight", default=1.0, type=float, required=False,
                    help="Weigthing of g term in loss")
                    
## ToDO
- Refactor code to make it more readable

## ToDo Experiments
- All the experiments in the paper #only Bert and causal Bert is added
- Add more experiments with different language models
- Ablation study on the model
- sythetic data experiments

Paper Link: 
- [Main Paper](http://proceedings.mlr.press/v124/veitch20a/veitch20a.pdf)
- [Sythetic Data Paper](https://github.com/QuantLet/DataGenerationForCausalInference)
- [NLP Sythetic Data Paper](https://arxiv.org/pdf/2102.05638.pdf). Also, interesting idea