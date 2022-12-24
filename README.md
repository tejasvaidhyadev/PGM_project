# PGM_project
Repo for PGM project

## Abstract
Adjusting for potential confounding in the text, such as the specific concept, topic, or writing style is challenging. In this paper, we design a method to estimate causal effects from observational text data. Since text data is very high-dimensional, one common approach is to find lower-dimensional representations or embeddings which make computations easier and at the same time preserve
vital information relevant to causal inference. For certain casual estimation tasks, we can assume an underlying causal structure that allows us to make accurate predictions and also help identify the effect of specific treatments. In this paper, we extend Veitch et al. (2019b)’s work and analyze the sufficient causal text embeddings that preserve causal information and help in the estimation of a causal effect. Using semi-synthetic datasets, we investigated how causally sufficient embeddings impact causal estimation compared to other embedding methods. The results of our study support the conclusion that causally sufficient embeddings can enhance the accuracy of causal estimation, which is consistent with the findings of previous research.

## Instruction to run the code
```
python trainer.py # use args parse to update params
```

Here are params list

`-- exp_name`: experiment  
`--data_dir`: The input training data file (a csv file)  
`--model_card`: The model card to use.  
`--batch_size`: Batch size for training.  
`--learning_rate`: The initial learning rate for Adam.  
`--num_train_epochs`: "Total number of training epochs to perform."   
`--output_dir`: The output directory where the model predictions and checkpoints will be written.  
`--Q_weigth`: Weigthing of Q term in loss  

`--g_weight`: Weigthing of g term in loss  

`--mlm_weight`: Weigthing of g term in loss  
                    