# PGM_project
Repo for PGM project

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
                    