from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer
import torch 
from collections import defaultdict
CUDA = (torch.cuda.device_count() > 0)
MASK_IDX = 103


def build_dataloader(model_card, batch_size, texts, treatments = None, outcomes = None, tokenizer= None, sampler = 'random'):
    """
    Build dataloader for text confound treatment and outcome
    Args:
        model_card: huggingface model card
        batch_size: batch size
        texts: list of text
        treatments: list of treatment
        outcomes: list of outcome
        tokenizer: huggingface tokenizer
        sampler: random or sequential
    Returns:
        dataloader
    """
    def collate_Confound_and_treatment(batch):
        "Puts each data field into a tensor with outer dimension batch size"
        texts, treatments = zip(*batch)
        return texts, treatments

    if treatments is None:
        "If treatment is not provided, replace treatment with -1"
        # replace treatment with -1
        treatments = [-1] * len(texts)
    if outcomes is None:
        # replace outcome with -1
        outcomes = [-1] * len(texts)
    if tokenizer is None:
        # use huggingface AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_card)
    
    # store tokenized text confound treatment and outcome in a single default dict
    data_dict = defaultdict(list)
    for text,  treatment, outcome in zip(texts, treatments, outcomes):
        # convert encode plus special tokens
        # encode_plus will return a dict with keys input_ids, token_type_ids, attention_mask
        tokenized_text = tokenizer.encode_plus(text, add_special_tokens=True, max_length=512, pad_to_max_length=True)
        # add tokenized text to data_dict
        data_dict['W_ids'].append(tokenized_text['input_ids'])
        data_dict['W_mask'].append(tokenized_text['attention_mask'])
        data_dict['W_len'].append(sum(tokenized_text['attention_mask']))
        #data_dict['text'].append(tokenized_text)
        data_dict['treatment'].append(treatment)
        data_dict['outcome'].append(outcome)
    
    # convert to tensor dataset
    # data_dict to tensor 
    for key in data_dict.keys():
        data_dict[key] = torch.tensor(data_dict[key])
    # create tensor dataset
    tensor_dataset = TensorDataset(data_dict['W_ids'], data_dict['W_len'], data_dict['W_mask'], data_dict['treatment'], data_dict['outcome'])
    # create sampler
    if sampler == 'random':
        sampler = RandomSampler(tensor_dataset)
    elif sampler == 'sequential':
        sampler = SequentialSampler(tensor_dataset)
    else:
        raise ValueError("Sampler should be either random or sequential")
    # create dataloader
    dataloader = DataLoader(tensor_dataset, sampler = sampler, batch_size = batch_size)
    return dataloader