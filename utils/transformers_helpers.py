import shutil
import os
import re
import glob
import torch
import random
import numpy as np
import urllib

def rotate_checkpoints(output_dir, checkpoint_prefix='checkpoint', use_mtime=False, save_total_limit=3):
    def _sorted_checkpoints():
        ordering_and_checkpoint_path = []
        glob_checkpoints = glob.glob(os.path.join(output_dir, "{}-*".format(checkpoint_prefix)))
        for path in glob_checkpoints:
            if use_mtime:
                ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
            else:
                regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
                if regex_match and regex_match.groups():
                    ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))
        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
        return checkpoints_sorted
    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints()
    if len(checkpoints_sorted) <= save_total_limit:
        return
    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)

def mask_tokens(inputs, tokenizer, mlm_probability=0.15):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    if tokenizer.mask_token is None:
        raise ValueError("This tokenizer does not have a mask token which is necessary for masked language modeling.")
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels

def set_seed(seed, no_cuda=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if not no_cuda:
        torch.cuda.manual_seed_all(seed)

def download_vocab_files_for_tokenizer(tokenizer, model_type, output_path):
    vocab_files_map = tokenizer.pretrained_vocab_files_map
    vocab_files = {}
    for resource in vocab_files_map.keys():
        download_location = vocab_files_map[resource][model_type]
        f_path = os.path.join(output_path, os.path.basename(download_location))
        urllib.request.urlretrieve(download_location, f_path)
        vocab_files[resource] = f_path
    return vocab_files
