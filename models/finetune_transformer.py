from utils.misc import suppress_stdout, get_file_md5
from models.base_model import BaseModel
from utils.transformers_helpers import mask_tokens, rotate_checkpoints, set_seed
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset
from tqdm import tqdm, trange
import pandas as pd
import logging
import os
import numpy as np
import torch
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from tokenizers import BertWordPieceTokenizer
import urllib
import itertools

logger = logging.getLogger(__name__)


class FinetuneTransformer(BaseModel):
    def __init__(self):
        super().__init__()

    def init(self, config):
        # Paths
        self.name = config.name
        self.train_data = config.train_data
        self.test_data = config.get('test_data', None)
        self.other_path = config.other_path
        self.output_path = config.output_path
        self.tmp_path = config.tmp_path
        self.model_name = config.get('model', 'bert')
        self.model_type = config.get('model_type', 'bert-base-uncased')
        self.model_path = os.path.join(self.other_path, self.model_name)
        self.overwrite = config.get('overwrite', False)
        self.load_data_into_memory = config.get('load_data_into_memory', False)
        self.num_workers_batch_loading = config.get('num_workers_batch_loading', 50) # only used when load_data_into_memory is False
        self.evaluate_during_training = config.get('evaluate_during_training', True)

        # config
        self.mlm = self.model_name in ["bert", "roberta", "distilbert", "camembert"]  # is masked LM
        self.mlm_probability = config.get('mlm_probability', 0.15)
        self.save_steps = config.get('save_steps', 500)  # save every n steps
        self.num_checkpoints = config.get('num_checkpoints', 1)  # save n last checkpoints (set to 1 due to large model files)

        # hyperparams
        self.max_seq_length = config.get('max_seq_length', 128)
        self.train_batch_size = config.get('train_batch_size', 16)
        self.test_batch_size = config.get('test_batch_size', 16)
        self.learning_rate = config.get('learning_rate', 5e-5)
        self.num_epochs = config.get('num_epochs', 1)
        self.warmup_steps = config.get('warmup_steps', 100)
        self.max_train_steps = config.get('max_train_steps', None)
        self.no_cuda = config.get('no_cuda', False)
        self.on_memory = config.get('on_memory', True)
        self.do_lower_case = 'uncased' in self.model_type
        self.local_rank = config.get('local_rank', -1)
        self.seed = config.get('seed', np.random.randint(1e4))
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
        self.fp16 = config.get('fp16', False)
        self.loss_scale = config.get('loss_scale', 0.0)

        # set seed
        set_seed(self.seed, no_cuda=self.no_cuda)

        # GPU config
        if self.local_rank == -1 or self.no_cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() and not self.no_cuda else "cpu")
            self.n_gpu = torch.cuda.device_count()
        else:
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
            self.n_gpu = 1
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.distributed.init_process_group(backend='nccl')
        if self.no_cuda:
            self.n_gpu = 0
        logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(self.device, self.n_gpu, bool(self.local_rank != -1), self.fp16))
        if self.gradient_accumulation_steps < 1:
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(self.gradient_accumulation_steps))
        self.train_batch_size = self.train_batch_size // self.gradient_accumulation_steps

    def download_vocab_files_for_tokenizer(self, tokenizer):
        vocab_files_map = tokenizer.pretrained_vocab_files_map
        vocab_files = {}
        for resource in vocab_files_map.keys():
            download_location = vocab_files_map[resource][self.model_type]
            f_path = os.path.join(self.output_path, os.path.basename(download_location))
            urllib.request.urlretrieve(download_location, f_path)
            vocab_files[resource] = f_path
        return vocab_files

    def train(self):
        # Model initialization
        tokenizer = AutoTokenizer.from_pretrained(self.model_type, cache_dir=self.model_path)
        vocab_files = self.download_vocab_files_for_tokenizer(tokenizer)
        fast_tokenizer = BertWordPieceTokenizer(vocab_files.get('vocab_file'), vocab_files.get('merges_file'), lowercase=self.do_lower_case)
        fast_tokenizer.enable_padding(max_length=self.max_seq_length)
        num_train_optimization_steps = None
        logger.debug(f'Loading Train Dataset {self.train_data}...')
        if self.load_data_into_memory:
            train_dataset = TextDataset(self.train_data, fast_tokenizer, max_seq_length=self.max_seq_length)
        else:
            train_dataset = TextIterableDataset(self.train_data, fast_tokenizer, max_seq_length=self.max_seq_length)
        logger.info(f'Loaded {len(train_dataset):,} examples...')
        num_train_optimization_steps = int(len(train_dataset) / self.train_batch_size / self.gradient_accumulation_steps) * self.num_epochs
        if self.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()
        config = AutoConfig.from_pretrained(self.model_type, cache_dir=self.model_path)
        model = AutoModelWithLMHead.from_pretrained(
            self.model_type,
            from_tf=bool(".ckpt" in self.model_type),
            config=config,
            cache_dir=self.model_path)

        if self.fp16:
            model.half()
        model.to(self.device)
        if self.local_rank != -1:
            try:
                from apex.parallel import DistributedDataParallel as DDP
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
            model = DDP(model)
        elif self.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Prepare optimizer
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        if self.fp16:
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=self.learning_rate,
                                  bias_correction=False,
                                  max_grad_norm=1.0)
            if self.loss_scale == 0:
                optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                optimizer = FP16_Optimizer(optimizer, static_loss_scale=self.loss_scale)
        else:
            optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=num_train_optimization_steps)

        # Run training
        global_step = 0
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Batch size = %d", self.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        if self.load_data_into_memory:
            if self.local_rank == -1:
                train_sampler = RandomSampler(train_dataset)
            else:
                train_sampler = DistributedSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=self.train_batch_size)
            total_batches = len(train_dataloader)
        else:
            # no sampling supported for iterative data loading
            train_dataloader = DataLoader(train_dataset, batch_size=self.train_batch_size, num_workers=self.num_workers_batch_loading)
            # len doesn't work for iterator datasets
            total_batches = len(train_dataloader)/self.train_batch_size
        set_seed(self.seed, no_cuda=self.no_cuda) # Added here for reproducibility
        for _ in trange(int(self.num_epochs), desc="Epoch"):
            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", total=total_batches)):
                inputs, labels = mask_tokens(batch, tokenizer, mlm_probability=self.mlm_probability) if self.mlm else (batch, batch)
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                model.train()
                outputs = model(inputs, masked_lm_labels=labels) if self.mlm else model(inputs, labels=labels)
                loss = outputs[0]
                if self.n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps
                if self.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += labels.size(0)
                nb_tr_steps += 1
                break
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    if self.save_steps > 0 and global_step % self.save_steps == 0:
                        output_dir = os.path.join(self.output_path, f'checkpoint-{global_step}')
                        os.makedirs(output_dir, exist_ok=True)
                        # Take care of distributed/parallel training
                        model_to_save = model.module if hasattr(model, 'module') else model
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)
                        logger.info(f'Saving model checkpoint to {output_dir}')
                        rotate_checkpoints(self.output_path, save_total_limit=self.num_checkpoints)
                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        logger.info(f'Saving optimizer and scheduler states to {output_dir}')
                        if self.evaluate_during_training:
                            logger.info('Evaluate...')
                            self.test(model=model, tokenizer=tokenizer, fast_tokenizer=fast_tokenizer, output_dir=output_dir)
                if self.max_train_steps is not None and step > self.max_train_steps:
                    logger.info(f'Reached max number of training steps {self.max_train_steps:,}')
                    break
                if step >= total_batches:
                    # finished epoch
                    break
            if self.max_train_steps is not None and step > self.max_train_steps:
                break

        # Save a trained model
        logger.info("** ** * Saving fine - tuned model ** ** * ")
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        model_to_save.save_pretrained(self.output_path)
        tokenizer.save_pretrained(self.output_path)
        self.add_to_config(self.output_path, vars(self))

    def test(self, model=None, tokenizer=None, fast_tokenizer=None, output_dir=None):
        if self.test_data is None:
            logger.warning('No test data provided. Aborting.')
            return
        if output_dir is None:
            output_dir = self.output_path
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(output_dir)
        if fast_tokenizer is None:
            vocab_files = self.download_vocab_files_for_tokenizer(tokenizer)
            fast_tokenizer = BertWordPieceTokenizer(vocab_files.get('vocab_file'), vocab_files.get('merges_file'), lowercase=self.do_lower_case)
            fast_tokenizer.enable_padding(max_length=self.max_seq_length)
        logger.debug(f'Loading test dataset {self.test_data}...')
        if self.load_data_into_memory:
            test_dataset = TextDataset(self.test_data, fast_totenizer, max_seq_length=self.max_seq_length)
        else:
            test_dataset = TextIterableDataset(self.test_data, fast_tokenizer, max_seq_length=self.max_seq_length)
        logger.info(f'Loaded {len(test_dataset):,} examples...')
        if model is None:
            config = AutoConfig.from_pretrained(output_dir)
            model = AutoModelWithLMHead.from_pretrained(output_dir, config=config) 
            model.to(self.device)
        if self.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        # evaluate
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(test_dataset))
        logger.info("  Batch size = %d", self.test_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        model.eval()
        if self.load_data_into_memory:
            eval_sampler = SequentialSampler(test_dataset)
            test_dataloader = DataLoader(test_dataset, sampler=eval_sampler, batch_size=self.test_batch_size)
            total_batches = len(test_dataloader)
        else:
            # no sampling supported for iterative data loading
            test_dataloader = DataLoader(test_dataset, batch_size=self.test_batch_size, num_workers=self.num_workers_batch_loading)
            # len doesn't work for iterator datasets
            total_batches = int(len(test_dataloader)/self.test_batch_size)
        for batch in tqdm(test_dataloader, desc="Evaluating", total=total_batches):
            inputs, labels = mask_tokens(batch, tokenizer, mlm_probability=self.mlm_probability) if self.mlm else (batch, batch)
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            with torch.no_grad():
                outputs = model(inputs, masked_lm_labels=labels) if self.mlm else model(inputs, labels=labels)
                lm_loss = outputs[0]
                eval_loss += lm_loss.mean().item()
            nb_eval_steps += 1
            if nb_eval_steps >= total_batches:
                break
        eval_loss = eval_loss / nb_eval_steps
        perplexity = float(torch.exp(torch.tensor(eval_loss)))
        logger.info(f'Eval perplexity: {perplexity}')
        return {'perplexity': perplexity}

    def prepare_input_data(self, min_tokens=3):
        """DEPRECATED: method used for NSP/SOP tasks"""
        import en_core_web_sm
        logger.info('Generating file hash...')
        file_hash = get_file_md5(self.train_data)
        input_data_path = os.path.join(self.tmp_path, 'fine_tune', 'finetune_transformer', f'{file_hash}.txt')
        if os.path.exists(input_data_path):
            logger.info('Found pre-existing input data file.')
            return input_data_path
        if not os.path.isdir(os.path.dirname(input_data_path)):
            os.makedirs(os.path.dirname(input_data_path))
        logger.info('Reading input data...')
        df = pd.read_csv(self.train_data, usecols=['text'])
        nlp = en_core_web_sm.load()
        logger.info('Generating input data...')
        with open(input_data_path, 'w') as f:
            for i, text in tqdm(enumerate(df['text']), total=len(df)):
                sentence_was_found = False
                doc = nlp(text, disable=['entity', 'tagger'])
                sentences = [sent.string.strip() for sent in doc.sents]
                for sentence in sentences:
                    try:
                        num_tokens = len(doc)
                    except:
                        logger.error('error with sentence: "{}"'.format(sentence))
                        continue
                    if num_tokens > min_tokens:
                        f.write(sentence + '\n')
                        sentence_was_found = True
                if sentence_was_found:
                    # add new line after sentences
                    f.write('\n')
        return input_data_path

class TextDataset(Dataset):
    """Load dataset in memory"""
    def __init__(self, file_path, tokenizer, max_seq_length=512):
        assert os.path.isfile(file_path)
        logger.info('Reading file...')
        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
        logger.info('Tokenizing...')
        lines = tokenizer.encode_batch(lines)
        self.examples = [l.ids for l in lines if len(l) <= max_seq_length]
        logger.info('... done')

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)

class TextIterableDataset(IterableDataset):
    """Load dataset iteratively and tokenize with tokenizer library on the fly"""
    def __init__(self, file_path, tokenizer, max_seq_length=512, truncate=True):
        assert os.path.isfile(file_path)
        self.f_name  = file_path
        self.tokenizer = tokenizer
        self.truncate = truncate
        self.max_seq_length = max_seq_length
        self.num_lines = sum(1 for line in open(self.f_name, 'r') if len(line.strip()) > 0)

    def parse_and_tokenize(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # single-process data loading
            with open(self.f_name, 'r') as f:
                for line in f:
                    if len(line.strip()) > 0:
                        encoding = self.tokenizer.encode(line)
                        encoding.truncate(self.max_seq_length)
                        yield torch.tensor(encoding.ids, dtype=torch.long)
        else:
            per_worker = int(self.num_lines/float(worker_info.num_workers))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.num_lines)
            logger.info(f'Start batching worker {worker_id}, start: {iter_start:,}, end: {iter_end:,} (of total {self.num_lines:,} lines)')
            with open(self.f_name, 'r') as f:
                for i, line in enumerate(f):
                    if i > iter_start and i < iter_end:
                        if len(line.strip()) > 0:
                            encoding = self.tokenizer.encode(line)
                            encoding.truncate(self.max_seq_length)
                            yield torch.tensor(encoding.ids, dtype=torch.long)

    def __iter__(self):
        return itertools.cycle(self.parse_and_tokenize())

    def __len__(self):
        return self.num_lines
