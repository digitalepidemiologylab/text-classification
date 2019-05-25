from models.base_model import BaseModel
from pytorch_pretrained_bert.modeling_gpt2 import GPT2PreTrainedModel, GPT2Model, GPT2LMHeadModel, GPT2Config, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization_gpt2 import GPT2Tokenizer
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import os
from tqdm import trange, tqdm
import pandas as pd
import torch
import numpy as np


class OpenAIGPT2(BaseModel):
    def __init__(self):
        super().__init__()
        self.tokenizer = None
        self.model = None

    def _setup_gpt2(self, config, setup_mode='train'):
        # model params
        self.train_batch_size = config.get('train_batch_size', 8)
        self.test_batch_size = config.get('test_batch_size', 8)
        self.learning_rate = config.get('learning_rate', 1e-4)
        self.num_epochs = int(config.get('num_epochs', 3))

        # paths
        self.model_path = os.path.join(config.other_path, 'openai_gpt2')
        self.output_path = config.output_path
        self.train_data = config.train_data
        self.test_data = config.test_data
        if config.fine_tune_name:
            self.fine_tune_path = config.get('fine_tuned_model_path', os.path.join(config.other_path, 'fine_tune', 'bert', config.fine_tune_name))
        else:
            self.fine_tune_path = None

        # GPU device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # seed
        self.seed = config.get('seed', np.random.randint(1e4))
        torch.random.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

        # label mapping
        if setup_mode in ['test', 'predict']:
            self.label_mapping = self.get_label_mapping(config)
        else:
            self.label_mapping = self.set_label_mapping(config)
        num_labels = len(self.label_mapping)

        # load model
        model_type = 'gpt2'
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_type, cache_dir=self.model_path)
        if setup_mode == 'train':
            self.model = GPT2Classifier.from_pretrained(model_type, cache_dir=self.model_path)
        else:
            config = GPT2Config(os.path.join(self.output_path, CONFIG_NAME))
            weights = torch.load(os.path.join(self.output_path, WEIGHTS_NAME))
            self.model = GPT2Classifier.from_pretrained(model_type, cache_dir=self.model_path, num_labels=num_labels, state_dict=weights.state_dict())
        self.model.to(self.device)


    def train(self, config):
        # Init params and models
        self._setup_gpt2(config, setup_mode='train')
        # Load data
        encode_transform = Encoder(self.tokenizer, self.label_mapping, device=self.device, max_size=self.model.config.n_positions)
        dataset = GPT2Dataset(self.train_data, transform=encode_transform)
        dataloader = DataLoader(dataset, batch_size=self.train_batch_size, shuffle=True)
        # Freeze all layers except linear layer
        for n, param in self.model.named_parameters(): 
            if 'linear' not in n:
                param.requires_grad = False
        # Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # Loss function
        criterion = torch.nn.CrossEntropyLoss()
        # Train
        for i in range(self.num_epochs):
            print("Running epoch {}/{}...".format(i + 1, self.num_epochs))
            pbar = tqdm(dataloader)
            for x, y in pbar:
                logits = self.model(x)
                loss = criterion(logits, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.set_description("Loss: {:8.4f}".format(loss))
        # Save model
        output_model_file = os.path.join(self.output_path, WEIGHTS_NAME)
        output_config_file = os.path.join(self.output_path, CONFIG_NAME)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        torch.save(model_to_save, output_model_file)
        with open(output_config_file, 'w') as f:
            f.write(self.model.config.to_json_string())

    def test(self, config):
        self._setup_gpt2(config, setup_mode='test')
        # Load test data
        encode_transform = Encoder(self.tokenizer, self.label_mapping, device=self.device, max_size=self.model.config.n_positions)
        test_data = GPT2Dataset(self.test_data, transform=encode_transform)
        dataloader = DataLoader(test_data, batch_size=self.test_batch_size)
        result = {'prediction': [], 'label': []}
        for x, y in tqdm(dataloader):
            with torch.no_grad():
                logits = self.model(x)
                logits = logits.detach().cpu()
                result['prediction'].extend(np.argmax(logits, axis=1).tolist())
                result['label'].extend(y.cpu().tolist())
        result_out = self.performance_metrics(result['label'], result['prediction'], label_mapping=self.label_mapping)
        if config.write_test_output:
            test_output = self.get_full_test_output(result['prediction'], result['label'], label_mapping=self.label_mapping, test_data_path=self.test_data)
            result_out = {**result_out, **test_output}
        return result_out

    def predict(self, config, data):
        self._setup_gpt2(config, setup_mode='predict')
        # Write to disk
        f_path = os.path.join('.', 'tmp', 'predict.csv')
        pd.DataFrame({'text': data}).to_csv(f_path)
        # Transform to dataloader objects
        predict_data = GPT2Predict(f_path, transform=EncoderPredict(self.tokenizer, self.label_mapping, device=self.device, max_size=self.model.config.n_positions))
        dataloader = DataLoader(predict_data, batch_size=self.test_batch_size)
        predictions = []
        for x in tqdm(dataloader):
            with torch.no_grad():
                logits = self.model(x)
                probabilities = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
                probabilities = self.format_predictions(probabilities, label_mapping=self.label_mapping)
                predictions.extend(probabilities)
        return predictions 

    def generate_text(self, seed, config):
        def top_k_logits(logits, k):
            if k == 0:
                return logits
            values, _ = torch.topk(logits, k)
            min_values = values[:, -1]
            return torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits)
        self._setup_generate_text(config)
        # generate output
        context = self.tokenizer.encode(seed)
        length = self.model.config.n_ctx // 2
        context = torch.tensor(context, device=self.device, dtype=torch.long).unsqueeze(0).repeat(self.generate_batch_size, 1)
        prev = context
        output = context
        past = None
        with torch.no_grad():
            for i in tqdm(range(length)):
                logits, past = self.model(prev, past=past)
                logits = logits[:, -1, :] / self.temperature
                logits = top_k_logits(logits, k=self.top_k)
                log_probs = torch.nn.functional.softmax(logits, dim=-1)
                prev = torch.multinomial(log_probs, num_samples=1)
                output = torch.cat((output, prev), dim=1)
        output = output[:, len(context):].tolist()
        text = self.tokenizer.decode(output[0])
        return text

    def _setup_generate_text(self, config):
        # Init params
        self.generate_batch_size = int(config.get('generate_batch_size', 8))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.temperature = float(config.get('temperature', 0.9))
        self.top_k = int(config.get('top_k', 0))
        self.model_path = os.path.join(config.other_path, 'openai_gpt2')
        # seed
        self.seed = config.get('seed', np.random.randint(1e4))
        torch.random.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        # load model
        model_type = 'gpt2'
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_type, cache_dir=self.model_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_type, cache_dir=self.model_path)
        self.model.to(self.device)
        self.model.eval()


class GPT2Classifier(GPT2PreTrainedModel):
    def __init__(self, config, num_labels=2):
        super(GPT2Classifier, self).__init__(config)
        self.transformer = GPT2Model(config)
        self.linear = torch.nn.Linear(config.n_embd*config.n_ctx, num_labels)
        # initialize linear layer
        torch.nn.init.normal_(self.linear.weight, std=0.02)
        torch.nn.init.normal_(self.linear.bias, 0)

    def forward(self, input_ids, token_type_ids=None, position_ids=None, past=None):
        hidden_states, presents = self.transformer(input_ids, position_ids, token_type_ids, past)
        logits = self.linear(hidden_states.reshape(len(hidden_states), -1))
        return logits


class GPT2Dataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path, engine='python')
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        sample = self.df.iloc[i][['text', 'label']].to_dict()
        if self.transform:
            sample = self.transform(sample)
        return sample

class GPT2Predict(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path, engine='python')
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        sample = self.df.iloc[i][['text']].to_dict()
        if self.transform:
            sample = self.transform(sample)
        return sample

class Encoder:
    def __init__(self, tokenizer, label_mapping, device=None, max_size=768):
        self.tokenizer = tokenizer
        self.label_mapping = label_mapping
        self.device = device
        self.max_size = max_size

    def __call__(self, sample):
        text_encoded = torch.zeros(self.max_size, dtype=torch.int64, device=self.device)
        tokenized = self.tokenizer.encode(sample['text'])
        text_encoded[:len(tokenized)] = torch.tensor(tokenized, dtype=torch.int64, device=self.device)
        label = torch.tensor(self.label_mapping[sample['label']], dtype=torch.int64).to(self.device)
        return text_encoded, label

    def make_one_hot(self, label):
        t = torch.eye(len(self.label_mapping))
        return t[self.label_mapping[label]]

class EncoderPredict:
    def __init__(self, tokenizer, label_mapping, device=None, max_size=768):
        self.tokenizer = tokenizer
        self.label_mapping = label_mapping
        self.device = device
        self.max_size = max_size

    def __call__(self, sample):
        text_encoded = torch.zeros(self.max_size, dtype=torch.int64, device=self.device)
        tokenized = self.tokenizer.encode(sample['text'])
        text_encoded[:len(tokenized)] = torch.tensor(tokenized, dtype=torch.int64, device=self.device)
        return text_encoded

    def make_one_hot(self, label):
        t = torch.eye(len(self.label_mapping))
        return t[self.label_mapping[label]]
