import pytorch_lightning as pl
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk, Dataset
import string
from transformers import DataCollatorForLanguageModeling


class CSVDatamodule(pl.LightningDataModule):

    def __init__(self,
            model_name,
            batch_size,
            data_paths,
            train_val_ratio=0.9,
            padding="max_length",
            truncation="only_first",
            max_length=512,
            num_workers=0
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.data_paths = data_paths
        self.train_val_ratio = train_val_ratio
        self.padding = padding
        self.truncation = truncation
        self.max_length = max_length
        self.num_workers = num_workers
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def prepare_data(self):
        dataset = load_dataset("csv", data_files=self.data_paths)
        dataset = dataset.map(self.clean_function)
        # more transforms?
        dataset = dataset.map(self.tokenize_function, batched=False)
        dataset = dataset.map(remove_columns=['id', 'text'])
        # dataset.set_format("pt", columns=['input_ids', 'attention_mask'], output_all_columns=True)
        
        
        print('\n\nDataset sample (prepare data): ')  # TODO: Delete
        print(dataset['train'][0])

        dataset.save_to_disk('tokenized_dataset')
        # return ?

    def setup(self, stage):
        dataset = load_from_disk('tokenized_dataset')
        dataset.set_format("pt", columns=['input_ids', 'attention_mask'], output_all_columns=True)
        dataset = dataset['train'].train_test_split(train_size=self.train_val_ratio)
        self.train_dataset = dataset['train']
        self.val_dataset = dataset['test']
        # transforms?
        # return ?

    def tokenize_function(
        self,
        examples,
        text_column_name='text'
        ):
        tokenized = self.tokenizer(examples[text_column_name], truncation=self.truncation, max_length=self.max_length)
        return tokenized

    @staticmethod
    def clean_function(
        examples,
        text_column_name='text'
        ):
        examples[text_column_name] = examples[text_column_name].rstrip(string.punctuation + ' ' + '\n')
        return examples

    def train_dataloader(self):
        # print('\n\nDataset sample (train data - train dataloader): ') # TODO: Delete
        # print(self.train_dataset[0])

        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)

    def val_dataloader(self):
        # print('\n\nDataset sample (val data - val dataloader): ') # TODO: Delete
        # print(self.val_dataset[0])

        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)

    @property
    def collate_fn(self):
        self.tokenizer.pad_token = self.tokenizer.eos_token
        return DataCollatorForLanguageModeling(self.tokenizer, mlm=False)