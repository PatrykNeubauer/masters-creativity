import pytorch_lightning as pl
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk
import string
from transformers import DataCollatorForLanguageModeling, DataCollatorForSeq2Seq


class CSVDatamodule(pl.LightningDataModule):
    """
    Datamodule class for .csv datasets.
    Creates datasets, performs tokenization and splitting. 
    It has different modes for different types of models:
        - "clm" - for Casual Language Modeling models (e.g. GPT-2), that reads "id" and "text", while tokenizing the latter.
        - "seq2seq" - for seq2seq models (e.g. T5), that reads "id", "text" and "target", while tokenizing the latter two.

    TODO: 
        - mode for BERT-like models
        - Split max_length into max_source_len, max_target_len
            - max_source_len only on CLM and BERT; both on seq2seq models
        - Utilize padding (read up which models prefer which)
    """
    def __init__(self,
            model_name,
            batch_size,
            data_paths,
            mode,
            train_val_ratio=0.9,
            prefix='',
            suffix='',
            padding="max_length",
            truncation="only_first",
            max_length=512,
            num_workers=0
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.data_paths = data_paths

        self.mode = mode.lower()
        assert self.mode in ['clm', 'seq2seq'], 'Only "clm" and "seq2seq" modes are supported.'

        self.train_val_ratio = train_val_ratio
        self.prefix = prefix
        self.suffix = suffix
        self.padding = padding
        self.truncation = truncation
        self.max_length = max_length
        self.num_workers = num_workers
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def prepare_data(self):
        """
        This function:
            1) Loads .csv data.
            2) Cleans the text.
            3) Adds prefixes/suffixes.
            4) Tokenizes the text (and target in case of seq2seq).
            5) Saves it to disk (caching).
        """
        dataset = load_dataset("csv", data_files=self.data_paths)
        dataset = dataset.map(self.clean_function)

        if self.prefix != '':
            dataset = dataset.map(self.add_prefix_function)
        if self.suffix != '':
            dataset = dataset.map(self.add_suffix_function)

        dataset = dataset.map(self.tokenize_function, batched=False)
        # TODO: more transforms?
        if self.mode == 'clm':
            dataset = dataset.map(remove_columns=['id', 'text'])
        elif self.mode =='seq2seq':
            dataset = dataset.map(self.clean_function, fn_kwargs={'text_column_name': 'target'})
            dataset = dataset.map(
                self.tokenize_function,
                batched=False,
                fn_kwargs={
                    "text_column_name": "target",
                    "input_ids_column_name": "labels",
                    "attention_mask_column_name": None,
                },
            )
            dataset = dataset.map(remove_columns=['id', 'text', 'target'])

        dataset.save_to_disk('tokenized_dataset')

    def setup(self, stage):
        """
        This function:
            1) Loads the already tokenized dataset saved in prepare_data.
            2) Sets the correct format depending on the mode.
            3) Splits it into train/test splits.
        """
        dataset = load_from_disk('tokenized_dataset')
        if self.mode == 'clm':
            dataset.set_format("pt", columns=['input_ids', 'attention_mask'], output_all_columns=False)
        elif self.mode =='seq2seq':
            dataset.set_format("pt", columns=['input_ids', 'attention_mask', 'labels'], output_all_columns=False)
        
        dataset = dataset['train'].train_test_split(train_size=self.train_val_ratio)
        self.train_dataset = dataset['train']
        self.val_dataset = dataset['test']

    def tokenize_function(
        self,
        examples,
        text_column_name='text',
        input_ids_column_name='input_ids',
        attention_mask_column_name='attention_mask'
        ):
        tokenized = self.tokenizer(examples[text_column_name], truncation=self.truncation, max_length=self.max_length)
        if input_ids_column_name != None:
            examples[input_ids_column_name] = tokenized['input_ids']
        if attention_mask_column_name != None:
            examples[attention_mask_column_name] = tokenized['attention_mask']
        return examples

    @staticmethod
    def clean_function(
        examples,
        text_column_name='text'
        ):
        examples[text_column_name] = examples[text_column_name].rstrip(string.punctuation + ' ' + '\n').replace('\\n', '\n')
        return examples
    
    def add_prefix_function(
        self,
        examples,
        text_column_name='text'
        ):
        examples[text_column_name] = self.prefix + examples[text_column_name]
        return examples

    def add_suffix_function(
        self,
        examples,
        text_column_name='text'
        ):
        examples[text_column_name] = examples[text_column_name] + self.suffix
        return examples

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)

    @property
    def collate_fn(self):
        if self.mode == 'clm':
            self.tokenizer.pad_token = self.tokenizer.eos_token
            return DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        elif self.mode =='seq2seq':
            return DataCollatorForSeq2Seq(self.tokenizer)