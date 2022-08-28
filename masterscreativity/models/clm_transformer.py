import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LinearLR


class CLMTransformer(pl.LightningModule):

    def __init__(self,
            lr=5e-5,
            warmup_steps=100,
            model_name='gpt2',
            model_save_path='gpt2.pkl',
            max_seq_len=512,
            model_load_path=''
    
    ) -> None:
        super().__init__()

        self.lr = lr
        self.warmup_steps = warmup_steps
        self.model_name = model_name
        self.model_save_path = model_save_path
        self.max_seq_len = max_seq_len

        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if model_load_path != '':
            self.model.load_state_dict(torch.load(model_load_path, map_location='cuda'))

    def _step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs[0]

        # print('\n\n_step\nbatch:')
        # print(batch)
        # print('\n\noutputs')
        # print(outputs)
        # print('\n\nloss')
        # print(loss)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log("train_loss", loss)
        return loss

    # TODO: Add additional metrics 
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self._step(batch, batch_idx)
        self.log("val_loss", loss, sync_dist=True)

    def generate(self, text, **kwargs):
        inputs = self.tokenizer(text, return_tensors="pt")
        return self.model.generate(inputs["input_ids"], **kwargs)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        # TODO: Change scheduler (cosine?) and probably training/warm up steps
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.trainer.estimated_stepping_batches
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }