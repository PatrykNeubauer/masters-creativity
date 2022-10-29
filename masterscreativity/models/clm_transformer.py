import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LinearLR


class CLMTransformer(pl.LightningModule):

    def __init__(self,
            lr=5e-5,
            warmup_steps=100,
            model_name='flax-community/papuGaPT2',
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

        if model_load_path not in ['', None]:
            print("Loading model...")
            self.model.load_state_dict(torch.load(model_load_path, map_location="cuda"))


    def _step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs[0]

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log("train_loss", loss)
        return loss

    def training_epoch_end(self, outputs):
        if self.trainer.global_step > 0:
            print("Saving model...")
            torch.save(self.model.state_dict(), self.model_save_path)

    # TODO: Add additional metrics 
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self._step(batch, batch_idx)
        self.log("val_loss", loss, sync_dist=True)

    def generate(self, text, **kwargs):
        inputs = self.tokenizer(text, max_length=self.max_seq_len, return_tensors="pt")
        return self.model.generate(inputs["input_ids"].to('cuda:0'), **kwargs) # TODO: Generalize device

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