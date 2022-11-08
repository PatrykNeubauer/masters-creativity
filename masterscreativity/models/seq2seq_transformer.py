import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup
import pytorch_lightning as pl


class Seq2SeqTransformer(pl.LightningModule):
    """
    Class for seq2seq (sequence-to-sequence) models e.g. T5.
    """
    def __init__(self,
            lr=3e-4,
            warmup_steps=100,
            model_name='allegro/plt5-base',
            model_save_path='plt5.pkl',
            model_load_path=''
    
    ) -> None:
        super().__init__()

        self.lr = lr
        self.warmup_steps = warmup_steps
        self.model_name = model_name
        self.model_save_path = model_save_path

        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if model_load_path not in ['', None]:
            print("Loading model...")
            self.model.load_state_dict(torch.load(model_load_path, map_location="cuda"))

    def _step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
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
        inputs = self.tokenizer(text, return_tensors="pt")
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