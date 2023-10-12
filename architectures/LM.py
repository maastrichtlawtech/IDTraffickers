"""
Python version: 3.9
Description: Contains model architecture of the language model
"""

# %% Importing Libraries
import torch
import lightning.pytorch as pl

from deepspeed.ops.adam import DeepSpeedCPUAdam
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoConfig

class LMModel(pl.LightningModule):
    def __init__(self, model_name_or_path, learning_rate, warmup_steps, nr_training_steps, decay, adam_epsilon, mode="training"):
        super().__init__()

        self.save_hyperparameters()

        config = AutoConfig.from_pretrained(model_name_or_path, return_dict=True)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name_or_path, config=config)

        self.hparams.warmup_steps = learning_rate
        self.hparams.warmup_steps = warmup_steps
        self.hparams.num_training_steps = nr_training_steps
        self.hparams.weight_decay = decay
        self.hparams.eps = adam_epsilon

    def forward(self, x, mode="training"):
        if mode == "training":
            return self.model(x).logits
        else:
            return self.model(x).hidden_states

    def training_step(self, batch, batch_idx):
        loss = self.model(**batch).loss
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.model(**batch).loss
        # calculating perplexity
        perplexity  = torch.exp(loss)
        self.log_dict({"valid_loss": loss, 'perplexity': perplexity}, on_step=True, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        # The configure_optimizers is a (virtual) method, specified in the interface, that the
        # pl.LightningModule class wants you to overwrite.

        # In this case we define that some parameters are optimized in a different way than others. In
        # particular we single out parameters that have 'bias', 'LayerNorm.weight' in their names. For those
        # we do not use an optimization technique called weight decay.

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [{'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay':self.hparams.weight_decay}, 
                                        {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        # optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.eps)
        optimizer = DeepSpeedCPUAdam(optimizer_grouped_parameters, adamw_mode=True, lr=self.hparams.learning_rate, betas=(0.9, 0.999), eps=self.hparams.eps)

        # We also use a scheduler that is supplied by transformers.
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.hparams.num_training_steps)
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]