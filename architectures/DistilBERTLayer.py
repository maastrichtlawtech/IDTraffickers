""" 
Python version: 3.9
Description: Contains helper classes and functions to load the DistilBERT layer into the LightningModule.
"""

# %% Importing libraries
from pytorch_lightning import LightningModule, Trainer

from pytorch_metric_learning import losses
from pytorch_metric_learning.losses import SelfSupervisedLoss

from transformers.models.distilbert.modeling_distilbert import DistilBertPreTrainedModel, DistilBertModel
from transformers import AdamW, get_linear_schedule_with_warmup

# Custom library
from layerUtilities import cl_init, sentemb_forward, cl_forward

# %% Helper classes
class DistilBERTforCL(DistilBertPreTrainedModel, LightningModule):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, temp, warmup_steps, adam_epsilon, weight_decay, learning_rate, pooler_type):
        super().__init__(config)
        self.distilbert = DistilBertModel(config)
        self.warmup_steps = warmup_steps
        self.adam_epsilon = adam_epsilon
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_fn = SelfSupervisedLoss(losses.NTXentLoss(temperature=temp))
        self.pooler_type = pooler_type
        cl_init(self, pooler_type, config, temp)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None,
                output_attentions=None, output_hidden_states=None, return_dict=None, sent_emb=False):

        if sent_emb:
            return sentemb_forward(self, self.distilbert, input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids,
                                    head_mask=head_mask, inputs_embeds=inputs_embeds, labels=labels, output_attentions=output_attentions,
                                    output_hidden_states=output_hidden_states, return_dict=return_dict)
        else:
            return cl_forward(self, self.distilbert, self.pooler_type, input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, 
                                head_mask=head_mask, inputs_embeds=inputs_embeds, labels=labels, output_attentions=output_attentions, 
                                output_hidden_states=output_hidden_states, return_dict=return_dict)

    def training_step(self, batch, batch_idx):
        # Processing the anchors from the batch
        input_ids = batch['input_ids'][:, 0, :]
        attention_mask = batch['attention_mask'][:, 0, :]
        
        anchor_embeddings = self(input_ids, attention_mask=attention_mask)
        
        # Processing the positives from the batch
        input_ids = batch['input_ids'][:, 1, :]
        attention_mask = batch['attention_mask'][:, 1, :]
        
        positive_embeddings = self(input_ids, attention_mask=attention_mask)

        loss = self.loss_fn(anchor_embeddings, positive_embeddings)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Processing the anchors from the batch
        input_ids = batch['input_ids'][:, 0, :]
        attention_mask = batch['attention_mask'][:, 0, :]
        
        anchor_embeddings = self(input_ids, attention_mask=attention_mask)
        
        # Processing the positives from the batch
        input_ids = batch['input_ids'][:, 1, :]
        attention_mask = batch['attention_mask'][:, 1, :]
        
        positive_embeddings = self(input_ids, attention_mask=attention_mask)

        val_loss = self.loss_fn(anchor_embeddings, positive_embeddings)
        self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return val_loss

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.distilbert
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate, eps=self.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]