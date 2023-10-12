"""
Python version: 3.9
Description: Contains model architecture of transformer-based NNs for performing multi-class classification on the Backpage advertisements.
"""

# %% Importing libraries
from sklearn.metrics import f1_score, balanced_accuracy_score

import torch
import lightning.pytorch as pl

from deepspeed.ops.adam import DeepSpeedCPUAdam

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

import t5_encoder

class HTClassifierModel(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
        self.save_hyperparameters()
        if isinstance(args, tuple) and len(args) > 0: 
            self.args = args[0]
            self.hparams.learning_rate = self.args.learning_rate
            self.hparams.eps = self.args.adam_epsilon
            self.hparams.weight_decay = self.args.weight_decay
            self.hparams.model_name_or_path = self.args.model_name_or_path
            self.hparams.num_classes = self.args.num_classes
            self.hparams.num_training_steps = self.args.num_training_steps
            self.hparams.warmup_steps = self.args.warmup_steps
        
        # freeze
        self._frozen = False

        # Handling the padding token in distilgpt2 by substituting it with eos_token_id
        if self.hparams.model_name_or_path == "distilgpt2":
            config = AutoConfig.from_pretrained(self.hparams.model_name_or_path, num_labels=self.hparams.num_classes, output_attentions=False, output_hidden_states=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.hparams.model_name_or_path, config=config)
            self.model.config.pad_token_id = self.model.config.eos_token_id
        else:
            config = AutoConfig.from_pretrained(self.hparams.model_name_or_path, num_labels=self.hparams.num_classes, output_attentions=False, output_hidden_states=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.hparams.model_name_or_path, config=config)

    def forward(self, batch):
        # The batch contains the input_ids, the input_put_mask and the labels (for training)
        input_ids = batch[0]
        input_mask = batch[1]
        labels = batch[2]

        outputs = self.model(input_ids, attention_mask=input_mask, labels=labels)
        loss = outputs[0]
        logits = outputs[1]
        
        return loss, logits

    def training_step(self, batch, batch_nb):
        # the training step is a (virtual) method,specified in the interface, that the pl.LightningModule
        # class stipulates you to overwrite. This we do here, by virtue of this definition
        outputs = self(batch)  # self refers to the model, which in turn acceses the forward method
        train_loss = outputs[0]
        self.log_dict({"train_loss": train_loss, "learning_rate":self.hparams.learning_rate}, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return train_loss
        # the training_step method expects a dictionary, which should at least contain the loss

    def validation_step(self, batch, batch_nb):
        # the training step is a (virtual) method,specified in the interface, that the pl.LightningModule
        # class  wants you to overwrite, in case you want to do validation. This we do here, by virtue of this definition.

        outputs = self(batch)
        # self refers to the model, which in turn accesses the forward method

        # Apart from the validation loss, we also want to track validation accuracy  to get an idea, what the
        # model training has achieved "in real terms".
        val_loss = outputs[0]
        logits = outputs[1]
        labels = batch[2]

        # Evaluating the performance
        predictions = torch.argmax(logits, dim=1)
        balanced_accuracy = balanced_accuracy_score(labels.detach().cpu().numpy(), predictions.detach().cpu().numpy(), adjusted=True)
        macro_accuracy = f1_score(labels.detach().cpu().numpy(), predictions.detach().cpu().numpy(), average='macro')
        micro_accuracy = f1_score(labels.detach().cpu().numpy(), predictions.detach().cpu().numpy(), average='micro')
        weighted_accuracy = f1_score(labels.detach().cpu().numpy(), predictions.detach().cpu().numpy(), average='weighted')
        
        self.log_dict({"val_loss": val_loss, 'accuracy': balanced_accuracy, 'macro-F1': macro_accuracy, 'micro-F1': micro_accuracy, 'weighted-F1':weighted_accuracy}, 
                      on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return val_loss
    
    def test_step(self, batch, batch_nb):
        # the training step is a (virtual) method,specified in the interface, that the pl.LightningModule
        # class  wants you to overwrite, in case you want to do test. This we do here, by virtue of this definition.

        outputs = self(batch)
        # self refers to the model, which in turn accesses the forward method

        # Apart from the validation loss, we also want to track validation accuracy  to get an idea, what the
        # model training has achieved "in real terms".
        test_loss = outputs[0]
        logits = outputs[1]
        labels = batch[2]

        # Evaluating the performance
        predictions = torch.argmax(logits, dim=1)
        balanced_accuracy = balanced_accuracy_score(labels.detach().cpu().numpy(), predictions.detach().cpu().numpy(), adjusted=True)
        macro_accuracy = f1_score(labels.detach().cpu().numpy(), predictions.detach().cpu().numpy(), average='macro')
        micro_accuracy = f1_score(labels.detach().cpu().numpy(), predictions.detach().cpu().numpy(), average='micro')
        weighted_accuracy = f1_score(labels.detach().cpu().numpy(), predictions.detach().cpu().numpy(), average='weighted')
        
        self.log_dict({"test_loss": test_loss, 'accuracy': balanced_accuracy, 'macro-F1': macro_accuracy, 'micro-F1': micro_accuracy, 'weighted-F1':weighted_accuracy}, 
                      on_step=True, on_epoch=True, prog_bar=True, logger=True)
    
    def predict_step(self, batch, batch_nb):
        # the training step is a (virtual) method,specified in the interface, that the pl.LightningModule
        # class  wants you to overwrite, in case you want to do validation. This we do here, by virtue of this definition.

        outputs = self(batch)
        # self refers to the model, which in turn accesses the forward method

        # Apart from the validation loss, we also want to track validation accuracy  to get an idea, what the
        # model training has achieved "in real terms".
        val_loss = outputs[0]
        logits = outputs[1]
        labels = batch[2]

        predictions = torch.argmax(logits, dim=1)
        return predictions.detach().cpu().numpy()

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

    def freeze(self) -> None:
        # freeze all layers, except the final classifier layers
        for name, param in self.model.named_parameters():
            if 'classifier' not in name:  # classifier layer
                param.requires_grad = False

        self._frozen = True

    def unfreeze(self) -> None:
        if self._frozen:
            for name, param in self.model.named_parameters():
                if 'classifier' not in name:  # classifier layer
                    param.requires_grad = True

        self._frozen = False

    def train_epoch_start(self):
        """pytorch lightning hook"""
        if self.current_epoch < self.hparams.nr_frozen_epochs:
            self.freeze()

        if self.current_epoch >= self.hparams.nr_frozen_epochs:
            self.unfreeze()