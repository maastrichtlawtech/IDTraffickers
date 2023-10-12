"""
Python version: 3.9
Description: Contains model architecture of transformer-based NNs for performing multi-class classification on the Backpage advertisements.
"""

# %% Importing libraries
from sklearn.metrics import f1_score, balanced_accuracy_score

import torch
import torch.nn as nn

import lightning.pytorch as pl

from deepspeed.ops.adam import DeepSpeedCPUAdam

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

# Custom Library
from LM import LMModel 

class HTTransferClassifierModel(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
        self.save_hyperparameters()
        if isinstance(args, tuple) and len(args) > 0: 
            self.args = args[0]
            self.hparams.learning_rate = self.args.learning_rate
            self.hparams.eps = self.args.adam_epsilon
            self.hparams.weight_decay = self.args.weight_decay
            self.hparams.pretrainedLM_path = self.args.pretrainedLM_path
            self.hparams.model_name_or_path = self.args.model_name_or_path
            self.hparams.num_classes = self.args.num_classes
            self.hparams.num_training_steps = self.args.num_training_steps
            self.hparams.warmup_steps = self.args.warmup_steps
            self.hparams.pooling_type = self.args.pooling_type
            self.hparams.dropout = self.args.dropout

        # freeze
        self._frozen = False

        # Loading the pretrained LM
        self.model = LMModel(model_name_or_path=self.hparams.model_name_or_path, learning_rate=self.hparams.learning_rate, 
                            warmup_steps=self.hparams.warmup_steps, nr_training_steps=self.hparams.num_training_steps, 
                            decay=self.hparams.weight_decay, adam_epsilon=self.hparams.eps).load_from_checkpoint(self.hparams.pretrainedLM_path, strict=False)

        self.drop = nn.Dropout(p=self.hparams.dropout)

        if self.hparams.pooling_type == "mean-max":
            # Since we will be using the mean-max pooling, the size of the hidden states will be 768 * 2
            self.out = nn.Linear(768 * 2, self.args.num_classes) 
        elif self.hparams.pooling_type in ["mean", "max"]:
            self.out = nn.Linear(768, self.args.num_classes) 
        else:
            raise NotImplementedError("Pooling type can only be amongst mean, max, or mean-max")

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, batch):
        # The batch contains the input_ids, the input_put_mask and the labels (for training)
        input_ids = batch[0]
        attention_mask = batch[1]
        labels = batch[2]

        hidden_states = self.model(input_ids, mode="fine-tuning")
        # Extracting the representation from the last layer of the hidden states
        hidden_states = torch.stack(hidden_states)[-1]

        # Generating the pooled output
        if self.hparams.pooling_type == "mean":
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            pooled_output = sum_embeddings / sum_mask

        elif self.hparams.pooling_type == "max":
            last_hidden_state = hidden_states
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            last_hidden_state[input_mask_expanded == 0] = float("-inf")  # Set padding tokens to large negative value
            pooled_output = torch.max(last_hidden_state, 1)[0]
        else:
            # Mean-max pooling
            last_hidden_state = hidden_states
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            mean_pooled_output = sum_embeddings / sum_mask

            last_hidden_state[input_mask_expanded == 0] = float("-inf")  # Set padding tokens to large negative value
            max_pooled_output = torch.max(last_hidden_state, 1)[0]

            pooled_output = torch.cat((mean_pooled_output, max_pooled_output), 1)

        with torch.cuda.amp.autocast(enabled=True):
            pooled_output = self.drop(pooled_output)

            outputs = self.out(pooled_output)
            # preds = torch.argmax(outputs, dim=1)
            loss = self.loss_fn(outputs, labels)

        return loss, outputs, hidden_states, attention_mask

    def training_step(self, batch, batch_nb):
        # the training step is a (virtual) method,specified in the interface, that the pl.LightningModule
        # class stipulates you to overwrite. This we do here, by virtue of this definition
        outputs = self(batch)  # self refers to the model, which in turn acceses the forward method
        train_loss = outputs[0]
        self.log_dict({"train_loss": train_loss}, on_step=False, on_epoch=True, prog_bar=True, logger=True)
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
        labels = batch[4]

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