# %%
from transformers import BertModel, BertTokenizer, AdamW, AutoModel, AutoTokenizer, pipeline, BertForSequenceClassification, BertConfig
from transformers import get_linear_schedule_with_warmup
import pytorch_lightning as pl
import torchmetrics
from torchmetrics.functional import accuracy
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from fake_review.dataset_custom import CusttomData
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch 
from model import DenyBertForSequenceClassification, BertDelightModel

# from transformers import BertTokenizer, BertModel
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained("bert-base-uncased")
# text = "Replace me by any text you'd like."
# encoded_input = tokenizer(text, return_tensors='tf')
# output = model(encoded_input)
# %%

def soft_cross_entropy(predicts, targets):
            student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
            targets_prob = torch.nn.functional.softmax(targets, dim=-1)
            return (- targets_prob * student_likelihood).mean()

class LitClassification(pl.LightningModule):
    def __init__(self):
        super().__init__()

        df_train = pd.read_csv("../fake_review_train.csv")
        df_test = pd.read_csv("../fake_review_test.csv")
        # df_train, df_test = train_test_split(
        #     df, random_state=42, shuffle=True, test_size=0.2
        # )
        # df_train, df_valid = train_test_split(
        #     df_train, random_state=42, shuffle=True, test_size=0.1
        # )

        self.df_train = df_train
        self.df_valid = df_test
        self.df_test = df_test

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        config = BertConfig.from_json_file("config_delight.json")

        
        from delight_config import args 
        config.args = args

        self.student_model = DenyBertForSequenceClassification(config=config)
        self.teacher_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
        self.teacher_model.load_state_dict(torch.load("/content/drive/MyDrive/log_fake_review/bert_finetuned/lightning_logs/version_4/checkpoints/bert_finetuned.bin"))
        self.teacher_model.eval()

        # print(self.teacher_model)

        self.fit_dense = nn.Linear(config.hidden_size, 768)
        self.acc = torchmetrics.Accuracy()
        self.f1 = torchmetrics.F1Score(num_classes=1, multiclass=False)
        self.pre = torchmetrics.Precision(num_classes=1, multiclass=False)
        self.rec = torchmetrics.Recall(num_classes=1, multiclass=False)
        

    
    def train_dataloader(self):
        dataset = CusttomData(self.df_train, self.tokenizer)
        return DataLoader(dataset, batch_size=64, num_workers=2, shuffle=True)
    
    def val_dataloader(self):
        dataset = CusttomData(self.df_valid, self.tokenizer)
        return DataLoader(dataset, batch_size=32, num_workers=2)
    
    def test_dataloader(self):
        dataset = CusttomData(self.df_test, self.tokenizer)
        return DataLoader(dataset, batch_size=32, num_workers=2)


    def configure_optimizers(self):

        params = list(self.student_model.parameters()) + list(self.fit_dense.parameters())

        optimizer = AdamW(params,
                  lr = 1e-4, # args.learning_rate - default is 5e-5,
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )
        # optimizer_fit = AdamW(self.fit_dense.parameters(),
        #           lr = 1e-4, # args.learning_rate - default is 5e-5,
        #           eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
        #         )
        # total_steps = len(self.train_dataloader()) * Epoch

        # # Create the learning rate scheduler.
        # scheduler = get_linear_schedule_with_warmup(optimizer, 
        #                 num_warmup_steps = 0, # Default value in run_glue.py
        #                 num_training_steps = total_steps)
        # return [optimizer], [scheduler]
        # return [optimizer_student, optimizer_fit], []
        
        return optimizer

    def share_batch(self, batch, state):
        input_ids, attention_masks, labels = batch

        out_student = self.student_model(input_ids=input_ids, 
                        attention_mask=attention_masks, 
                        # labels=labels,
                        output_hidden_states=True,
                        output_attentions=True,
                        ) 
        with torch.no_grad():
            out_teacher = self.teacher_model(input_ids=input_ids, 
                            attention_mask=attention_masks, 
                            # labels=labels,
                            output_hidden_states=True,
                            output_attentions=True,
                            ) 

        att_loss = 0.
        rep_loss = 0.
        cls_loss = 0.

        student_logits, student_atts, student_reps = out_student.logits, out_student.attentions, out_student.hidden_states
        teacher_logits, teacher_atts, teacher_reps = out_teacher.logits, out_teacher.attentions, out_teacher.hidden_states

        # print("Teacher output", teacher_logits[0].shape, teacher_atts[0].shape, teacher_reps[0].shape)

        teacher_layer_num = len(teacher_atts)
        student_layer_num = len(student_atts)
        assert teacher_layer_num % student_layer_num == 0
        layers_per_block = int(teacher_layer_num / student_layer_num)
        new_teacher_atts = [teacher_atts[i * layers_per_block + layers_per_block - 1]
                            for i in range(student_layer_num)]
        
        for student_att, teacher_att in zip(student_atts, new_teacher_atts):
            # student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(self.device),
            #                             student_att)
            # teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(self.device),
            #                             teacher_att)

            tmp_loss = F.mse_loss(student_att, teacher_att)
            att_loss += tmp_loss

        new_teacher_reps = [teacher_reps[i * layers_per_block] for i in range(student_layer_num + 1)]
        new_student_reps = student_reps
        for student_rep, teacher_rep in zip(new_student_reps, new_teacher_reps):
            # print(student_rep.shape, teacher_rep.shape, )
            tmp_loss = F.mse_loss(self.fit_dense(student_rep), teacher_rep)
            rep_loss += tmp_loss

        cls_loss = soft_cross_entropy(student_logits / 1.,
                                            teacher_logits / 1.)
        


        loss = rep_loss + att_loss + cls_loss

        # self.log('train_loss', loss)
        self.log(f"{state}_loss", loss, on_step=False, on_epoch=True)

        acc = self.acc(student_logits, labels)
        pre = self.pre(student_logits, labels)
        rec = self.rec(student_logits, labels)
        f1 = self.f1(student_logits, labels)
        self.log(f'{state}_acc', acc, on_step=False, on_epoch=True)
        self.log(f'{state}_rec', rec, on_step=False, on_epoch=True)
        self.log(f'{state}_pre', pre, on_step=False, on_epoch=True)
        self.log(f'{state}_f1', f1, on_step=False, on_epoch=True)

        # self.log('train_acc', acc, on_step=True, on_epoch=False)
        return loss


    def training_step(self, train_batch, batch_idx):
        loss = self.share_batch(train_batch, "train")
        return loss

    def validation_step(self, val_batch, batch_idx):

        loss = self.share_batch(val_batch, "valid")

    def test_step(self, test_batch, batch_idx):

        loss = self.share_batch(test_batch, "test")
# 
# from fake_review.transformer import TinyBertForSequenceClassification, BertTokenizer
from pytorch_lightning.callbacks import ModelCheckpoint
filename = f"model"
checkpoint_callback = ModelCheckpoint(
    filename=filename,
    save_top_k=1,
    verbose=True,
    monitor='valid_loss',
    mode='min',
)
# tokenizer = BertTokenizer("../../bert-base-uncased/vocab.txt")
# %%
model_lit = LitClassification()
# %%
trainer = pl.Trainer(gpus=1, 
                    max_epochs=100,
                    limit_train_batches=0.2,
                    reload_dataloaders_every_n_epochs=1,
                    default_root_dir="/content/drive/MyDrive/log_fake_review/deny_bert_other",
                    callbacks=[checkpoint_callback]
                    )
trainer.fit(model_lit)
trainer.test(model_lit)