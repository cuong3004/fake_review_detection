import pytorch_lightning as pl
import torch 
from transformers import BertForSequenceClassification

data_load = torch.load("epoch=0-step=253.ckpt", map_location=torch.device('cpu'))

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

model = Model()
model.load_state_dict(data_load["state_dict"])
torch.save(model.model.state_dict(), "bert_finetuned.bin")

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.load_state_dict(torch.load("bert_finetuned.bin"))