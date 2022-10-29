from main import LitClassification 
import pytorch_lightning as pl

model_lit = LitClassification.load_from_checkpoint()

trainer = pl.Trainer(gpus=1, 
                    )
# trainer.fit(model_lit)
trainer.test(model_lit)