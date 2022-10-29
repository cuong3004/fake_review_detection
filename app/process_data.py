from transformers import AutoTokenizer
import torch 
import pandas as pd
from tqdm import tqdm
from collections import Counter

class ProcessData:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=False)
        self.model = torch.jit.load("traced_review.pt")
        self.list_predict = []
        self.batch = 10

    def process_data(self):
        df = pd.read_csv("output.csv")
        df.dropna(axis=0)

        sentent_batch = []
        predict_batch = []
        # print(len(df.iloc[:,-1]))
        for i, sentent in tqdm(enumerate(df.iloc[:,-1])):
            sentent = str(sentent)
            # print(len(self.tokenizer(sentent)))
            # print(self.tokenizer(sentent))
            if len(self.tokenizer(sentent).input_ids) <= 10:
                continue
            # if i > 100:
            #     break
            sentent_batch.append(sentent)

            if len(sentent_batch) == self.batch:
                input_ids = []
                attention_masks = []
                for text in sentent_batch:
                    # print(text)
                    encoded_dict = self.tokenizer.encode_plus(
                        text,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 50,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                    )

                    input_ids.append(encoded_dict['input_ids'])
                    attention_masks.append(encoded_dict['attention_mask'])

                input_ids = torch.cat(input_ids, dim=0)
                attention_masks = torch.cat(attention_masks, dim=0)


                with torch.no_grad():
                    out = self.model(input_ids, attention_masks)
                    # print(out)
                    out_confident = torch.softmax(out, -1)
                    out_argmax = torch.argmax(out_confident, -1)
                    # print(out_confident.shape)
                    # print(out_argmax.shape)
                    # out_confident = torch.index_select(out_confident, 1, out_argmax)
                    # print(out_confident)
                    # print("out_argmax", out_argmax)
                    # print("out_confident", out_confident[range(10), out_argmax])
                    # # if out_confident > 0.8:
                    # print(out_confident)
                    # print([out_confident > 0.8])
                    # print(out_confident[out_confident > 0.8])
                    out_confident = out_confident[range(10), out_argmax]
                    # print(out_confident)
                    # print([out_confident > 0.8])
                    # print(out_argmax[out_confident > 0.8].tolist())
                    # print("OKOKOK")
                    # out_confident = out_confident[out_confident > 0.8]
                    # print(out_confident.shape)

                    predict_batch = predict_batch + out_argmax[out_confident > 0.8].tolist()
                    # print(predict_batch)

                sentent_batch = []

        # print(Counter(predict_batch))
        return dict(Counter(predict_batch)), len(df.iloc[:,-1])






                # if len(predict_batch)  100:
                #     print(predict_batch)
                #     break


                    # print(predict_batch)
            # print("Ok")



# ProcessData().process_data()