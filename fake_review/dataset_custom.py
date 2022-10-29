class CusttomData:
    def __init__(self, df, tokenizer, len_seq=128, transforms=None):
        self.df = df
        self.tokenizer = tokenizer
        self.transforms = transforms
        self.len_seq = len_seq

        label_list = sorted(list(set(df.iloc[:,-2])))
        self.label_dir = {k:v for v, k in enumerate(label_list)}

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx,-1]
        label = self.df.iloc[idx,-2]

        encoded_dict = self.tokenizer.encode_plus(
                text,                      # Sentence to encode.
                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                max_length = self.len_seq,           # Pad & truncate all sentences.
                pad_to_max_length = True,
                return_attention_mask = True,   # Construct attn. masks.
                return_tensors = 'pt',     # Return pytorch tensors.
            )

        input_id = encoded_dict['input_ids'][0]
        attention_mask = encoded_dict['attention_mask'][0]
        label = self.label_dir[label]
        
        return input_id, attention_mask, label

