import config
import torch



class BERTdataset:
    def __init__(self, review, sentiment):
        self.review = review # input
        self.sentiment = sentiment # target
        self.tokenizer = config.TOKENIZER
        self.max_length = config.MAX_LEN
        
        
    def __len__(self):
        return len(self.review)
    
    
    
    def __getitem__(self, item_index):
        review = str(self.review[item_index])
        review = ' '.join(review.split()) # first make a list out of sentences than make a sentnces with only one space between words
                                          #this just removes if there are some weired spaces between words
            
        # BERT can take as input either one or two sentences, and uses [SEP] token to separate them.
        # [CLS] token always appears at start of sentences
        # Both tokens are always required even if we only have one sentences becasue thats how BERT was pretrained and how expects input
        inputs = self.tokenizer.encode_plus(
            review,
            None,
            add_special_tokens = True,
            max_length = self.max_length,
            truncation = True
        )
        
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']
        
        # we could have done padding  as parametar in encode_plus but lets act fancy
        padding_length = self.max_length - len(input_ids)
        
        input_ids = input_ids + ([0] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'sentiments': torch.tensor(self.sentiment[item_index], dtype=torch.float)
        }
    