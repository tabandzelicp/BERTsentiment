import torch.nn as nn
import transformers
import config




class BERTsentiment(nn.Module):
    def __init__(self):
        super(BERTsentiment, self).__init__()
        
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)
        self.drop = nn.Dropout(0.5) # for regularization
        self.out_layer = nn.Linear(768, 1) # BERT model uses 768 in last, 1 output because its binary 
        
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        # out1 = Sequence of hidden states at the output of the last layer of the model
        # out2 =  Last layer hidden-state of the first token of the sequence (classification token) further processed by a Linear layer and a Tanh activation function.
        # The Linear layer weights are trained from the next sentence prediction (classification) objective during pretraining.
        out1, out2 = self.bert(
            input_ids = input_ids, # token indices 
            attention_mask = attention_mask, # indices for padding 0 and 1
            token_type_ids = token_type_ids  # indices for sentences, we dont really need this becasue our input is only one sentence so its always gona be 0
        )
        
        bert_output = self.drop(out2) # apply Dropout
        output = self.out_layer(bert_output) # pass to Linear layer
        
        return output # Linear output 
    