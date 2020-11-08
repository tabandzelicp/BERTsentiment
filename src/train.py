import pandas as pd
import config
import dataset
import torch
import model
import engine
from sklearn import model_selection
from sklearn import metrics
import numpy as np
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
# main function for training and evaluation
def main():
    
    dataframe = pd.read_csv('../input/imdb.csv') # load dataframe
    dataframe.sentiment = dataframe.sentiment.apply(
        lambda x: 1 if x == 'positive' else 0
    )
    # sentiment is category target variable so we have to label encode it, we can do it like this by hand, or simply with sklearn.model_selection.LabelEncoder
    
    
    # now split data into validation and training
    
    df_train, df_valid = model_selection.train_test_split(
        dataframe,
        test_size = 0.1, # 10 percent of dataframe will be for validation
        random_state = 42, # if we are going to run multiple time this script, random state enables that everytime we get same split with same random state
        shuffle = True, # shuffle indices
        stratify = dataframe.sentiment.values # same distribution in train and valid 
    )
    
    df_train = df_train.reset_index(drop=True) # we reset indices from 0 to len(df_train)
    df_valid = df_valid.reset_index(drop=True) # we reset indices from 0 to len(df_valid)
    
    # make datasets with our class in order to make data loaders
    training_dataset = dataset.BERTdataset(
        review = df_train.review.values,
        sentiment = df_train.sentiment.values
    )
    # from dataset to dataloader
    training_data_loader = torch.utils.data.DataLoader(
        dataset = training_dataset,
        batch_size = config.TRAINING_BATCH_SIZE,
        shuffle = True,
        num_workers = 4
    )
    
    validation_dataset = dataset.BERTdataset(
        review = df_valid.review.values,
        sentiment = df_valid.sentiment.values,
    )
    # from dataset to dataloader
    validation_data_loader = torch.utils.data.DataLoader(
        dataset = validation_dataset,
        batch_size = config.VALIDATION_BATCH_SIZE,
        shuffle = False,
        num_workers = 4
    )
    
    device = torch.device('cuda')
    model = model.BERTsentiment()
    model.to(device) # move model to cuda device 
    # params to optimize 
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if  any(nd in n for nd in no_decay)], 'weight_decay': 0.00}
    ]
    
    number_of_training_steps = int(len(df_train) / config.TRAINING_BATCH_SIZE * config.EPOCHS) 
    #AdamW focuses on regularization and model does better on  generalization
    optimizer = AdamW(
        params = param_optimizer,
        lr = 3e-5
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer = optimizer,
        num_warmup_steps = 0,
        num_training_steps = number_of_training_steps,
        
    )
    
    
    
    
    best_accuracy = []
    
    for epoch in range(config.EPOCHS):
        print('EPOCH:', epoch + 1)
        engine.training_loop(
            training_data_loader,
            model,
            optimizer,
            scheduler,
            device)
        outputs, sentiments = engine.validation_loop(
            validation_data_loader, 
            model, 
            device)
        # distribution is 50 50 so we can use acc score
        outputs = np.array(outputs) >= 0.5 # positive class
        accuracy = metrics.accuracy_score(sentiments, outputs)
        print('ACCURACY SCORE',{accuracy})
        
        if accuracy > best_accuracy:
            torch.save(model.state_dict(), config.MODEL_PATH) # save model in working dir
            best_accuracy = accuracy

            
            

            
if __name__ == '__main__': # call function
    main()