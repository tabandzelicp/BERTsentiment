from tqdm import tqdm
import torch.nn as nn
import torch

def loss_function(outputs, sentiments):
    return nn.BCEWithLogitsLoss()(outputs,sentiments.view(-1, 1))
    # This loss combines a Sigmoid layer and the BCELoss in one single class.




def training_loop(training_data_loader, model, optimizer, scheduler, device):
    # training state
    model.train()
    
    for batch_index, dataset in tqdm(enumerate(training_data_loader), total=len(training_data_loader)):
        # load from dataset
        input_ids = dataset['input_ids']
        attention_mask = dataset['attention_mask']
        token_type_ids = dataset['token_type_ids']
        sentiments = dataset['sentiments']
        # move to cuda device
        input_ids = input_ids.to(device, dtype=torch.long)
        attention_mask = attention_mask.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        sentiments = sentiments.to(device, dtype=torch.float)
        
        # set gradients to zero before every backprop becasue pytorch does not do that
        optimizer.zero_grad()
        outputs = model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
        )
        
        loss = loss_function(outputs, sentiments)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if batch_index % 500 == 0 and batch_index != 0:
            print('BATCH_INDEX: ',batch_index '==========', 'LOSS: ', loss.item())
            
            
def evaluation(validation_data_loader, model, device):
    
    # evaluation state
    model.eval()
    final_sentiments = []
    final_outputs = []
    with torch.no_grad():
        # deactivate autograd, helps with memory usage
        
        for batch_index, dataset in tqdm(enumerate(validation_data_loader), total=len(validation_data_loader)):
            # load from dataset
            input_ids = dataset['input_ids']
            attention_mask = dataset['attention_mask']
            token_type_ids = dataset['token_type_ids']
            sentiments = dataset['sentiments']
            # move to cuda device
            input_ids = input_ids.to(device, dtype=torch.long)
            attention_mask = attention_mask.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            sentiments = sentiments.to(device, dtype=torch.float)

            # set gradients to zero before every backprop becasue pytorch does not do that
            outputs = model(
                input_ids = input_ids,
                attention_mask = attention_mask,
                token_type_ids = token_type_ids,
            )
            
            final_sentiments.extend(sentiments.cpu().detach().numpy().tolist())
            # move to cpu
            # detach beacause no need for gradients
            # numpy array
            #list
            final_outputs.extend(torch.sigmoidmoid(outputs).cpu().detach().numpy().tolist())
            
    return final_outputs, final_sentiments    
        
        
        