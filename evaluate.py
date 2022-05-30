import pandas as pd
import os
import datetime
from utils.dataset_utils import get_test_dataloader
from tqdm import tqdm
import torch.nn as nn
import torch

def evaluate (vgg_model,train_df, val_df, test_df, name, Ncrop=True):

    # Create A folder to store all results for the experiment
    current_directory = os.getcwd()
    path = os.path.join(current_directory, "results")
    name_with_date = name+' '+datetime.datetime.now().strftime("%d-%m-%Y-%H-%M")
    path = os.path.join(path, name_with_date)
    os.makedirs(path)

    # Get Dataloader
    test_dataloader = get_test_dataloader(test_df)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = nn.CrossEntropyLoss()
    # Evaluate with changes
    vgg_model = vgg_model.eval()
    predictions = []
    correct = []
    loss_tr, correct_count, n_samples = 0.0, 0.0, 0.0
    for i , data in enumerate(tqdm(test_dataloader)):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        if Ncrop:
            # fuse crops and batchsize
            bs, ncrops, c, h, w = inputs.shape
            inputs = inputs.view(-1, c, h, w)
            # forward
            outputs = vgg_model(inputs)
            # combine results across the crops
            outputs = outputs.view(bs, ncrops, -1)
            outputs = torch.sum(outputs, dim=1) / ncrops
        else:
            outputs = vgg_model(inputs)

        y_hat = torch.argmax(outputs).item()
        predictions.append(y_hat)
        correct.append("Correct" if y_hat==labels.item() else "Wrong")

        loss = criterion(outputs, labels)

        # calculate performance metrics
        loss_tr += loss.item()

        _, preds = torch.max(outputs.data, 1)
        correct_count += (preds == labels).sum().item()
        n_samples += labels.size(0)

    acc = 100 * correct_count / n_samples
    loss = loss_tr / n_samples

    test_df["prediction"] = predictions
    test_df["correct"] = correct
    train_df.to_csv(os.path.join(path, "Train set.csv"))
    val_df.to_csv(os.path.join(path, "Validation set.csv"))
    test_df.to_csv(os.path.join(path, "Test set.csv"))

    # Calculate performance on test set
    s = "\t\t".join(['Test Accuracy: %2.4f %%' % acc,'Test Loss: %2.6f' % loss])

    with open(os.path.join(path,'results.txt'), 'w') as f:
        f.write(s)
