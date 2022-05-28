import torch
import sys
import warnings
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.dataset_utils import get_dataloaders_by_ids
from vgg_model import Vgg
from tqdm import tqdm
from utils import logger, hparams
from utils.loops import train, evaluate
from utils.checkpoint import save

MODEL_PATH = 'models/VGGNet'
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    # Hyper Parameters (What was used in the paper):
    hps = hparams.setup_hparams(["network=VGGNet", "name=vgg_fine_tuned_kdef"])
    print(hps)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = logger.Logger()

    # First Load the model
    vgg_model = Vgg().to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device(device))
    vgg_model.load_state_dict(checkpoint["params"])

    # Get the dataloaders
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders_by_ids(train_proportion=[0.6,0.2,0.2])

    # Training methods and criterion
    learning_rate = float(hps['lr'])
    optimizer = torch.optim.SGD(vgg_model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True,
                                weight_decay=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.75, patience=5, verbose=True)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    best_acc = 0.0
    print("Training", hps['name'], "on", device)
    for epoch in tqdm(range(hps['start_epoch'], hps['n_epochs'])):

        acc_tr, loss_tr = train(vgg_model, train_dataloader, criterion, optimizer, scaler)
        logger.loss_train.append(loss_tr)
        logger.acc_train.append(acc_tr)

        acc_v, loss_v = evaluate(vgg_model, val_dataloader, criterion)
        logger.loss_val.append(loss_v)
        logger.acc_val.append(acc_v)

        # Update learning rate
        scheduler.step(acc_v)

        if acc_v > best_acc:
            best_acc = acc_v

            save(vgg_model, logger, hps, epoch + 1)
            logger.save_plt(hps)

        if (epoch + 1) % hps['save_freq'] == 0:
            save(vgg_model, logger, hps, epoch + 1)
            logger.save_plt(hps)

        print('Epoch %2d' % (epoch + 1),
              'Train Accuracy: %2.4f %%' % acc_tr,
              'Val Accuracy: %2.4f %%' % acc_v,
              sep='\t\t')

    # Calculate performance on test set
    acc_test, loss_test = evaluate(vgg_model, test_dataloader, criterion)
    print('Test Accuracy: %2.4f %%' % acc_test,
          'Test Loss: %2.6f' % loss_test,
          sep='\t\t')

