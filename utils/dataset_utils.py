from KDEF_Dataset import KDEF, POSSIBLE_EMOTIONS
from torch.utils.data import DataLoader , Dataset
import random
import pandas as pd
import torch
from torchvision import transforms


def get_train_val_test_idx(k: int, proportion: [float]):
    ids = [i + 1 for i in range(k)]
    random.shuffle(ids)
    split_idx1 = int(len(ids) * proportion[0])
    split_idx2 = int(len(ids) * (proportion[0]+proportion[1]))
    train_idx, val_idx, test_idx = ids[:split_idx1], ids[split_idx1:split_idx2], ids[split_idx2:]
    return train_idx, val_idx, test_idx


def get_transofrms(augment = True):
    mu, st = 0, 255

    test_transform = transforms.Compose([
        # transforms.Scale(52),
        transforms.Grayscale(),
        transforms.CenterCrop((562, 562)),
        transforms.Resize((48, 48)),
        transforms.TenCrop(40),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        # transforms.Lambda(lambda tensors: torch.stack([transforms.Normalize(mean=(mu,), std=(st,))(t) for t in tensors])),
    ])

    if augment:
        train_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.CenterCrop((562, 562)),
            transforms.Resize((48, 48)),
            transforms.RandomResizedCrop(48, scale=(0.8, 1.2)),
            transforms.RandomApply([transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),

            transforms.TenCrop(40),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            #transforms.Lambda(lambda tensors: torch.stack([transforms.Normalize(mean=(mu,), std=(st,))(t) for t in tensors])),
            transforms.Lambda(lambda tensors: torch.stack([transforms.RandomErasing(p=0.5)(t) for t in tensors])),
        ])
    else:
        train_transform = test_transform
    return train_transform, test_transform


def get_dataloaders_by_ids(img_path: str, csv_path: str, train_proportion: [int] = [0.6,0.2,0.2]) -> (DataLoader, DataLoader):

    # Load the dataset
    df = pd.read_csv(csv_path)

    # Split By Ids
    ids_train, ids_val, ids_test = get_train_val_test_idx(35, train_proportion)
    train_transform, test_transform = get_transofrms()

    train_df = df[df["actor_id"].isin(ids_train)]
    val_df = df[df["actor_id"].isin(ids_val)]
    test_df = df[df["actor_id"].isin(ids_test)]

    kdef_train = KDEF(train_df, transform=train_transform)
    kdef_val = KDEF(val_df, transform=test_transform)
    kdef_test = KDEF(test_df, transform=test_transform)

    # To deal with some issue that occurred on my cpu
    n_workers = 4 if torch.cuda.is_available() else 0

    # Create Dataloaders
    train_dataloader = DataLoader(kdef_train, batch_size=64, shuffle=True, num_workers=n_workers)
    val_dataloader = DataLoader(kdef_val, batch_size=64, shuffle=False, num_workers=n_workers)
    test_dataloader = DataLoader(kdef_test, batch_size=64, shuffle=False, num_workers=n_workers)

    return train_dataloader, val_dataloader, test_dataloader


def get_train_val_dataloaders(train_df : pd.DataFrame, val_df : pd.DataFrame):

    train_transform, test_transform = get_transofrms()

    kdef_train = KDEF(train_df, transform=train_transform)
    kdef_val = KDEF(val_df, transform=test_transform)

    # To deal with some issue that occurred on my cpu
    n_workers = 4 if torch.cuda.is_available() else 0

    # Create Dataloaders
    train_dataloader = DataLoader(kdef_train, batch_size=64, shuffle=True, num_workers=n_workers)
    val_dataloader = DataLoader(kdef_val, batch_size=64, shuffle=False, num_workers=n_workers)

    return train_dataloader, val_dataloader

def get_test_dataloader(test_df : pd.DataFrame):

    _, test_transform = get_transofrms()
    kdef_test = KDEF(test_df, transform=test_transform)

    # To deal with some issue that occurred on my cpu
    n_workers = 4 if torch.cuda.is_available() else 0

    test_dataloader = DataLoader(kdef_test, batch_size=1, shuffle=False, num_workers=n_workers)

    return test_dataloader
