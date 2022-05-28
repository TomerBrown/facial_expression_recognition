import pandas as pd
import os.path
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
from typing import List

CSV_PATH = 'datasets/KDEF.csv'
IMG_PATH = 'datasets/KDEF'
POSSIBLE_EMOTIONS = ['surprised', 'angry', 'neutral', 'sad', 'happy', 'disgusted', 'afraid']

class KDEF(Dataset):
    def __init__(
                    self,
                    df: pd.DataFrame,
                    transform=None,
                    img_path: str = IMG_PATH,
                 ):
        """
        :param df: A pandas df of desired subset of the kdef.csv file according to the experiment.
        :param transform: A transformation to apply to each image in the Dataset.
        :param img_path : A path to the kdef root folder to access each image.
        :returns: a PyTorch compatible kdef dataset to train and test on.
        """

        self.df = df.copy()
        # Make sure indexing works fine after taking the relevant rows.
        self.df.reset_index(drop=True, inplace=True)
        self.img_path: str = img_path
        self.transform = transform

    def __getitem__(self, index: int):

        file_name = self.df['file_name'][index]
        x = Image.open(os.path.join(self.img_path, file_name))
        y_label = torch.tensor(int(self.df["emotion_label"][index]))
        if self.transform:
            x = self.transform(x)
        return x, y_label

    def __len__(self) -> int:
        return len(self.df)

    def __repr__(self) ->str:
        title = "Subset of the KDEF Dataset:"
        ids = set(self.df["actor_id"])
        emotions = set(self.df["emotion"])
        angles = set(self.df["angle"])
        l = len(self)

        s = f"{title}\n\tids : {ids}\n\temotions: {emotions}\n\tangles : {angles}\n\tlen : {l}"
        return s
