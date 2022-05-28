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
                    transform=None,
                    csv_path: str = CSV_PATH,
                    img_path: str = IMG_PATH,
                    only_straight: bool = False,
                    ids_to_take: List[int] = None,
                    emotions_to_take: List[str] = None
                 ):

        df = pd.read_csv(csv_path)

        # If needed take only straight posed images
        if only_straight:
            df = df[df['angle'] == 'S']


        # Take the given ids to dataset
        if ids_to_take is not None:
            df = df[df['actor_id'].isin(ids_to_take)]

        # Take the given emotions to dataset
        if emotions_to_take is not None:
            # First, check for error in values and raise exception if found
            for emotion in emotions_to_take:
                if emotion not in POSSIBLE_EMOTIONS:
                    raise ValueError(f"every element in emotions_to_take should be in: {POSSIBLE_EMOTIONS}\n"
                                     f"but got {emotion}")
            df = df[df['emotion'].isin(emotions_to_take)]
        df.reset_index(drop=True, inplace=True)
        self.df = df
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
