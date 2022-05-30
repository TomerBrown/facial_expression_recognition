import pandas as pd
from KDEF_Dataset import CSV_PATH
import random

def experiment1():
    """
        Creating test set , validation set and test set in the following way:
        60% of identities in the train set
        20% of identities in the validation set
        20% of identities in the test set
    """

    name = "split_by_identity"
    df = pd.read_csv(CSV_PATH)

    # Actors Id's are 1-35
    ids = [i+1 for i in range(35)]
    random.shuffle(ids)

    train_por, val_por, test_por = 0.6, 0.2, 0.2
    border1, border2 = int(35*train_por) , int(35*(train_por + val_por))

    train_ind = ids[:border1]
    val_ind = ids [border1:border2]
    test_ind = ids[border2:]

    train_df = df[df["actor_id"].isin(train_ind)]
    val_df = df[df["actor_id"].isin(val_ind)]
    test_df = df[df["actor_id"].isin(test_ind)]

    return train_df, val_df, test_df, name





