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
    ids = [i+1 for i in range(70)]
    random.shuffle(ids)
    train_por, val_por, test_por =  4/7, 1/7 , 2/7
    border1, border2 = int(35*train_por) , int(35*(train_por + val_por))

    train_ind = ids[:border1]
    val_ind = ids [border1:border2]
    test_ind = ids[border2:]

    train_df = df[df["actor_id"].isin(train_ind)]
    val_df = df[df["actor_id"].isin(val_ind)]
    test_df = df[df["actor_id"].isin(test_ind)]

    return train_df, val_df, test_df, name


def experiment2():
    """
        Creating test set , validation set and test set in the following way:
        60% of identities in the train set
        20% of identities in the validation set
        20% of identities in the test set
    """

    name = "random_split"
    df = pd.read_csv(CSV_PATH)

    # Actors Id's are 1-35
    indices = [i+1 for i in range(len(df))]
    random.shuffle(indices)

    train_por, val_por, test_por =  4/7, 1/7 , 2/7
    border1, border2 = int(len(df)*train_por) , int(len(df)*(train_por + val_por))

    train_ind = indices[:border1]
    val_ind = indices[border1:border2]
    test_ind = indices[border2:]

    train_df = df[df["Unnamed: 0"].isin(train_ind)]
    val_df = df[df["Unnamed: 0"].isin(val_ind)]
    test_df = df[df["Unnamed: 0"].isin(test_ind)]

    return train_df, val_df, test_df, name

def experiment3():
    """
        Creating test set , validation set and test set in the following way:
        60% of identities in the train set
        20% of identities in the validation set
        20% of identities in the test set
    """

    name = "leave_out_emotion"
    df = pd.read_csv(CSV_PATH)

    # Actors Id's are 1-35
    ids = [i+1 for i in range(70)]
    emotions = [i for i in range(7)]

    train_df = pd.DataFrame()
    val_df = pd.DataFrame()
    test_df = pd.DataFrame()
    for id in ids:
        random.shuffle(emotions)
        emotions_test = emotions[:2]
        emotions_val = emotions[2]
        emotions_train = emotions[3:]

        temp_id_df = df[df["actor_id"] == id]
        train_df = pd.concat([train_df, temp_id_df[temp_id_df["emotion_label"].isin(emotions_train)]])
        val_df = pd.concat([val_df, temp_id_df[temp_id_df["emotion_label"].isin([emotions_val])]])
        test_df = pd.concat([test_df, temp_id_df[temp_id_df["emotion_label"].isin(emotions_test)]])

    train_df.to_csv("try_train.csv")
    val_df.to_csv("try_val.csv")
    test_df.to_csv("try_test.csv")
    return train_df, val_df, test_df, name
