from sklearn.model_selection import train_test_split
import pandas as pd


def create_train_val_split(csv_name):  # this is the validation set (not test)
    """
    Creates a train and test csv from input csv
    """
    df = pd.read_csv(csv_name)

    train, val = train_test_split(df, train_size=0.8, random_state=0)

    train.to_csv(csv_name.replace('.csv', '') + '-train.csv', index=False)
    val.to_csv(csv_name.replace('.csv', '') + '-val.csv', index=False)


create_train_val_split('../subsets/30k-2.csv')
