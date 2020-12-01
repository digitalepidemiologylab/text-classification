import argparse
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument(
    '--data-train', type=str, default='./data/original/train.csv',
    help='train data path')
parser.add_argument(
    '--data-test', type=str, default='./data/original/test.csv',
    help='test data path')
parser.add_argument(
    '--seed', type=int, default=42,
    help='seed')

args = parser.parse_args()

df_train = pd.read_csv(args.data_train, header=None)
df_train = df_train[[0, 1]]
df_train = df_train.rename(columns={0: 'label', 1: 'text'})

df_test = pd.read_csv(args.data_test, header=None)
df_test = df_test.rename(columns={0: 'label', 1: 'text'})

df = pd.concat([df_train, df_test])
df = df.sample(frac=1, random_state=args.seed)

df.to_csv('./data/all.csv', index=False, header=True)
