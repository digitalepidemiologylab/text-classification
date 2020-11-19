import argparse
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument(
    '--data', type=str, default='./original/labeled_data.csv',
    help='data path')

args = parser.parse_args()

df = pd.read_csv(args.data)
df = df[['tweet', 'class']]
df = df.rename(columns={'tweet': 'text', 'class': 'label'})
df.to_csv('./all.csv', index=False, header=True)
