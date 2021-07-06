import pandas as pd
from sklearn.model_selection import train_test_split
import re

df_all = pd.read_csv('dataset/booksummaries.txt', sep='\t', names=['wiki_id', 'freebase_id', 'title', 'author', 'date', 'genres', 'summary'])


def build_dataset(df, dest_path):
    f = open(dest_path, 'w')
    data = ''
    summaries = df['summary'].tolist()
    for summary in summaries:
        summary = str(summary).strip()
        summary = re.sub(r"\s", " ", summary)
        # bos_token = '<|endoftext|>'
        # eos_token = '<|endoftext|>'
        # data += bos_token + ' ' + summary + ' ' + eos_token + '\n'
        data += summary + '\n'
    f.write(data)

build_dataset(df_all, 'train.txt')