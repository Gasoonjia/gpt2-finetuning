import pandas as pd
from sklearn.model_selection import train_test_split
import re

df_all = pd.read_csv('data/train.csv', sep=',')

def build_dataset(df, dest_path):
    f = open(dest_path, 'w')
    data = ''
    texts = df['text'].tolist()
    for text in texts:
        text = str(text).strip()
        text = re.sub(r"\s", " ", text)
        # bos_token = '<|endoftext|>'
        # eos_token = '<|endoftext|>'
        # data += bos_token + ' ' + text + ' ' + eos_token + '\n'
        data += text + '\n'
    f.write(data)

build_dataset(df_all, 'train_only_text.txt')