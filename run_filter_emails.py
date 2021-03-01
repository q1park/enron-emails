import argparse

import re
import numpy as np
import pandas as pd

from multiprocessing import Pool

def reget(regex, string):
    re_string = regex.search(string)
    return re_string.group() if re_string is not None else ''

re_trim_text1a = re.compile(r'[^a-zA-Z0-9]{5,6}')
re_trim_text1b = re.compile(r'.*?(?=[^a-zA-Z0-9]{5,6})')
re_trim_text2a = re.compile(r'\<|\[')
re_trim_text2b = re.compile(r'.*?(?=(\<|\[))')
re_trim_text3a = re.compile(r'From\:|To\:')
re_trim_text3b = re.compile(r'.*?(?=From\:)|.*?(?=To\:)')
re_address1a = re.compile('[\w\.-]+@[\w\.-]+\.\w+')
re_address1b = re.compile('.*?[\w\.-]+@[\w\.-]+\.\w+')

def trim_text(text):
    if re_trim_text1a.search(text) is not None:
        text = reget(re_trim_text1b, text)
    if re_trim_text2a.search(text) is not None:
        text = reget(re_trim_text2b, text)
    if re_trim_text3a.search(text) is not None:
        text = reget(re_trim_text3b, text)
    if re_address1a.search(text) is not None:
        text = reget(re_address1b, text)
        text = re_address1a.sub('', text).strip()
    return text

def count_words(text):
    return len(text.split())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load_path",
        default='data/enron/emails_clean.csv',
        type=str,
        help="raw data path",
    )

    parser.add_argument(
        "--save_path",
        default='data/enron/emails_filtered.csv',
        type=str,
        help="cleaned data path",
    )
    
    args = parser.parse_args()
    
    print('loading data')
    
    data = pd.read_csv(args.load_path).fillna('')

    print('transforming data')
    
    data_clean = data[
        (~data.subject.str.contains(r'ANSI|Mime|MIME'))&
        (~data.sender.str.contains(r'system\.administrator|enron\.announcements'))&
        (~data.text.str.contains(r'^\<|^X\-|[^a-zA-Z0-9]{5,6}'))
    ]
    
    print('trimming text')
    
    data_clean['text'] = list(Pool().map(trim_text, data_clean.text))
    
    print('filter n_words in (4, 400)')
    
    data_clean = data_clean[
        (data_clean.text.apply(count_words)>3)&
        (data_clean.text.apply(count_words)<=400)
    ].drop_duplicates().reset_index(drop=True)
    
    
    print('saving')
    
    data_clean.to_csv(args.save_path, index=False)
    
    data_small = data_clean[data_clean.date>'2001-01-01'].reset_index(drop=True)
    data_small.to_csv('data/enron/emails_small.csv', index=False)
    
if __name__ == "__main__":
    main()