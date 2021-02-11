import argparse

import re
import numpy as np
import pandas as pd

from multiprocessing import Pool

def standard_format(df, Series, string, slicer):
    """Drops rows containing messages without some specified value in the expected locations. 
    Returns original dataframe without these values. Don't forget to reindex after doing this!!!"""
    rows = []
    for row, message in enumerate(Series):
        message_words = message.split('\n')
        if string not in message_words[slicer]:
            rows.append(row)
    df = df.drop(df.index[rows])
    return df

def clean_data(data):
    x = len(data.index)
    headers = ['Message-ID: ', 'Date: ', 'From: ', 'To: ', 'Subject: ']
    for i, v in enumerate(headers):
        data = standard_format(data, data.message, v, i)
    data = data.reset_index()
    print("Got rid of {} useless emails! That's {}% of the total number of messages in this dataset.".format(x - len(data.index), np.round(((x - len(data.index)) / x) * 100, decimals=2)))
    return data

def load_chunk(csv_path, start_idx, end_idx):
    pd.options.mode.chained_assignment = None

    chunk = pd.read_csv(csv_path, chunksize=end_idx)
    data = next(chunk)
    data = data.iloc[start_idx-end_idx:].reset_index(drop=True)
    data = clean_data(data)
#     data = format_data(data)
    return data


re_spaces = re.compile(r'\s+')
re_address = re.compile('[\w\.-]+@[\w\.-]+\.\w+')

def msg_text_date_sender_recipients_subject(message, n_recipients=3, text_start=15):
    message_words = message.split('\n')
    
    date = message_words[1].replace('Date: ', '')
    subject = message_words[4].replace('Subject: ', '')
    
    senders = re_address.findall(message_words[2])
    recipients = re_address.findall(message_words[3])
    
    text = [x.strip() for x in message_words[text_start:] if len(x.strip())>0]
    
    if len(recipients)<n_recipients:
        recipients+=['']*(n_recipients-len(recipients))
    elif len(recipients)>n_recipients:
        recipients = recipients[:n_recipients]
        
    if len(senders)<1:
        senders = ['']
        
    sender = senders[0]
    recipient1, recipient2, recipient3 = recipients
    text = re_spaces.sub(' ', ' '.join(text))
    
    return date, subject, sender, recipient1, recipient2, recipient3, text

def name(address):
    return address.split('@')[0]

def org(address):
    pieces = address.split('@')[-1].split('.')
    if len(pieces)>1:
        return pieces[-2]
    else:
        return pieces[-1]
    
def name_org(address):
    _name, _org = name(address), org(address)
    
    if len(_name)>0 or len(_org)>0:
        return '{}@{}'.format(_name, _org)
    else:
        return ''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load_path",
        default='data/enron/emails.csv',
        type=str,
        help="raw data path",
    )

    parser.add_argument(
        "--save_path",
        default='data/enron/emails_clean.csv',
        type=str,
        help="cleaned data path",
    )
    
    args = parser.parse_args()
    
    print('loading data')
    
    data = pd.read_csv(args.load_path)

    print('transforming data')
    
    data_clean = pd.DataFrame(
        Pool().map(msg_text_date_sender_recipients_subject, data.message),
        columns = ['date', 'subject', 'sender', 'recipient1', 'recipient2', 'recipient3', 'text']
    )
    
    data_clean['sender'] = list(Pool().map(name_org, data_clean.sender))
    data_clean['recipient1'] = list(Pool().map(name_org, data_clean.recipient1))
    data_clean['recipient2'] = list(Pool().map(name_org, data_clean.recipient2))
    data_clean['recipient3'] = list(Pool().map(name_org, data_clean.recipient3))
    
    print('to datetime')
    
    data_clean.date = data_clean.date.apply(lambda x: x.replace(' 000', ' 200')) ### Y2K BUG!!!!
    data_clean.date = pd.to_datetime(
        data_clean.date.apply(lambda x: re.search(r'[0-9].+\:[0-9]{2}(?=\s)', x).group()),
        format="%d %b %Y %H:%M:%S"
    )
    
    data_clean = data_clean.sort_values(by='date')
    
    print('saving')
    
    data_clean.to_csv(args.save_path, index=False)
    
if __name__ == "__main__":
    main()