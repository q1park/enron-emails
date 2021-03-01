import argparse

import re
import numpy as np
import pandas as pd

from multiprocessing import Pool

convert_names = {
    'john t lau': 'jeff t skilling', 
    'john lau': 'jeff skilling',
    'betty s lau': 'kenneth s lay',
    'betty lau': 'kenneth lay',
    'wanda wong': 'kenneth s lay',
    'wanda': 'kenneth lay',
    'scott homes': 'jeff skilling',
    'cindy':'rosalee rleming',
    'peter lyons':'rosalee fleming'
}

convert_orgs = {
    'absolute protective systems':'mindspring',
    'adt security services':'mindspring',
    'cmg consortium management group, INC':'mindspring',
    'ark technology solutions':'mediaone',
    'benjamin thomas keenan':'mediaone',
    'ariba':'weforum',
    'bradley baron':'weforum',
}


def load_invoice(inv_path):
    inv = pd.read_excel(inv_path).fillna('').drop(columns=['filename', 'name'])
    inv = inv[(inv.date!='')&(inv.amount!='')]
    
    inv.recipient = inv.recipient.str.lower()
    inv.recipient = inv.recipient.replace(convert_names, regex=True)
    
    inv.vendor_name = inv.vendor_name.str.lower().replace(r'pllc|llc|inc|\n', r' ', regex=True)
    inv.vendor_name = inv.vendor_name.apply(lambda x: re.sub(r'\s+', ' ', x.strip(' .,')))
    inv.vendor_name = inv.vendor_name.replace(convert_orgs, regex=True)

    
    inv.date = pd.to_datetime(inv.date.apply(lambda x: x.replace(',2',', 2')))-pd.Timedelta(days=19*365+90) 
    inv.amount = inv.amount.str.replace(r'\$|\,', r'', regex=True).astype(np.float)

    
    inv = inv.sort_values(by='date')
    return inv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load_path",
        default='data/invoices/invoices.xlsx',
        type=str,
        help="raw data path",
    )

    parser.add_argument(
        "--save_path",
        default='data/invoices/invoices_clean.csv',
        type=str,
        help="cleaned data path",
    )
    
    args = parser.parse_args()
    
    print('loading data')
    
    data = load_invoice(args.load_path)
    
    print('saving')
    
    data.to_csv(args.save_path, index=False)
    
if __name__ == "__main__":
    main()