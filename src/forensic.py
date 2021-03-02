import re
import numpy as np
import pandas as pd
from datetime import datetime

from src.structures import DictInv, PGraph
from src.spellcheck import find_close


def filterby_date(df, start_date=datetime(1979, 1, 1), end_date=datetime(2005, 1, 1)):
    return df[(df.datetime>=start_date)&(df.datetime<=end_date)]

node_attrs = ['name1', 'name2', 'email1', 'email2', 'org1', 'org2']
edge_bounds = ['sender', 'receiver1', 'receiver2', 'receiver3']

def infer_name(row):
    name = row['email1'].split('@')[0]
    pieces = [x for x in re.split(r'[\.\-\_]', name) if len(x)>0]
    if len(pieces)>=2 and len(pieces[0])>=2:
        return ' '.join(pieces)
    else:
        return ''
    
def infer_org(row):
    suffix = row['email1'].split('@')[-1]
    pieces = [x for x in re.split(r'\.', suffix) if len(x)>0]
    if len(pieces)==2 and pieces[0] not in ['aol', 'earthlink']:
        return pieces[0]
    elif len(pieces)>2 and pieces[-1]=='edu':
        return pieces[-2]
    else:
        return ''
    
def emails2nodes(df):
    emails = np.concatenate([df[x].unique() for x in ['sender', 'recipient1', 'recipient2', 'recipient3']])
    emails = [x for x in np.sort(np.unique(emails)) if len(x)>0]
    
    nodes = pd.DataFrame({k:['']*len(emails) if k!='email1' else emails for k in node_attrs})
    nodes['name1'] = nodes.apply(infer_name, axis=1)
    nodes['org1'] = nodes.apply(infer_org, axis=1)
    nodes = nodes.sort_values(by='org1').reset_index(drop=True)
    return nodes

def emails2edges(df, nodes):
    emails2idx = {v:k for k,v in nodes.to_dict()['email1'].items()}
    
    edges = pd.DataFrame({
        'sender':list(map(emails2idx.get, df['sender'])),
        'receiver1':list(map(emails2idx.get, df['recipient1'])),
        'receiver2':list(map(emails2idx.get, df['recipient2'])),
        'receiver3':list(map(emails2idx.get, df['recipient3'])),
        'type':['email']*len(df),
        'datetime':df['date'].tolist(),
        'desc':df['subject'].tolist(),
        'data':df['text'].tolist()
    }).fillna('')
    
    return edges

def invoices2nodes(df):
    nodes_dict = {k:[] for k in node_attrs}
    
    for i,row in df.iterrows():
        sender = row['recipient'].lower()
        recipient = row['vendor_name']
        datetime = row['date']
        desc = row['description']
        data = row['amount']

        nodes_dict['name1'].append(sender)
        nodes_dict['name2'].append('')
        nodes_dict['email1'].append('')
        nodes_dict['email2'].append('')
        nodes_dict['org1'].append('enron')
        nodes_dict['org2'].append('')
        
        nodes_dict['name1'].append('')
        nodes_dict['name2'].append(recipient)
        nodes_dict['email1'].append('')
        nodes_dict['email2'].append('')
        nodes_dict['org1'].append(recipient)
        nodes_dict['org2'].append('')
        
    return pd.DataFrame(nodes_dict).drop_duplicates().reset_index(drop=True)

def invoices2edges(df, nodes):
    receiver2idx = {v:k for k,v in nodes.to_dict()['org1'].items() if v!=''}
    sender2idx = {v:k for k,v in nodes.to_dict()['name1'].items() if v!=''}
    
    df = df[df['recipient']!='']
    
    edges = pd.DataFrame({
        'sender':list(map(sender2idx.get, df['recipient'])),
        'receiver1':list(map(receiver2idx.get, df['vendor_name'])),
        'receiver2':['']*len(df),
        'receiver3':['']*len(df),
        'type':['invoice']*len(df),
        'datetime':df['date'].tolist(),
        'desc':df['description'].tolist(),
        'data':df['amount'].tolist()
    }).fillna('')
    
    return edges

class ForensicGraph(PGraph):
    @classmethod
    def from_emails(cls, emails_path):
        emails = pd.read_csv(emails_path).fillna('')
        emails['date'] = pd.to_datetime(emails['date'])
        nodes = emails2nodes(emails)
        edges = emails2edges(emails, nodes)
        return cls(nodes=nodes, edges=edges)
    
    def __init__(self, nodes, edges):
        super().__init__(nodes, edges)
        self.add_invoices()
        
    def add_invoices(self, inv_path='data/invoices/invoices_clean.csv'):
        nodes_start, edges_start = self.nodes.index[-1]+1, self.edges.index[-1]+1
        
        invoices = pd.read_csv(inv_path).fillna('')
        invoices.date = pd.to_datetime(invoices.date)
        
        nodes = invoices2nodes(invoices)
        edges = invoices2edges(invoices, nodes)

        nodes.index+=nodes_start
        edges.index+=edges_start
        
        edges['sender']+=nodes_start
        edges['receiver1']+=nodes_start
        
        self.nodes = pd.concat([self.nodes, nodes])
        self.edges = pd.concat([self.edges, edges])
        
    def edges_from_nodes(self, *nodes, edges=None):
        if edges is None:
            return self.edges[self.edges[edge_bounds].isin(nodes).any(axis=1)]
        else:
            return edges[edges[edge_bounds].isin(nodes).any(axis=1)]
    
    def nodes_from_edges(self, *edges, nodes=None):
        if nodes is None:
            return self.nodes[self.nodes.index.isin(self.uniq_edge_values(edges, edge_bounds))]
        else:
            return nodes[nodes.index.isin(self.uniq_edge_values(edges, edge_bounds))]
        
    def make_subgraph(self, f1, f2=None, start_date=datetime(2001, 1, 1), end_date=datetime(2002, 1, 1)):
        edges = filterby_date(self.edges, start_date=start_date, end_date=end_date)
        edges = self.edges_from_nodes(*f1, edges=edges)
        
        if f2 is not None:
            edges = self.edges_from_nodes(*f2, edges=edges)
            
        nodes = self.nodes_from_edges(*edges.index)
        return PGraph(nodes=nodes, edges=edges)
        
    def _merged_row(self, *idxs):
        idxs = list(idxs)
        def pair(values):
            return values+['']*(2-len(values)) if len(values)<2 else values[:2]
            
        row = {}
        row['name1'], row['name2'] = pair(self.uniq_node_values(idxs, ['name1', 'name2']))
        row['email1'], row['email2'] = pair(self.uniq_node_values(idxs, ['email1', 'email2']))
        row['org1'], row['org2'] = pair(self.uniq_node_values(idxs, ['org1', 'org2']))
        return pd.Series(row)

    def merge(self, *idxs):
        if len(idxs)>1:
            idxs = list(sorted(idxs))
            prime, aliases = idxs[0], idxs[1:]
            merged_row = self._merged_row(*idxs)
            self.nodes.loc[prime].update(self._merged_row(*idxs))
            self.nodes = self.nodes.drop(index=aliases)

            edge_repl = dict(zip(aliases, [prime]*len(aliases))) 
            self.edges = self.edges.replace({k:edge_repl for k in edge_bounds})
            
    def search_name(self, name):
        idxs = set()
        idxs.update(find_close(name, self.nodes))
        
        pieces = [x.strip() for x in name.split() if len(x.strip())>0]
        if len(pieces)>=2:
            alt_1 = '.'.join([pieces[0][0], pieces[-1]])
            idxs.update(find_close(alt_1, self.nodes))
            
            if len(pieces)==2:
                alt_2 = '  '.join([pieces[0], pieces[1]])
                idxs.update(find_close(alt_2, self.nodes))
        return list(sorted(idxs))
    
    def search_names(self, names):
        names = [x.strip() for x in names.split(',')]
        return [self.search_name(x) for x in names]
    
    def node_summaries(self, *idxs):
        def summarize(row):
            name=row['name1']
            tag=row['email1'].split('@')[0]
            return name, row['org1'], tag
        return {idx:summarize(self.nodes.loc[idx]) for idx in idxs}

    def node_assoc(self, *idxs):
        idxs = list(sorted(idxs))
        edges = self.edges_from_nodes(*idxs).index
        nodes = self.nodes_from_edges(*edges)
        return nodes.drop(index=idxs).index.tolist()
    
    def assoc_summaries(self, *idxs):
        assoc = self.node_assoc(*idxs)
        return self.node_summaries(*assoc)