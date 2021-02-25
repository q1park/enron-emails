import streamlit as st
st. set_page_config(layout="wide") 
import re
import glob

import SessionState
    
import os
import pickle
    
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from datetime import datetime
def filterby_date_old(df, start_date='1979-01-01', end_date='2005-01-01'):
    return df[(df.date>=start_date)&(df.date<=end_date)].reset_index(drop=True)

def filter_poi(df, poi):
    return df[
        (df.sender.isin(poi))|
        (df.recipient1.isin(poi))|
        (df.recipient2.isin(poi))|
        (df.recipient3.isin(poi))
    ]

def format_text_view_keys(date, subject, recipient):
    return "{}, {}, {}".format(date, recipient, subject[:15])

def get_text_view(df, sender):
    text_view = {}
    
    for i,row in df[df.sender==sender].iterrows():
        key = format_text_view_keys(row.date, row.subject, row.recipient1)
        text_view[key] = row.text
    return text_view

def search_emails(df, query, by_name=True, by_org=False, by_text=False):
    results = pd.DataFrame()
    
    if by_name:
        results = pd.concat([
            results,
            df[df.sender.apply(lambda x: x.split('@')[0]).str.contains(query)]
        ])

    if by_org:
        results = pd.concat([
            results,
            df[df.sender.apply(lambda x: x.split('@')[-1]).str.contains(query)]
        ])

    if by_text:
        results = pd.concat([
            results,
            df[df.text.str.contains(query)]
        ])
    return results.sort_values(by='date').drop_duplicates()

from collections import Counter

class PGraph:
    def __init__(self, nodes=None, edges=None):
        self.nodes = nodes
        self.edges = edges
        
    def node_values(self, idxs, cols=None):
        idxs = list(idxs)
        if cols is None:
            cols = self.nodes.columns.tolist()
            
        return [x for x in self.nodes.loc[idxs][cols].values.reshape(-1) if x!='']
    
    def edge_values(self, idxs, cols=None):
        idxs = list(idxs)
        if cols is None:
            cols = self.nodes.columns.tolist()
            
        if 'sender' in cols:
            return [int(x) for x in self.edges.loc[idxs][cols].values.reshape(-1) if x!='']
        else:
            return [x for x in self.edges.loc[idxs][cols].values.reshape(-1) if x!='']
        
    def uniq_node_values(self, idxs, cols):
        return list(sorted(set(self.node_values(idxs, cols))))
    
    def uniq_edge_values(self, idxs, cols):
        return list(sorted(set(self.edge_values(idxs, cols))))
    
    def node_counts(self, idxs, cols):
        return Counter(self.node_values(idxs, cols))
    
    def edge_counts(self, idxs, cols):
        return Counter(self.edge_values(idxs, cols))

from datetime import datetime
from src.spellcheck import find_close

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

def filterby_date(df, start_date=datetime(1979, 1, 1), end_date=datetime(2005, 1, 1)):
    return df[(df.datetime>=start_date)&(df.datetime<=end_date)].reset_index(drop=True)

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
        
    def make_subgraph(self, f1, f2=None, start_date=datetime(2001, 1, 1), end_date=datetime(2002, 1, 1)):
        edges = filterby_date(self.edges, start_date=start_date, end_date=end_date)
        edges = self.edges_from_nodes(*f1, edges=edges)
        
        if f2 is not None:
            edges = self.edges_from_nodes(*f2, edges=edges)
            
        nodes = self.nodes_from_edges(*edges.index)
        return PGraph(nodes=nodes, edges=edges)
        
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
#             alt_2 = '.'.join([pieces[0], pieces[-1][0]])
#             idxs.update(find_close(alt_2, self.nodes))
        return list(sorted(idxs))
    
    def search_names(self, names):
        names = [x.strip() for x in names.split(',')]
        return [self.search_name(x) for x in names]
    
def node_summaries(idxs, nodes):
    def summarize(row):
#         if len(row['name1'])>0:
#             name=row['name1']
#         elif len(row['email1'])>0:
#             name=row['email1'].split('@')[0]
        name=row['name1']
        tag=row['email1'].split('@')[0]
        return name, row['org1'], tag
    return {summarize(nodes.loc[idx]):idx for idx in idxs}
    
def edge_summaries(idxs, nodes, edges):
    def summarize(row):
        sender_row = nodes.loc[int(row['sender'])]
        receiver_row = nodes.loc[int(row['receiver1'])]
        sender = sender_row['name1'] if sender_row['name1'] is not '' else sender_row['email1'].split('@')[0]
        receiver = receiver_row['name1'] if receiver_row['name1'] is not '' else receiver_row['email1'].split('@')[0]
        return sender, receiver, row['datetime'].date(), row['desc'][:10]
    return {summarize(edges.loc[idx]):idx for idx in idxs}

def get_orgs(idxs, nodes):
    return [x[0] for x in Counter(nodes.loc[idxs]['org1']).most_common() if len(x[0])>0]

def filter_nodes_by_org(idxs, nodes, org):
    filtered_nodes = nodes.loc[idxs]
    return filtered_nodes[filtered_nodes['org1'].isin(org)].index

def filter_nodes_by_date(idxs, edges, start_date, end_date):
    edges = filterby_date(edges, start_date=start_date, end_date=end_date)
    all_nodes = [int(x) for x in edges[edge_bounds].values.reshape(-1) if x!='']
    return [x for x in idxs if x in all_nodes]

class DictInv(dict):
    def __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)
        self.inv = dict()

    def __getitem__(self, key):
        val = dict.__getitem__(self, key)
        return val

    def __setitem__(self, key, val):
        dict.__setitem__(self, key, val)
        self.inv.__setitem__(val, key)
        
    def __delitem__(self, key):
        if key in self:
            val = self[key]
            self.inv.__delitem__(val)
            return dict.__delitem__(self, key)
        elif key in self.inv:
            val = self.inv[key]
            dict.__delitem__(self, val)
            return self.inv.__delitem__(key)

    def __repr__(self):
        dictrepr = dict.__repr__(self)
        return '%s(%s)' % (type(self).__name__, dictrepr)
        
    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v
            
    def pop(self, key):
        if key in self:
            val = self[key]
            self.__delitem__(key)
        elif key in self.inv:
            val = self.inv[key]
            self.__delitem__(val)
        else:
            raise KeyError(key)
        return val
    
def run():
    data = filterby_date_old(pd.read_csv('data/enron/emails_filtered.csv').fillna(''))
    data['date'] = pd.to_datetime(data['date'])
    
    state = SessionState.get(
        graph=ForensicGraph.from_emails('data/enron/emails_filtered.csv'),
        subgraph=PGraph(), 
        node_dict=DictInv(), edge_dict=DictInv(), merge_dict=DictInv(),
        poi=[], assoc=[], org=[], merge_cands=[]
    )
    
    ###############################################
    ### Sidebar
    ###############################################
    
    st.sidebar.header('Persons of Interest')
    
    poi_find = st.sidebar.text_area(
        "Add person names separated by commas", 
        "kenneth lay, jeff skilling"
    )
    
    
    if st.sidebar.button("Find Persons of Interest"):
        merge_cands = [
            dict(sorted(node_summaries(x, state.graph.nodes).items(), key=lambda x: x[1])) 
            for x in state.graph.search_names(poi_find)
        ]
        
        state.merge_cands = [list(x.values()) for x in merge_cands]
        state.merge_dict.update({k:v for cand in merge_cands for k,v in cand.items()})
        
    ###############################################
        
    for i, x in enumerate(state.merge_cands):
        if len(x)>=2:
            merge_keys = st.sidebar.multiselect(
                'Merge suggestion {}'.format(i),
                list(map(state.merge_dict.inv.get, x)), 
                default=list(map(state.merge_dict.inv.get, x))
            )

            if st.sidebar.button("Merge {}".format(i+1)):
                idxs = sorted(list(map(state.merge_dict.get, merge_keys)))
                state.graph.merge(*idxs)
                state.merge_cands[i] = idxs[0:1]
                state.merge_dict.update(node_summaries(idxs[0:1], state.graph.nodes))
                SessionState.rerun()
    
    _poi_add = list(map(state.merge_dict.inv.get, [x for cand in state.merge_cands for x in cand]))
    poi_add = st.sidebar.multiselect('Persons List', _poi_add, default=_poi_add)
    
    if st.sidebar.button("Add Persons of Interest"):
        poi_dict = {x:state.merge_dict[x] for x in poi_add}
        
        assoc_dict = node_summaries(
            state.graph.nodes_from_edges(
                *state.graph.edges_from_nodes(
                    *list(poi_dict.values())).index
            ).drop(index=list(poi_dict.values())).index.tolist(), 
            state.graph.nodes
        )

        state.node_dict.update(poi_dict)
        state.node_dict.update(assoc_dict)
#         state.edge_dict.update({add_poi[x]:x for x in poi_add})

        state.poi.extend(list(poi_dict.values()))
        state.assoc.extend(list(assoc_dict.values()))
        state.merge_dict = DictInv()
        state.merge_cands = []
        SessionState.rerun()

    ###############################################
    ### Main Page
    ###############################################
    
    with st.beta_expander('Search'):
        st.write('Note: Currently key-word search only... to be superseded by "SinguSearch"')
        by_name = st.checkbox("by Name", True)
        by_org = st.checkbox("by Org", False)
        by_text = st.checkbox("by Text", False)

        query = st.text_area("Search Field", "")

        if st.button("Search"):
            st.subheader('Search Results')
            st.write(search_emails(df=data, query=query, by_name=by_name, by_org=by_org, by_text=by_text))
    
    ###############################################
    
    st.header('Persons of Interest')
    
    poi = st.multiselect(
        'Persons of interest', 
        list(state.node_dict.inv[x] for x in state.poi), 
        default=list(state.node_dict.inv[x] for x in state.poi)
    )
    state.poi = [state.node_dict[x] for x in poi]
    
    start_date, end_date = st.slider(
        "Select Date Range:",
        min_value=datetime(2001, 1, 1),
        max_value=datetime(2002, 1, 1),
        value=(datetime(2001, 5, 1), datetime(2001, 7, 1)),
        format="MM/DD/YY"
    )
    
    _assoc = filter_nodes_by_date(state.assoc, state.graph.edges, start_date=start_date, end_date=end_date)
    
    with st.beta_expander('Filter by Org'):
        _org = get_orgs(_assoc, state.graph.nodes)
        org = st.multiselect('Linked organizations', sorted(_org), default=_org[:5])
        
    with st.beta_expander('Filter by Assoc'):
        _assoc = filter_nodes_by_org(_assoc, state.graph.nodes, org)

        assoc = st.multiselect(
            'Linked persons', 
            list(state.node_dict.inv[x] for x in _assoc), 
            default=list(state.node_dict.inv[x] for x in _assoc)[:300]
        )
    
    state.subgraph = state.graph.make_subgraph(
        f1=state.poi, 
        f2=[state.node_dict[x] for x in assoc],
        start_date=start_date, end_date=end_date
    )
    
    st.write('Found {} records'.format(len(assoc)))
    
    with st.beta_expander('Show Graph'):
        if len(state.poi)==0:
            st.write('Before drawing a graph you must add a person of interest using the sidebar')
        else:
            G = nx.from_pandas_edgelist(
                state.subgraph.edges, 
                'sender', 
                'receiver1', 
                edge_attr=['datetime', 'desc'], 
                create_using=nx.MultiDiGraph
            )
#             nx.set_node_attributes(G, dict(map(lambda x: (x, {'name':x.split('@')[0], 'org': x.split('@')[-1]}), G.nodes)))
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,10))
            plt.figure()
            pos = nx.spring_layout(G, k=.3)
#             names = nx.get_node_attributes(G, 'name')
            nx.draw_networkx(G, ax=ax, pos=pos, node_size=150, node_color='red', with_labels=True, edge_color='blue')
            st.write(fig)
            st.write(G.edges)
            st.write(state.subgraph.edges)
            
#     st.header('Inspect Evidence')

#     sender = st.selectbox(
#         "Sender:", 
#         list(state.subgraph.sender.unique())
#     )

#     state.text_view = get_text_view(state.subgraph, sender)

#     email = st.selectbox(
#         "Receiver:", 
#         list(state.text_view.keys())
#     )
    
#     col1, col2 = st.beta_columns(2)
    
    
#     payment = col1.selectbox(
#         "Payments", 
#         [1,2,3,4]
#     )
#     col1.text("Test payment")

#     if email is not None:
#         col2.text("Text:")
#         col2.write(state.text_view[email])

#     with st.beta_expander('Find Similar Examples'):

#         top_k = st.slider('Number of Examples', 1, 20, 10)
#         by_bert = st.checkbox("by BERT", True)
#         by_topic = st.checkbox("by Topic", False)

#         if st.button("Find Similar"):
#             st.write('Not yet implemented')

            
    


if __name__ == "__main__":
    run()
