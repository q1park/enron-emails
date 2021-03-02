import streamlit as st
st.set_page_config(layout="wide") 

import SessionState
import re
import glob
        
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from collections import Counter

from src.structures import DictInv, PGraph
from src.forensic import filterby_date, ForensicGraph
from src.nxutils import graph_to_networkx, draw_graph, make_circos, get_centrality, get_betweenness

edge_bounds = ['sender', 'receiver1', 'receiver2', 'receiver3']
DEFAULT_ORGS=['mindspring', 'mediaone', 'as-coa', 'weforum', 'bellsouth', 'gte', 'i2']

def get_text_view(df, sender):
    text_view = {}
    
    for i,row in df[df.sender==sender].iterrows():
        key = format_text_view_keys(row.date, row.subject, row.recipient1)
        text_view[key] = row.text
    return text_view
    
def edge_summaries(idxs, nodes, edges):
    def summarize(row):
        sender_row = nodes.loc[int(row['sender'])]
        receiver_row = nodes.loc[int(row['receiver1'])]
        sender = sender_row['name1'] if sender_row['name1'] is not '' else sender_row['email1'].split('@')[0]
        receiver = receiver_row['name1'] if receiver_row['name1'] is not '' else receiver_row['email1'].split('@')[0]
        return sender, receiver, row['datetime'].date(), row['desc'][:10]
    return {idx:summarize(edges.loc[idx]) for idx in idxs}

def get_orgs(idxs, nodes):
    return [x[0] for x in Counter(nodes.loc[idxs]['org1']).most_common() if len(x[0])>0]

def filter_nodes_by_org(idxs, nodes, org):
    filtered_nodes = nodes.loc[idxs]
    return filtered_nodes[filtered_nodes['org1'].isin(org)].index

def filter_nodes_by_date(idxs, edges, start_date, end_date):
    edges = filterby_date(edges, start_date=start_date, end_date=end_date)
    all_nodes = [int(x) for x in edges[edge_bounds].values.reshape(-1) if x!='']
    return [x for x in idxs if x in all_nodes]




    
def run():
    state = SessionState.get(
        graph=ForensicGraph.from_emails('data/enron/emails_filtered.csv'), subgraph=PGraph(), 
        node_dict=DictInv(), edge_dict=DictInv(), merge_dict=DictInv(),
        poi=[], assoc=[], org=[], merge_cands=[]
    )
    
    ###############################################
    ### Sidebar
    ###############################################
    
    st.sidebar.header('Persons of Interest')
    
    poi_find = st.sidebar.text_area(
        "Add person names separated by commas", 
        "kenneth lay"#, jeff skilling"
    )

    if st.sidebar.button("Find Persons of Interest"):
        merge_cands = [
            dict(sorted(state.graph.node_summaries(*x).items(), key=lambda x: x[1])) 
            for x in state.graph.search_names(poi_find)
        ]
        
        state.merge_cands = [list(x.keys()) for x in merge_cands]
        state.merge_dict.update({k:v for cand in merge_cands for k,v in cand.items()})
        
    ###############################################
        
    for i, x in enumerate(state.merge_cands):
        if len(x)>=2:
            _merge_keys = list(map(state.merge_dict.get, x))
            merge_keys = st.sidebar.multiselect('Suggestion {}'.format(i), _merge_keys, default=_merge_keys)

            if st.sidebar.button("Merge {}".format(i+1)):
                merge_idxs = sorted([k for k,v in state.merge_dict.items() if v in merge_keys])
                state.graph.merge(*merge_idxs)
                
                state.merge_cands[i] = merge_idxs[0:1]
                state.merge_dict.update(state.graph.node_summaries(*merge_idxs[0:1]))
                SessionState.rerun()
    
    _add_keys = list(map(state.merge_dict.get, [x for cand in state.merge_cands for x in cand]))
    add_keys = st.sidebar.multiselect('Persons List', _add_keys, default=_add_keys)
    
    if st.sidebar.button("Add Persons of Interest"):
        add_idxs = sorted([k for k,v in state.merge_dict.items() if v in add_keys])
        poi_dict = state.graph.node_summaries(*add_idxs)
        assoc_dict = state.graph.assoc_summaries(*add_idxs)

        state.node_dict.update(poi_dict)
        state.node_dict.update(assoc_dict)
        state.poi.extend(list(poi_dict.keys()))
        state.assoc.extend(list(assoc_dict.keys()))
        
        state.merge_dict = DictInv()
        state.merge_cands = []
        SessionState.rerun()

    ###############################################
    ### Main Page
    ###############################################
    
    with st.beta_expander('Search'):
        st.write('Removed: To be superseded by "SinguSearch"')
    
    ###############################################
    
    st.header('Persons of Interest')
    
    _poi_keys = list(map(state.node_dict.get, state.poi))
    poi_keys = st.multiselect('Persons of interest', _poi_keys, default=_poi_keys)
    state.poi = [state.node_dict.inv[x] for x in poi_keys]
    
    start_date, end_date = st.slider(
        "Select Date Range:", min_value=datetime(2001, 1, 1), max_value=datetime(2002, 1, 1),
        value=(datetime(2001, 5, 1), datetime(2001, 9, 1)), format="MM/DD/YY"
    )

    _assoc = filter_nodes_by_date(state.assoc, state.graph.edges, start_date=start_date, end_date=end_date)
    
    with st.beta_expander('Filter by Org', expanded=True):
        _org = get_orgs(_assoc, state.graph.nodes)
        org = st.multiselect('Linked organizations', sorted(_org), default=[x for x in DEFAULT_ORGS if x in _org])
        
    with st.beta_expander('Filter by Assoc'):
        _assoc_keys = list(map(state.node_dict.get, filter_nodes_by_org(_assoc, state.graph.nodes, org)))
        assoc_keys = st.multiselect('Linked persons', _assoc_keys, default=_assoc_keys[:300])
    
    state.subgraph = state.graph.make_subgraph(
        f1=state.poi, f2=list(map(state.node_dict.inv.get, assoc_keys)),
        start_date=start_date, end_date=end_date
    )
    
    st.write('Found {} records'.format(len(state.subgraph.edges)))
    
    with st.beta_expander('Show Graph', expanded=True):
        if len(state.poi)==0:
            st.write('Before drawing a graph you must add a person of interest using the sidebar')
        else:
            G = graph_to_networkx(state.subgraph)
            st.write(draw_graph(G))
            
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
