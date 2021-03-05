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
        receiver = receiver_row['name1'] if receiver_row['name1'] is not 'invoice' else receiver_row['org1']
        return row['datetime'].strftime("%y/%m/%d"), sender, receiver, row['desc'][:10]
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


from src.utils import load_pickle, load_numpy

def get_similarity_dict(edge_idx, idx_dict_inv, emb_arr):
    emb_query = emb_arr[idx_dict_inv[edge_idx]]
    similarities = emb_arr@emb_query.T/(np.linalg.norm(emb_arr, axis=-1)*np.linalg.norm(emb_query))
    return dict(zip(idx_dict_inv.keys(), similarities))

def get_similarity_dict2(edge_idx, idx_dict_inv, emb_arr):
    emb_query = emb_arr[idx_dict_inv[edge_idx]]
    print(emb_query)
    similarities = 1-0.5*np.abs(emb_query-emb_arr)
    return dict(zip(idx_dict_inv.keys(), similarities.squeeze()))

    
def run():
    idx_dict = load_pickle('test3.pkl')
    idx_dict_inv = {v:k for k,v in idx_dict.items()}
    embs = load_numpy('test3.npy')
    
    idx_dict2 = load_pickle('testinv.pkl')
    idx_dict_inv2 = {v:k for k,v in idx_dict2.items()}
    embs2 = load_numpy('testinv.npy')
    embs2a = load_numpy('testinvamt.npy')
    
    state = SessionState.get(
        graph=ForensicGraph.from_emails('data/enron/emails_filtered.csv'), subgraph=PGraph(), 
        node_dict=DictInv(), edge_dict=DictInv(), 
        merge_dict=DictInv(), merge_cands=[],
        poi=[], assoc=[], org=[],
        show_dict=DictInv()
    )
    
    ###############################################
    ### Sidebar
    ###############################################
    
    st.sidebar.header('Persons of Interest')
    
    poi_find = st.sidebar.text_area(
        "Add person names separated by commas", 
        "kenneth lay"
    )
    
    st.sidebar.write('Hint: Also try searching for "jeff skilling"')

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
        add_idxs = sorted([idx for cand in state.merge_cands for idx in cand])
        
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
    
    st.sidebar.header('Search')
    by_name = st.sidebar.checkbox("by Name", True)
    by_org = st.sidebar.checkbox("by Org", False)
    by_text = st.sidebar.checkbox("by Text", False)
    st.sidebar.write('Removed: To be superseded by "SinguSearch"')
    
    ###############################################
    ### Main Page
    ###############################################
    
    st.header('Persons of Interest')
    
    _poi_keys = list(map(state.node_dict.get, state.poi))
    poi_keys = st.multiselect('Persons of interest', _poi_keys, default=_poi_keys)
    state.poi = [state.node_dict.inv[x] for x in poi_keys]
    
    start_date, end_date = st.slider(
        "Select Date Range:", min_value=datetime(1999, 1, 1), max_value=datetime(2002, 1, 1),
        value=(datetime(2000, 12, 11), datetime(2001, 10, 1)), format="MM/DD/YY"
    )

    _assoc = filter_nodes_by_date(state.assoc, state.graph.edges, start_date=start_date, end_date=end_date)
    
    with st.beta_expander('Filter by Org'):
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
    
    ###############################################
    
    st.header('Inspect Evidence')

    with st.beta_expander('Show Graph'):
        if len(state.poi)==0:
            st.write('Before drawing a graph you must add a person of interest using the sidebar')
        else:
            G = graph_to_networkx(state.subgraph)
            st.write(draw_graph(G))

    _senders = list(sorted(set([x for x in state.subgraph.edges['sender'].tolist() if x!=''])))
    _sender_keys = list(map(state.node_dict.get, _senders))
    sender_keys = st.multiselect('Sender:', _sender_keys, default=_sender_keys)

    if len(sender_keys)>0:
        sender_idxs = sorted([k for k,v in state.node_dict.items() if v in sender_keys])
        _edges = state.subgraph.edges[state.subgraph.edges['sender'].isin(sender_idxs)]
        
        _receivers = list(sorted(set([x for x in _edges['receiver1'].tolist() if x!=''])))
    else:
        _receivers = []
        
    _receiver_keys = list(map(state.node_dict.get, _receivers))
    receiver_keys = st.multiselect('Receiver:', _receiver_keys, default=_receiver_keys)
    
    if len(sender_keys)>0 and len(receiver_keys)>0:
        sender_idxs = sorted([k for k,v in state.node_dict.items() if v in sender_keys])
        receiver_idxs = sorted([k for k,v in state.node_dict.items() if v in receiver_keys])
        df_show = state.subgraph.edges[state.subgraph.edges['sender'].isin(sender_idxs)]
        df_show = df_show[df_show['receiver1'].isin(receiver_idxs)]
        state.show_dict.update(edge_summaries(df_show.index.tolist(), state.subgraph.nodes, df_show))
    else:
        df_show = pd.DataFrame(columns=state.subgraph.edges.columns)
        state.show_dict = DictInv()
        
    df_show.datetime = df_show.datetime.apply(lambda x: x.strftime("%y/%m/%d"))
    df_emails = df_show[df_show['type']=='email']
    df_invoices = df_show[df_show['type']=='invoice']
    df_ccs = df_show[df_show['type']=='cc']
        
    col1, col2, col3 = st.beta_columns(3)
    
    email = col1.selectbox(
        "E-mails", 
        [state.show_dict[x] for x in df_emails.index.tolist()]
    )
    
    if email is not None:
        col1.write('E-mail edge-id: {}'.format(state.show_dict.inv[email]))
        col1.write(df_show[['datetime', 'desc', 'data']].loc[state.show_dict.inv[email]].to_dict())
    
    invoice = col2.selectbox(
        "Invoices", 
        [state.show_dict[x] for x in df_invoices.index.tolist()]
    )
    if invoice is not None:
        col2.write('invoice edge-id: {}'.format(state.show_dict.inv[invoice]))
        col2.write(df_show[['datetime', 'desc', 'data']].loc[state.show_dict.inv[invoice]].to_dict())
    
    cc = col3.selectbox("Card Purchase", [])
    
    if cc is not None:
        col3.text("Example credit card statement entry")
    
    ###############################################
    
    with st.beta_expander('Similarity Search', expanded=True):
        st.subheader('Evidence Type:')
        col4, col5, col6 = st.beta_columns(3)
        
        thru_email = col4.checkbox("E-mails", True)
        thru_invoice = col5.checkbox("Invoices", True)
#         thru_ccstate = col6.checkbox("Card Statements", False)
        
        st.subheader('Similarity Metrics:')
        
        col7, col8, col9 = st.beta_columns(3)
        
        if thru_email:        
            by_bert = col7.checkbox("by BERT", True)
#             by_topic = col7.checkbox("by Topic", False)
        if thru_invoice:   
            by_desc_inv = col8.checkbox("by Inv. Desc.", True)
            by_amt_inv = col8.checkbox("by Inv. Amount", True)
            
#         if thru_ccstate:   
#             by_desc_cc = col9.checkbox("by CC Desc.", True)
#             by_amt_cc = col9.checkbox("by CC Amount", False)
            
        
        top_k = st.slider('Number of Examples', 1, 20, 5)
        

        if st.button("Find Similar"):
            col10, col11, col12 = st.beta_columns(3)
            if thru_email and email is not None:
                query_idx = state.show_dict.inv[email]
                scores = dict(
                    sorted(
                        get_similarity_dict(query_idx, idx_dict_inv, embs).items(), key=lambda x: x[1], reverse=True
                    )[1:top_k]
                )
                show_list = state.graph.edges[['sender', 'receiver1', 'datetime', 'desc', 'data']].loc[list(scores.keys())]
                show_list.sender = show_list.sender.apply(
                    lambda x: state.graph.nodes.loc[int(x)].email1 if x!='' else ''
                )
                show_list.receiver1 = show_list.receiver1.apply(
                    lambda x: state.graph.nodes.loc[int(x)].email1 if x!='' else ''
                )
                show_list.datetime = show_list.datetime.apply(lambda x: x.strftime("%y/%m/%d"))
                show_dict = show_list.T.to_dict()
                col10.write({'{}'.format(scores[k]):show_dict[k] for k in list(scores.keys())})
                
            if thru_invoice and invoice is not None:
                query_idx2 = state.show_dict.inv[invoice]

                if by_desc_inv and not by_amt_inv:
                    scores2 = dict(
                        sorted(
                            get_similarity_dict(query_idx2, idx_dict_inv2, embs2).items(), key=lambda x: x[1], reverse=True
                        )[1:top_k]
                    )
                    
                elif not by_desc_inv and by_amt_inv:
                    scores2 = dict(
                        sorted(
                            get_similarity_dict2(query_idx2, idx_dict_inv2, embs2a).items(), key=lambda x: x[1], reverse=True
                        )[1:top_k]
                    )
                    
                elif by_desc_inv and by_amt_inv:
                    _scores2 = get_similarity_dict(query_idx2, idx_dict_inv2, embs2)
                    _scores2a = get_similarity_dict2(query_idx2, idx_dict_inv2, embs2a)
                    scores2 = dict(
                        sorted(
                            {k:np.mean([v, _scores2a[k]]) for k,v in _scores2.items() if k in _scores2a}.items(), 
                            key=lambda x: x[1], reverse=True
                        )[1:top_k]
                    )
                    
                else:
                    scores2 = {}

                show_list2 = state.graph.edges[['sender', 'receiver1', 'datetime', 'desc', 'data']].loc[list(scores2.keys())]
                show_list2.sender = show_list2.sender.apply(
                    lambda x: state.graph.nodes.loc[int(x)].name1 if x!='' else ''
                )
                show_list2.receiver1 = show_list2.receiver1.apply(
                    lambda x: state.graph.nodes.loc[int(x)].org1 if x!='' else ''
                )
                show_list2.datetime = show_list2.datetime.apply(lambda x: x.strftime("%y/%m/%d"))
                show_dict2 = show_list2.T.to_dict()
                col11.write({'{}'.format(scores2[k]):show_dict2[k] for k in list(scores2.keys())})

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



            
    


if __name__ == "__main__":
    run()
