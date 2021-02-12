import streamlit as st
st. set_page_config(layout="wide") 
import re
import glob

import SessionState
    
import os
import pickle
    
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from streamlit_agraph import agraph, Node, Edge, Config
from datetime import datetime

def filterby_date(df, start_date='1979-01-01', end_date='2005-01-01'):
    return data[(data.date>=start_date)&(data.date<=end_date)].reset_index(drop=True)

def filter_poi(df, poi):
    return df[(df.sender.isin(poi))|(df.recipient1.isin(poi))|(df.recipient2.isin(poi))|(df.recipient3.isin(poi))]

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

def format_text_view_keys(date, subject, recipient):
    return "{}, {}, {}".format(date, recipient, subject[:15])

def get_text_view(df, sender):
    text_view = {}
    
    for i,row in df[df.sender==sender].iterrows():
        key = format_text_view_keys(row.date, row.subject, row.recipient1)
        text_view[key] = row.text
    return text_view

    
def run():
    data = pd.read_csv('data/enron/emails_small.csv').fillna('')
    state = SessionState.get(data_search=pd.DataFrame(), data_graph=pd.DataFrame(), text_view={}, poi=[])
    
    st.sidebar.header('Keyword Search')
    
    st.sidebar.write('Hint: Try searching by Name for ["ljm"](https://en.wikipedia.org/wiki/Enron_scandal#LJM_and_Raptors) and notice the recipient')

    by_name = st.sidebar.checkbox("by Name", True)
    by_org = st.sidebar.checkbox("by Org", False)
    by_text = st.sidebar.checkbox("by Text", False)
    
    query = st.sidebar.text_area("Search Field", "ljm")

    if st.sidebar.button("Search"):
        state.data_search = search_emails(df=data, query=query, by_name=by_name, by_org=by_org, by_text=by_text)
    
    st.sidebar.write('')
    st.sidebar.header('Graph Builder')
    
    poi_add = st.sidebar.text_area(
        "Add people and orgs as e.g. first.person@org1, second.person@org2", 
        "larry.may@enron, jeffrey.gossett@enron, errol.mclaughlin@enron"
    )

    if st.sidebar.button("Add Persons of Interest"):
        state.poi.extend([x.strip() for x in poi_add.split(',')])
            
    state.poi = st.sidebar.multiselect('Persons of interest', state.poi, default=state.poi)
    state.data_graph = filter_poi(data, state.poi)
    
    st.sidebar.header('Time Filter')
    st.sidebar.write('***Not Yet Implemented***')
    
    start_time = st.sidebar.slider(
        "Select Date Range:",
        min_value=datetime(1990, 1, 1),
        max_value=datetime(2005, 1, 1),
        value=(datetime(1999, 1, 1), datetime(2002, 1, 1)),
        format="MM/DD/YY"
    )
    
    st.title('Search Results')
    
    st.write(state.data_search)

    st.title('View Content')
    
    st.write('Under Construction')
    st.subheader('Select Content')
    
    sender = st.selectbox(
        "Sender:", 
        list(state.data_graph.sender.unique())
    )
    
    state.text_view = get_text_view(state.data_graph, sender)
    
    email = st.selectbox(
        "Recipient:", 
        list(state.text_view.keys())
    )
    
    if email is not None:
        st.text("Text:")
        st.write(state.text_view[email])
    
    st.subheader('Find Similar Examples')
    
    top_k = st.slider('Number of Examples', 1, 20, 10)
    by_bert = st.checkbox("by BERT", True)
    by_topic = st.checkbox("by Topic", False)
    
    if st.button("Find Similar"):
        st.write('Not yet implemented')
    
    st.title('Graph')
    
    if st.button("Draw Graph"):
        if len(state.poi)==0:
            st.write('Error: Before drawing a graph you must add a person of interest using the Graph Builder in the sidebar')
        else:
            state.data_graph = filter_poi(data, state.poi)
            G = nx.from_pandas_edgelist(
                state.data_graph, 
                'sender', 
                'recipient1', 
                edge_attr=['date', 'subject'], 
                create_using=nx.DiGraph
            )
            nx.set_node_attributes(G, dict(map(lambda x: (x, {'name':x.split('@')[0], 'org': x.split('@')[-1]}), G.nodes)))
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,20))
            plt.figure()
            pos = nx.spring_layout(G, k=.3)
            names = nx.get_node_attributes(G, 'name')
            nx.draw_networkx(G, ax=ax, pos=pos, node_size=150, node_color='red', with_labels=True, edge_color='blue')
            st.write(fig)

#     nodes = []
#     edges = []
#     nodes.append( Node(id="Spiderman", label="Peter Parker", size=400, svg="http://marvel-force-chart.surge.sh/marvel_force_chart_img/top_spiderman.png") ) # includes **kwargs
#     nodes.append( Node(id="Captain_Marvel", size=400, svg="http://marvel-force-chart.surge.sh/marvel_force_chart_img/top_captainmarvel.png") )
#     edges.append( Edge(source="Captain_Marvel", label="friend_of", target="Spiderman", type="CURVE_SMOOTH") ) # includes **kwargs

#     config = Config(width=500, 
#                     height=500, 
#                     directed=True,
#                     nodeHighlightBehavior=True, 
#                     highlightColor="#F7A7A6", # or "blue"
#                     collapsible=True,
#                     node={'labelProperty':'label'},
#                     link={'labelProperty': 'label', 'renderLabel': True}
#                     # **kwargs e.g. node_size=1000 or node_color="blue"
#                     ) 

#     return_value = agraph(nodes=nodes, 
#                           edges=edges, 
#                           config=config)

    


if __name__ == "__main__":
    run()
