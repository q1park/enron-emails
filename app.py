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

def filter_poi(df, poi):
    return df[(df.sender.isin(poi))|(df.recipient1.isin(poi))|(df.recipient2.isin(poi))|(df.recipient3.isin(poi))]

    
def run():
    data = pd.read_csv('data/enron/emails_small.csv').fillna('')
    state = SessionState.get(data_view=pd.DataFrame(), data_graph=pd.DataFrame(), poi=[])
    
    st.sidebar.header('Keyword Search')

    by_name = st.sidebar.checkbox("by Name", True)
    by_org = st.sidebar.checkbox("by Org", False)
    by_text = st.sidebar.checkbox("by Text", False)
    
    query = st.sidebar.text_area("Search Field", "ljm")

    if st.sidebar.button("Search"):
        if len(state.data_view)>0:
            state.data_view = pd.DataFrame()
            
        if by_name:
            state.data_view = pd.concat([
                state.data_view,
                data[data.sender.apply(lambda x: x.split('@')[0]).str.contains(query)]
            ])
            
        if by_org:
            state.data_view = pd.concat([
                state.data_view,
                data[data.sender.apply(lambda x: x.split('@')[-1]).str.contains(query)]
            ])
                
        if by_text:
            state.data_view = pd.concat([
                state.data_view,
                data[data.text.str.contains(query)]
            ])
        
    st.sidebar.write('Hint: Try searching for ["ljm"](https://en.wikipedia.org/wiki/Enron_scandal#LJM_and_Raptors) and notice the recipient')
    st.sidebar.write('')
    st.sidebar.header('Graph Builder')
    
    poi_add = st.sidebar.text_area("Add person and org in 'person@org' format", "larry.may@enron")

    if st.sidebar.button("Add"):
        state.poi.append(poi_add)
            
    state.poi = st.sidebar.multiselect('Persons of interest', state.poi, default=state.poi)
    state.data_graph = filter_poi(data, state.poi)
        
    st.sidebar.write('Hint: larry.may@enron, jeffrey.gossett@enron, errol.mclaughlin@enron')
    
    st.title('Search Results')
    
    st.write(state.data_view)

    st.title('View Content')
    
    st.write('Under Construction')
#     st.subheader('Select Content')
    
#     sender = st.selectbox(
#         "Sender:", 
#         list(state.data_graph.sender.unique())
#     )
    
#     recipient = st.selectbox(
#         "Recipient:", 
#         list(state.data_graph[state.data_graph.sender.isin([sender])].recipient1.unique())
#     )
#     st.text("Text:")
#     st.write("--")
    
#     st.subheader('Find Similar')
    
#     by_bert = st.checkbox("by BERT", True)
#     by_topic = st.checkbox("by Topic", False)
    
#     if st.button("Find Similar"):
#         st.write('Not yet implemented')
    
    
    
    st.title('Graph')
    
    if st.button("Draw Graph"):
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
