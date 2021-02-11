import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from circos import CircosPlot

def make_circos(G, ax):
    nodes = sorted(G.nodes())
    edges = G.edges()
    edgeprops = dict(alpha=0.1)
    nodecolor = plt.cm.viridis(np.arange(len(nodes)) / len(nodes)) 
    return CircosPlot(nodes, edges, radius=10, ax=ax, edgeprops=edgeprops, nodecolor=nodecolor)

def get_centrality(G):
    cent = nx.degree_centrality(G)
    name = []
    centrality = []

    for key, value in cent.items():
        name.append(key)
        centrality.append(value)

    cent = pd.DataFrame()    
    cent['name'] = name
    cent['centrality'] = centrality
    cent = cent.sort_values(by='centrality', ascending=False)
    return cent

def get_betweenness(G):
    between = nx.betweenness_centrality(G)
    name = []
    betweenness = []

    for key, value in between.items():
        name.append(key)
        betweenness.append(value)

    bet = pd.DataFrame()
    bet['name'] = name
    bet['betweenness'] = betweenness
    bet = bet.sort_values(by='betweenness', ascending=False)
    return bet