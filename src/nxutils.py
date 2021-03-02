import random
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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

def graph_to_networkx(graph):
    G = nx.MultiDiGraph()
    
    for i, row in graph.nodes.iterrows():
        G.add_nodes_from([(i, {
            'label':row['name1'] if len(row['name1'])>0 else row['email1'].split('@')[0],
            'org':row['org1']
        })])
        
    for i, row in graph.edges.iterrows():
        if row['sender']=='' or row['receiver1']=='':
            continue
            
        G.add_edges_from([(int(row['sender']),int(row['receiver1']),{
            'type':row['type'],
            'date':row['datetime'].date,
            'desc':row['desc'],
            'data':row['data']
        })])
    return G

def grouped_layout(G, rad = 3.5):
    random.seed(7)
    colors = list(mcolors.CSS4_COLORS.keys())
    random.shuffle(colors)

    node_network_map = nx.get_node_attributes(G, 'org')
    networks = sorted(list(set(node_network_map.values())))
    color_map = dict(zip(networks, colors[:len(networks)]))
    
    enron_nodes = [k for k,v in node_network_map.items() if v=='enron']
    print(node_network_map)
    
    nodes_by_color = {
        val: [node for node in G if node in node_network_map and color_map[node_network_map[node]] == val]
        for val in colors
    }
    
    pos = nx.circular_layout(G)   # replaces your original pos=...
    # prep center points (along circle perimeter) for the clusters
    angs = np.linspace(0, 2*np.pi, 1+len(networks))
    repos = []
    
    for ea in angs:
        if ea > 0:
#             print(rad*np.cos(ea), rad*np.sin(ea))  # location of each cluster
            repos.append(np.array([rad*np.cos(ea), rad*np.sin(ea)]))

    color_pos = dict(zip(nodes_by_color.keys(), range(len(nodes_by_color))))

    for ea in pos.keys():
        posx = 0

        for c, p in color_pos.items():
            if ea in nodes_by_color[c]:
                posx = p
        if ea in enron_nodes:
            pass
        else:
            pos[ea] += repos[posx]
            
    return pos, nodes_by_color

def draw_graph(G):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,10))
    plt.figure()

    pos, nodes_by_color = grouped_layout(G, rad=3.5)

    for color, node_names in nodes_by_color.items():
        nx.draw_networkx_nodes(
            G, ax=ax, pos=pos, nodelist=node_names, node_color=color, label={x:G.nodes[x]['label'] for x in G.nodes}
        )

    nx.draw_networkx_edges(
        G, ax=ax, pos=pos, edgelist=[x for x in G.edges if G.edges[x]['type']=='email'], 
        edge_color='blue', connectionstyle='arc3, rad = 0.1'
    )
    nx.draw_networkx_edges(
        G, ax=ax, pos=pos, edgelist=[x for x in G.edges if G.edges[x]['type']=='invoice'], 
        edge_color='red', connectionstyle='arc3, rad = 0.1'
    )
    nx.draw_networkx_labels(
        G, ax=ax, pos=pos, labels={x:G.nodes[x]['label'] for x in G.nodes}
    )
    return fig