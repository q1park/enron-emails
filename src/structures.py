from collections import Counter

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
    
    