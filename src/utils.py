import json
import pickle
import numpy as np

def save_numpy(data, path):
    with open(path, 'wb') as f:
        np.save(f, data)
        
def load_numpy(path):
    with open(path, 'rb') as f:
        return np.load(f)

#####################################################################################
### Functions load and save json
#####################################################################################

def int_keys(ordered_pairs):
    result = {}
    for key, value in ordered_pairs:
        try:
            key = int(key)
        except ValueError:
            pass
        result[key] = value
    return result

def load_json(path):
    with open(path, 'r') as f: 
        pydict = json.loads(f.read(), object_pairs_hook=int_keys)
    return pydict

def save_json(pydict, path):
    with open(path, 'w') as f: 
        f.write(json.dumps(pydict))
    
#####################################################################################
### Functions to load and save pickles
#####################################################################################

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)
    
def save_pickle(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)