import pickle

def save_to_pickle(item,file_path: str):
    with open(file_path,'wb') as p:
        pickle.dump(item,p)

def read_pickle(file_path: str):
    with open(file_path,'rb') as p:
        item = pickle.load(p)
    return item