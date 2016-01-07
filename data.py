import numpy as np

def one_hot(yy,labels=None):
    if labels == None:
        fset = set(yy)
    else:
        fset = set(labels)
    onehotmap = {x: list(map(int, list(bin(2 ** y)[2:].zfill(len(fset))))) for y, x in enumerate(fset)}
    lst = [onehotmap[x] for x in yy]
    return lst




def train_batch_data(data,labels,set_size):
    l = len(data)
    r = np.random.randint(l,size=set_size)
    return data[r], one_hot([labels[x] for x in r],labels)


# a = [1,2,3,4,5,6,7]
# r = np.random.randint(7,size=2)
# print([a[x] for x in r])
