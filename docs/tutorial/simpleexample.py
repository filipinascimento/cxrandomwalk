import numpy as np
import cxrandomwalk as rw
from tqdm.auto import tqdm 

import pickle
import sys
from scipy import sparse
import time
import multiprocessing as mp
import ctypes

ar = np.array

mp.set_start_method('fork')


def smart_random_walk(network, seednode, length):
    import random
    walk = []
    connections = network[seednode]
    for i in range(length):
        node = random.choice(connections)
        connections = network[node]
        walk.append(node)
    return(walk)

def walkstat(index):
    try:
        focal_node = todo[index]
        estimates = np.zeros(n)
        for j in range(numwalks):
            # print(j)
            walk = smart_random_walk(adjlist, focal_node, wlkl)
            np.add.at(estimates,walk, 1)
        walkhit[index] = estimates
    except Exception as e:
        print(e)
        print("Error in walkstat")
        return None


vertexCount = 1000
sampleSize = 500
windowSize = 20
walksPerNode = 100_000
NCPUs = 10
batchSize = ((walksPerNode*sampleSize)//NCPUs)//NCPUs

wlkl = windowSize
sample_size = sampleSize
numwalks = walksPerNode
ncpu = mp.cpu_count()


# if __name__ == '__main__':

edges = np.random.randint(0,vertexCount,(vertexCount*15, 2))
sampleNodes = np.random.choice(np.arange(vertexCount), sampleSize, replace = False)

adjlist = [[] for i in range(vertexCount)]
for edge in edges:
    adjlist[edge[0]].append(edge[1])
    adjlist[edge[1]].append(edge[0])



n =  len(adjlist)
samplepapers = sampleNodes

# Random Walkers
network = adjlist
todo = samplepapers
N = sample_size
# random walks

################
def make_pbar():
    pbar = None
    def inner(current,total):
        nonlocal pbar
        if(pbar is None):
            pbar= tqdm(total=total)
        pbar.update(current - pbar.n)
    return inner

agent = rw.Agent(vertexCount,edges,False)
T = time.time()
hits = agent.walkHits(nodes=list(sampleNodes),
                      q=1.0,
                      p=1.0,
                      walksPerNode=walksPerNode,
                      batchSize=batchSize,
                      windowSize=windowSize,
                      verbose=True,
                      updateInterval=1,)
                    #  callback=make_pbar())
print("hits(RW): ",np.sum(hits))
elapsed = time.time() - T
print("RW calc. time (min): " + str(round((time.time() - T)/60, 2)))

################


# T = time.time()
# SIZE_A = sample_size
# SIZE_B = n
# # create a block of bytes, reshape into a local numpy array
# NBR_ITEMS_IN_ARRAY = SIZE_A * SIZE_B
# shared_array_base = mp.Array(ctypes.c_uint32, NBR_ITEMS_IN_ARRAY, lock=False)
# walkhit = np.frombuffer(shared_array_base, dtype="uint32")
# walkhit = walkhit.reshape(SIZE_A, SIZE_B)
# # assert no copy was made
# assert walkhit.base.base is shared_array_base
# # compute walk hits
# l = list(range(len(todo)))
# pool = mp.Pool(ncpu)
# # need to pass todo and walkhit as arguments to walkstat but no copy should be done
# # parameters = [(iteration, todo, walkhit) for iteration in l]
# results = pool.map_async(walkstat, [iteration for iteration in l])
# # results = pool.map(walkstat, parameters)
# pool.close()
# pool.join()
# elapsed = time.time() - T
# print("hits(RW): ",np.sum(walkhit))
# print("TP calc. time (min): " + str(round((time.time() - T)/60, 2)))
# # Save

# # compare hits and walkhit
# totalDifference = np.sum(np.abs(np.array(hits,dtype=int) - np.array(walkhit,dtype=int)))
# print("Total Difference percentage: ",totalDifference/np.sum(hits)*100.0)

