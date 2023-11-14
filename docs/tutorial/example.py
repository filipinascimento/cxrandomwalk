import numpy as np
import cxrandomwalk as rw
# from tqdm.auto import tqdm 

vertexCount = 10
edges = np.random.randint(0,vertexCount,(vertexCount*4, 2))
weights = np.random.random(size=vertexCount*4)
names = ["ID %d"%i for i in range(vertexCount)]

agent = rw.Agent(vertexCount,edges,False,weights)

# def make_pbar():
#   pbar = None
#   def inner(current,total):
#     nonlocal pbar
#     if(pbar is None):
#       pbar= tqdm(total=total);
#     pbar.update(current - pbar.n)

print(agent.generateWalks(p=2,q=3,verbose=True,labels=names))
# print(len(agent.generateWalks(q=2,p=3,verbose=False,updateInterval=1000,callback=make_pbar())))


