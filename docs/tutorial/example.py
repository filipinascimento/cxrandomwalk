import numpy as np
import cxrandomwalk as rw
# from tqdm.auto import tqdm 

vertexCount = 10;
edges = np.random.randint(0,vertexCount-1,(vertexCount*2, 2))
weights = np.random.random(size=vertexCount*2);

agent = rw.Agent(vertexCount,edges,False,weights)

# def make_pbar():
#   pbar = None
#   def inner(current,total):
#     nonlocal pbar
#     if(pbar is None):
#       pbar= tqdm(total=total);
#     pbar.update(current - pbar.n)

#   return inner
print(agent.generateWalks(p=2,q=3,verbose=True))

# print(len(agent.generateWalks(q=2,p=3,verbose=False,updateInterval=1000,callback=make_pbar())))
