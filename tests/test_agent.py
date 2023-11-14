import unittest
import numpy as np
import cxrandomwalk as rw
from tqdm.auto import tqdm 

class TestAgent(unittest.TestCase):
    def setUp(self):
        self.vertexCount = 1000
        self.edges = np.random.randint(0, self.vertexCount-1, (self.vertexCount*2, 2))
        self.weights = np.random.random(size=self.vertexCount*2)
        self.agent = rw.Agent(self.vertexCount, self.edges, False, self.weights)

    def make_pbar(self):
        pbar = None
        def inner(current,total):
            nonlocal pbar
            if(pbar is None):
                pbar= tqdm(total=total)
            pbar.update(current - pbar.n)
        return inner

    def test_generateWalks(self):
        walks = self.agent.generateWalks(q=1.0, p=1.0, verbose=False, updateInterval=1000, callback=self.make_pbar())
        self.assertIsNotNone(walks)
        self.assertTrue(len(walks) > 0)

if __name__ == '__main__':
    unittest.main()