

#credit:
#https://github.com/jaromiru/AI-blog/blob/master/SumTree.py 


import numpy

class SumTree:

    '''
    self.tree is a tree structure(fundamentally numpy array), value is priority score, parents value is the sum of its children. 
    when the value change,u just propagate the changed value up all the way to top.  
    the index of the tree(array position) is related to experience object stored in numpy array. 
    write is a global variale, pointer to data, 
    only the leaves of tree will be used for storing data, the 
                                  0         |
                                /  \        | 
                               0    0       |  capacity above=4-1    sum of all leave= 2*capacity-1
                              / \   / \     |  
                              1  2  3  4     capacity=4
    '''
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = numpy.zeros( 2*capacity - 1 )
        self.data = numpy.zeros( capacity, dtype=object )

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        #print("type is ",type(data))
        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])