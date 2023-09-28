import numpy as np

from ...base.random import RNG
from ...base import types
from .tree_distribution import TreeProposal
from .tree_target import TreeTarget


class Tree(types.DiscreteVariable):
    def __init__(self, X_train, y_train, tree, leafs, lastAction=""):
        self.X_train = X_train
        self.y_train = y_train
        self.tree = tree
        self.leafs = leafs
        self.lastAction = lastAction
        

    def __eq__(self, x) -> bool:
        return (x.X_train == self.X_train).all() and\
                (x.y_train == self.y_train).all() and\
                x.tree == self.tree and x.leafs == self.leafs \
                and x.lastAction == self.lastAction

    def __str__(self):
        return str(self.tree)

    @classmethod
    def getProposalType(self):
        return TreeProposal

    @classmethod
    def getTargetType(self):
        return TreeTarget
    
    
    def depth_of_leaf(self, leaf):
        depth = 0
        for node in self.tree:
            if node[1] == leaf or node[2] == leaf:
                depth = node[5]+1
                
        return depth
    
    def grow_leaf(self, index, rng=RNG()):
        action = "grow"
        self.lastAction = action
        '''
        grow tree by just simply creating the individual nodes. each node
        holds their node index, the left and right leaf index, the node
        feature and threshold
        '''
        random_index = index
        leaf_to_grow = self.leafs[random_index]

        # generating a random feature
        feature = rng.randomInt(0, len(self.X_train[0])-1)
        # generating a random threshold
        threshold = rng.randomInt(0, len(self.X_train)-1)
        threshold = (self.X_train[threshold, feature])
        depth = self.depth_of_leaf(leaf_to_grow)
        node = [leaf_to_grow, max(self.leafs)+1, max(self.leafs)+2, feature,
                threshold, depth]

        # add the new leafs on the leafs array
        self.leafs.append(max(self.leafs)+1)
        self.leafs.append(max(self.leafs)+1)
        # delete from leafs the new node
        self.leafs.remove(leaf_to_grow)
        self.tree.append(node)

        return self

    def grow(self, rng=RNG()):
        action = "grow"
        self.lastAction = action
        '''
        grow tree by just simply creating the individual nodes. each node
        holds their node index, the left and right leaf index, the node
        feature and threshold
        '''
        random_index = rng.randomInt(0, len(self.leafs)-1)
        leaf_to_grow = self.leafs[random_index]

        # generating a random feature
        feature = rng.randomInt(0, len(self.X_train[0])-1)
        # generating a random threshold
        threshold = rng.randomInt(0, len(self.X_train)-1)
        threshold = (self.X_train[threshold, feature])
        depth = self.depth_of_leaf(leaf_to_grow)
        node = [leaf_to_grow, max(self.leafs)+1, max(self.leafs)+2, feature,
                threshold, depth]

        # add the new leafs on the leafs array
        self.leafs.append(max(self.leafs)+1)
        self.leafs.append(max(self.leafs)+1)
        # delete from leafs the new node
        self.leafs.remove(leaf_to_grow)
        self.tree.append(node)

        return self

    def prune(self, rng=RNG()):
        action = "prune"
        self.lastAction = action
        '''
        For example when we have nodes 0,1,2 and leafs 3,4,5,6 when we prune
        we take the leafs 6 and 5 out, and the
        node 2, now becomes a leaf.
        '''
        random_index = rng.randomInt(0, len(self.tree)-1)
        node_to_prune = self.tree[random_index]
        while random_index == 0:
            random_index = rng.randomInt(0, len(self.tree)-1)
            node_to_prune = self.tree[random_index]

        if (node_to_prune[1] in self.leafs) and\
                (node_to_prune[2] in self.leafs):
            # remove the pruned leafs from leafs list and add the node as a
            # leaf
            self.leafs.append(node_to_prune[0])
            self.leafs.remove(node_to_prune[1])
            self.leafs.remove(node_to_prune[2])
            # delete the specific node from the node lists
            del self.tree[random_index]
        else:

            delete_node_indices = []
            i = 0
            for node in self.tree:
                if node_to_prune[1] == node[0] or node_to_prune[2] == node[0]:
                    delete_node_indices.append(node)
                i += 1
            self.tree.remove(node_to_prune)
            for node in delete_node_indices:
                self.tree.remove(node)

            for i in range(len(self.tree)):
                for p in range(1, len(self.tree)):
                    count = 0
                    for k in range(len(self.tree)-1):
                        if self.tree[p][0] == self.tree[k][1] or\
                                self.tree[p][0] == self.tree[k][2]:
                            count = 1
                    if count == 0:
                        self.tree.remove(self.tree[p])
                        break

        new_leafs = []
        for node in self.tree:
            count1 = 0
            count2 = 0
            for check_node in self.tree:
                if node[1] == check_node[0]:
                    count1 = 1
                if node[2] == check_node[0]:
                    count2 = 1

            if count1 == 0:
                new_leafs.append(node[1])

            if count2 == 0:
                new_leafs.append(node[2])

        self.leafs[:] = new_leafs[:]
        return self

    def change(self, rng=RNG()):
        action = "change"
        self.lastAction = action
        '''
        we need to choose a new feature at first
        we then need to choose a new threshold base on the feature we have
        chosen and then pick unoformly a node and change their features and
        thresholds
        '''
        random_index = rng.randomInt(0, len(self.tree)-1)
        node_to_change = self.tree[random_index]
        new_feature = rng.randomInt(0, len(self.X_train[0])-1)
        new_threshold = rng.randomInt(0, len(self.X_train)-1)
        node_to_change[3] = new_feature
        node_to_change[4] = self.X_train[new_threshold, new_feature]

        return self

    def swap(self, rng=RNG()):
        action = "swap"
        self.lastAction = action
        '''
        need to swap the features and the threshold among the 2 nodes
        '''
        random_index_1 = rng.randomInt(0, len(self.tree)-1)
        random_index_2 = rng.randomInt(0, len(self.tree)-1)
        node_to_swap1 = self.tree[random_index_1]
        node_to_swap2 = self.tree[random_index_2]

        # in case we choose the same node
        while node_to_swap1 == node_to_swap2:
            random_index_2 = rng.randomInt(0, len(self.tree)-1)
            node_to_swap2 = self.tree[random_index_2]

        temporary_feature = node_to_swap1[3]
        temporary_threshold = node_to_swap1[4]

        node_to_swap1[3] = node_to_swap2[3]
        node_to_swap1[4] = node_to_swap2[4]

        node_to_swap2[3] = temporary_feature
        node_to_swap2[4] = temporary_threshold

        return self
