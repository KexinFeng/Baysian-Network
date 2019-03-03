import numpy as np
import re, copy
import random

class BNode:
    def __init__(self, label):
        self.label = label
        self.conditions = []
        self.back_list = [] # receivors
        self.cpt = {}
        # cpt = {hashkey: prob}


    def buildCPT(self, logic, prob):
        # logic = {var: 1/0, var: 1/0}
        if not self.conditions:
            self.conditions = list(logic.keys())

        self.cpt[self.vec_hash(logic)] = prob

    def vec_hash(self, query_logic):
        # query_logic: {var: 1/0}
        if not query_logic:
        # -1: unconditional
            return -1

        hashkey = 0
        for var in self.conditions:
            hashkey = hashkey * 2
            hashkey = hashkey + query_logic[var]
        return hashkey

    def prob(self, feed_dict):
        return self.cpt[self.vec_hash(feed_dict)]
    def get_CPT(self, seq):
        feed_dict = {}
        for c in self.conditions:
            feed_dict[c] = seq[c]
        return self.prob(feed_dict)


    def tensor_rep(self, evidence = None):
        # evidence = {var: boolean} denotes the exemption
        # return factor, dim
        # factor is the tensor, dim stores dimension name
        rank = len(self.conditions) + 1
        if evidence:
            for e in evidence:
                if e in self.conditions:
                    rank = rank - 1


        factor = np.zeros([2 for idx in range(rank)])

        # dim = dict((self.label, 0))
        dim = {}
        dim[self.label] = 0
        i=1
        for var in self.conditions:
            if evidence and var in evidence:
                continue
            dim[var] = i
            i += 1

        logic = {}
        index = [0]
        # recursively find the leaves, store the logic and index along
        # logic is for reading cpt, index is for indexing factor
        self.traverse_Ltree(self.conditions, factor, evidence, logic, index)

        factor[1] = 1 - factor[0]

        assert np.ndim(factor) == rank
        print('rank = ', rank)

        return factor, dim

    def traverse_Ltree(self, vars, factor, evidence, logic, index):
        # called by tenor_rep()
        # index = [0, 1, 0, 0]
        # logic = {var: 0}

        if not vars:
            factor[tuple(index)] = self.cpt[self.vec_hash(logic)]
            return

        vars_cp = copy.copy(vars)
        logic_cp = copy.copy(logic)

        # var projection
        var = vars_cp.pop(0)
        while evidence and var in evidence:
            # index = [index, evidence[var]]
            logic_cp[var] = evidence[var]
            if not vars_cp:
                factor[tuple(index)] = self.cpt[self.vec_hash(logic_cp)]
                return
            var = vars_cp.pop(0)

        index_left = index + [0]
        logic_cp[var] = 0
        self.traverse_Ltree(vars_cp, factor, evidence, logic_cp, index_left)

        index_right = index + [1]
        logic_cp[var] = 1
        self.traverse_Ltree(vars_cp, factor, evidence, logic_cp, index_right)





class Solution:
    def __init__(self):
        # self.bn_graph = []
        self.bn_graph = {} # node -> BNode

    #######################################################
    def readData(self, file):
        with open(file, "r") as fr:
            lines = fr.readlines()
            # process line-> node {conditions: boolean} prob
            for line in lines:
                terms = line.split()
                if len(terms) == 0:
                    continue

                prob = float(terms[2])
                tmp = re.split('\||\(|,|\)', terms[0])
                node = tmp[1]
                conditions = tmp[2:-1]
                dict = self.build_dict(conditions)

                self.build_graph_dynamic(node, dict, prob)

            self.BFS(self.build_back_list)

    def build_back_list(self, node):
        for child in node.conditions:
            # add node to each child's back-list
            self.bn_graph[child].back_list.append(node.label)

    @staticmethod
    def build_dict(conditions):
        dict = {}
        for c in conditions:
            value = 0 if len(c) == 2 else 1
            # 1: var, 0: ~var
            key = c[-1]
            dict[key] = value

        return dict

    def build_graph_dynamic(self, node, dict, prob):
        if not node in self.bn_graph:
            self.bn_graph[node] = BNode(node)

        self.bn_graph[node].buildCPT(dict, prob)


    #######################################################
    # verification
    def BFS(self, visit):
        # traverse the graph
        # call visit on each node
        # return a list of visit record
        if len(self.bn_graph.values()) == 0:
            print('None')
            return None

        incount = dict((x.label, 0) for x in self.bn_graph.values())
        for node in self.bn_graph.values():
            for neighbor in node.conditions:
                incount[neighbor] += 1


        queue = []
        for key, val in incount.items():
            if val == 0:
                queue.append(key)

        record = []
        while queue:
            label = queue.pop(0)
            g = self.bn_graph[label]
            rec = visit(g)
            if rec:
                record.append(rec)

            for neighbor in g.conditions:
                incount[neighbor] -= 1
                if incount[neighbor] == 0:
                    queue.append(neighbor)
        return record


    def BFS_rev(self, visit):
        # count outcount, only when i has no outgoing arrow, start to add the node
        # reverse topological order
        if len(self.bn_graph.values()) == 0:
            print('None')
            return None

        incount = dict((x.label, 0) for x in self.bn_graph.values())
        for node in self.bn_graph.values():
            for n in node.back_list:
                incount[n] += 1

        queue = []
        for key, val in incount.items():
            if val == 0:
                queue.append(key)

        record = []
        while queue:
            node = self.bn_graph[queue.pop(0)]
            rec = visit(node)
            if rec:
                record.append(rec)

            for neighbor in node.back_list:
                incount[neighbor] -= 1
                if incount[neighbor] == 0:
                    queue.append(neighbor)
        return record


    @staticmethod
    def print_node(g):
        print(g.label)
        print(g.conditions)
        print(g.cpt)
        print(g.tensor_rep({'e': 0}))
        print('---------------------------------------')

    @staticmethod
    def read_label(g):
        return g.label

    #######################################################
    # Application1

    def likelihood_weighting(self, qvar, feed_dict, N):
        # qvar = [string]
        num = len(qvar)
        w_tot = np.zeros((num, 2))
        # [var * prob]

        for n in range(N):
            seq, w = self.weighted_sample(feed_dict)
            for row, var in enumerate(qvar):
                # [row, 0/1] ~var: 0, var: 1
                w_tot[row, seq[var]] += w
        w_tot = w_tot / np.sum(w_tot, 1)[:, None]

        return w_tot

    def weighted_sample(self, feed_dict):
        w = 1
        seq = {} # {var: logic_val} which is a vector
        order = self.BFS_rev(self.read_label)

        for var in order:
            node = self.bn_graph[var]
            parent_logic = {}

            if feed_dict and var in feed_dict.keys():
                # evidence variable
                for par_var in node.conditions:
                    # read the parent(var) condition
                    assert par_var in seq.keys()
                    parent_logic[par_var] = seq[par_var]
                if feed_dict[var] == 1:
                    w = w * node.prob(parent_logic)
                else:
                    w = w * (1 - node.prob(parent_logic))
                seq[var] = feed_dict[var]
            else:
                # non-evidence variable
                for par_var in node.conditions:
                    assert par_var in seq.keys()
                    parent_logic[par_var] = seq[par_var]
                seq[var] = 1 if random.random() < node.prob(parent_logic) else 0

        return seq, w


    #######################################################
    # Application 2
    def Gibbs_sampling(self, qvar, feed_dict, N):
        seq = {}
        nevidence_vars = []
        for x in self.bn_graph.keys():
            if feed_dict and x in feed_dict:
                seq[x] = feed_dict[x]
                nevidence_vars.append(x)
            else:
                seq[x] = 1 if random.random() > 0.5 else 0
        w_tot = np.zeros(2)
        # 0:~var 1: var

        for n in range(N):
            for var in nevidence_vars:
                seq[var] = 1 if random.random() < self.mb_prob(var, seq) else 0
                w_tot[seq[qvar]] += 1

        return w_tot/np.sum(w_tot)

    def mb_prob(self, var, seq):
        weight = np.zeros(2)
        # 0: ~var 1: var

        node = self.bn_graph[var]
        prob_qnode = node.get_CPT(seq)

        prob = prob_qnode
        dual_prob = 1 - prob_qnode

        seq_cp = copy.copy(seq)
        seq_cp[var] = 0 # set Var = ~var for the dual_prob
        seq_cp2 = copy.copy(seq)
        seq_cp2[var] = 1
        for child in node.back_list:
            cnode = self.bn_graph[child]
            prob *= cnode.get_CPT(seq_cp)
            dual_prob *= cnode.get_CPT(seq_cp2)
        print(prob, dual_prob)

        weight = np.array([prob, dual_prob])
        return weight /np.sum(weight)

    # def var_elim(self, node=None, evidence=None):
    #     factors = None
    #     for var in self.BFS(self.read_label):
    #         # print(var)
    #         factor = self.make_factor(factor, self.bn_graph[var])
    #         if not evidence and self.bn_graph[var] in evidence:
    #             pass
    #         else:
    #             self.sum_out(var, factors)



    @staticmethod
    def make_factor(factor, g):
        if not factor:
            return g
        # point wise product



    @staticmethod
    def sum_out(var, factors):
        return 0






def main():
    print('hello world')
    random.seed(10)

    sol = Solution()
    file = "prob345.txt"
    sol.readData(file)
    # # test graph building
    sol.BFS(sol.print_node)
    #
    # # test topological order
    record = sol.BFS(sol.read_label)
    print(record)

    record = sol.BFS_rev(sol.read_label)
    print(record)

    # test likelihood weighting
    qvar = 'g'
    feed_dict = {'k': 1, 'b':0, 'c':1}
    prob = sol.likelihood_weighting(qvar, feed_dict, 10)
    print(prob)

    # test gibbs likelihood
    qvar = 'g'
    feed_dict = {'k': 1, 'b':0, 'c':1}
    prob = sol.Gibbs_sampling(qvar, feed_dict, 10)
    print(prob)




if __name__ == '__main__':
    main()