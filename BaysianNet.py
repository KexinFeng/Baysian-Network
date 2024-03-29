import numpy as np
import re, copy
import random
import matplotlib.pyplot as plt

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
        # seq is a state of the whole graph
        feed_dict = {}
        query = self.label
        for c in self.conditions:
            assert c in seq # error: not given enough condition
            feed_dict[c] = seq[c]

        default_prob = self.prob(feed_dict)
        return default_prob if seq[query] else 1 - default_prob


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

        self.traverse_Ltree(self.conditions, factor, evidence, logic, index)
        # Recursively find the leaves, store the logic and index along the way.
        # logic is for reading cpt, index is for indexing factor

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

    def readInput(self, term):
        # P(g|a,~b,~c)
        # terms = line.split()
        if len(term) == 0:
            return
        # prob = float(terms[2])
        tmp = re.split('\||\(|,|\)', term)
        q_var = tmp[1]
        conditions = tmp[2:-1]
        dict = self.build_dict(conditions)

        qvar = q_var[-1]
        feed_dict = dict
        dual = 1 if len(q_var) == 1 else 0
        return qvar, feed_dict, dual


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
            for row, qv in enumerate(qvar):
                # [row, 0/1] ~var: 0, var: 1
                w_tot[row, seq[qv]] += w
            # print('one_sample', seq[qvar])
        w_tot = w_tot / np.sum(w_tot, 1)[:, None]

        return w_tot[0]

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
            else:
                nevidence_vars.append(x)
                seq[x] = 1 if random.random() > 0.5 else 0
        w_tot = np.zeros(2)
        # 0:~var 1: var

        # print('feed_dict', feed_dict)
        # print('nevidence_vars:', nevidence_vars)
        # print('init_seq', seq)
        # print(seq.keys())

        pmean = []
        for n in range(N):
            for var in nevidence_vars:
                p = self.mb_prob(var, seq)
                if var == qvar:
                    # print('p=', p)
                    pmean.append(p)
                    # equivalent to a FUZZY sample!

                    # print_dict(seq)
                    # print(p)
                    # print('~~~~~~~~~~~~~~~~~~~~~')

                sample = 1 if random.random() < p else 0
                seq[var] = sample
                # print('one sample', sample)
                w_tot[seq[qvar]] += 1
            # print('seq[qvar]:', seq[qvar])
                # print(w_tot)

        return w_tot/np.sum(w_tot), np.mean(pmean)

    def mb_prob(self, var, seq, test=0):
        weight = np.zeros(2)
        # 0: ~var 1: var

        seq_cp1 = copy.copy(seq)
        seq_cp1[var] = 1  # var
        seq_cp2 = copy.copy(seq)
        seq_cp2[var] = 0  # ~var

        node = self.bn_graph[var]
        # prob_qnode = node.get_CPT(seq)

        prob = node.get_CPT(seq_cp1) # var
        dual_prob = 1 - prob # ~var
        assert dual_prob == node.get_CPT(seq_cp2)
        if test:
            print(dual_prob, prob)
            # print('seq1:',seq_cp1)
            # print('seq2:',seq_cp2)


        for child in node.back_list:
            cnode = self.bn_graph[child]
            ptmp = cnode.get_CPT(seq_cp1)
            prob *= ptmp

            dptmp = cnode.get_CPT(seq_cp2)
            dual_prob *= dptmp
            # print('pair prob: ',prob, dual_prob)
            if test:
                print(dptmp, ptmp)

        # prob = prob_qnode
        # for child in node.back_list:
        #     cnode = self.bn_graph[child]
        #     default_prob = cnode.get_CPT(seq)



        weight = np.array([dual_prob, prob])
        weight = weight /np.sum(weight)

        if test:
            print('weight=', weight)
        return weight[1]

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

def print_dict(seq):
    str = ''
    for key, val in seq.items():
        if not val:
            str = str + '~'
        str = str + key
    print(str)

def main():
    print('hello world')
    random.seed(10)

    input = 'P(d|c,~d,e,~f)'
    input = 'P(~f|~a,~b,~c,~d,~e,~g,i)'

    # Initialize
    sol = Solution()
    file = "prob345.txt"
    sol.readData(file)

    # read input
    qvar, feed_dict, dual = sol.readInput(input)
    # input: P(d|~a,~b,~d,~c,e,g)
    # qvar = 'd'
    # feed_dict = {'b':0,'a':0,'e': 0,'g': 0,'c':0,'d':0}
    print(qvar)
    print(feed_dict)

    # # test graph building
    # sol.BFS(sol.print_node)

    # # test topological order
    record = sol.BFS(sol.read_label)
    print(record)
    record = sol.BFS_rev(sol.read_label)
    print(record)

    # test mb_prob
    print('#test mb_prob')
    # qvar = 'd'
    # feed_dict = {'b':0,'a':0,'e': 0,'g': 0,'c':0,'d':0}
    print(qvar)
    print(feed_dict)
    sol.mb_prob(qvar, feed_dict, test=1)
    print('P('+qvar+'|mb)', sol.mb_prob(qvar, feed_dict, test=1))
    print('____________________________________')

    # test application
    # qvar = 'g'
    # feed_dict = {'c': 1, 'd': 0, 'e': 1, 'f':0}
    for i in range(1):
        # test likelihood weighting
        prob = sol.likelihood_weighting(qvar, feed_dict, 10000)
        print(prob)

        # test gibbs
        prob,_ = sol.Gibbs_sampling(qvar, feed_dict, 10000)
        print(prob)

        print('____________________________________')


    # # Plot convergence
    # ntot = 4 # ntot_critical = 10**4: err ~ 0.005
    # prob_LW = np.zeros(ntot)
    # prob_Gibbs = np.zeros(ntot)
    #
    # qvar = 'g'
    # feed_dict = {'k': 1, 'b': 0, 'c': 1}
    # for n in range(ntot + 1):
    #     Ntot = 10**n
    #     print('Ntot=', Ntot)
    #
    #     # test likelihood weighting
    #     prob = sol.likelihood_weighting(qvar, feed_dict, Ntot)
    #     print(prob)
    #     prob_LW[n] = prob[1]
    #
    #     # test gibbs likelihood
    #     prob,_ = sol.Gibbs_sampling(qvar, feed_dict, Ntot)
    #     print(prob)
    #     prob_Gibbs[n] = prob[1]
    #
    # plt.plot(np.arange(ntot + 1), prob_LW, prob_Gibbs)
    # plt.show()





if __name__ == '__main__':
    main()