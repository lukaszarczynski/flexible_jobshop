# -*- coding: UTF-8 -*-

from copy import deepcopy
from parsing import load_problem, load_solution
from collections import deque


class Node(object):
    def __init__(self, weight, neighbors):
        self.weight = weight
        self.successors = neighbors  # lista trójek (numer cyklu, zadanie, operacja)
        self.predecessors = []
        self.indegree = 0


class Graph(object):
    def __init__(self, n, m, problem, solution):
        self.n = n
        self.m = m
        self.problem = problem
        self.solution = solution
        self.subgraph = {}
        self.create_subgraph()
        self.graph = deepcopy(self.subgraph)
        self.topological_order = None
        for cycle_idx in xrange(m):
            for k, v in self.subgraph.iteritems():
                new_v = (cycle_idx + 1,) + k[1:]
                self.graph[new_v] = deepcopy(v)
                successors = []
                for successor in v.successors:
                    successors.append((cycle_idx + 1,) + successor[1:])
                self.graph[new_v].successors = successors

        for machine in solution:
            for cycle_idx in xrange(m):
                self.graph[(cycle_idx,) + machine[-1]].successors.append((cycle_idx + 1,) + machine[0])

        for k, v in self.graph.iteritems():
            for successor in v.successors:
                self.graph[successor].indegree += 1
                self.graph[successor].predecessors.append(k)

        self.vertices = self.graph.keys()

    def create_subgraph(self):
        for machine_idx, machine in enumerate(self.solution):
            for pair_idx, (task, operation) in enumerate(machine):
                weights = self.problem[task - 1][operation - 1]  # indeksować od 0?
                        
                if machine_idx + 1 not in weights:
                    raise Exception("Incorrect solution")
                weight = weights[machine_idx + 1]
                
                neighbors = []
                if len(machine) > pair_idx + 1:
                    neighbors.append((0,) + machine[pair_idx + 1])
                if len(self.problem[task - 1]) > operation:
                    neighbors.append((0, task, operation + 1))
                self.subgraph[(0, task, operation)] = Node(weight, neighbors)

    def topological_sort(self):
        indegs_cp = {v : self.graph[v].indegree for v in self.vertices}
        Q = deque([v for v,deg in indegs_cp.items() if deg == 0])
        result = []
        while Q:
            v = Q.pop()
            result.append(v)
            for u in self.graph[v].successors:
                if indegs_cp[u] == 1:
                    Q.appendleft(u)
                indegs_cp[u] -= 1
        
        self.topological_order = result
    
    def longest_path_length(self, v1, v2, justH1=False):
        assert self.topological_order is not None
        if v1 not in self.graph or v2 not in self.graph:
            return None
        temp_longest = {v1 : self.graph[v1].weight}

        if v1 == v2:
            return temp_longest[v1]

        for v in self.topological_order[self.topological_order.index(v1)+1:]:
            if (not justH1) or v[0] == 0:
                preds = [p for p in self.graph[v].predecessors if p in temp_longest]
                if preds:
                    best_pred = max(preds, key=lambda p: temp_longest[p] + self.graph[p].weight)
                    temp_longest[v] = temp_longest[best_pred] + self.graph[v].weight
                if v == v2:
                    if v2 in temp_longest:
                        return temp_longest[v2]
                    return None
    
    # ruch jest zły wtw. gdy któraś z tych ścieżek istnieje w starym grafie:
    # następnik technologiczny -> nowy poprzednik maszynowy (path1)
    # nowy następnik maszynowy -> poprzednik technologiczny (path2)
    def valid_move(self, (i,k,s)):
        next_tech = (0, i[1], i[2]+1)
        prev_tech = (0, i[1], i[2]-1)
        old_i_machine_idx = self.solution[i[0]].index(i[1:])
        
        if k != i[0]:
            new_next_m = (0,) + self.solution[k][s] if s < len(self.solution[k]) else None
        else:
            if s < old_i_machine_idx:
                new_next_m = (0,) + self.solution[k][s]
            elif s < len(self.solution[k])-1:
                new_next_m = (0,) + self.solution[k][s+1]
            else:
                new_next_m = None
                
        if k != i[0]:
            new_prev_m = (0,) + self.solution[k][s-1] if s > 0 else None
        else:
            if s > old_i_machine_idx:
                new_prev_m = (0,) + self.solution[k][s]
            elif s > 0:
                new_prev_m = (0,) + self.solution[k][s-1]
            else:
                new_prev_m = None
        
        path1_exists = self.longest_path_length(next_tech, new_prev_m, justH1=True) is not None
        path2_exists = self.longest_path_length(new_next_m, prev_tech, justH1=True) is not None
        
        return (not path1_exists) and (not path2_exists)

    def generate_neighborhood(self):
        moves = []
        
        for i in [(x[0],) + op for x in enumerate(self.solution) for op in x[1]]:
            for k in xrange(self.m):
                for s in xrange(len(self.solution[k])+1) if k != i[0] else xrange(len(self.solution[k])):
                    move = (i,k,s)
                    if self.valid_move(move):
                        moves.append(move)
        return moves
    
    # m_from, m_to - skąd zabieramy, gdzie wsadzamy
    # op - zabierana operacja, para (nr_maszyny, nr_operacji)
    # pos - pozycja do wsadzenia op na maszynę m_to
    # tutaj numery maszyn idą od 0
    def lower_bound(self, m_from, op, m_to, pos):
        def alpha(k,i):
            assert i >= 0
            if k != m_from or self.solution[k].index(op) > i:
                return self.solution[k][i]
            #print k, i
            return self.solution[k][i+1]

        # jeśli longest_path nie istnieje, to powinien zwrócić None
        # te drogi poniżej są zawarte w H1 (pierwszej składowej G), więc
        # można by je liczyć trochę szybciej (tylko w H1)

        def R(k,opr):
            return self.longest_path_length((0,) + alpha(k,0), (0,) + opr, justH1=True)

        def Q(k,opr):
            last_op_k_idx = len(self.solution[k]) - 1
            if k == m_from:
                last_op_k_idx -= 1
            return self.longest_path_length((0,) + opr, (0,) + alpha(k, last_op_k_idx), justH1=True)

        def LB(l):
            op_old_weight = self.graph[(0,) + op].weight
            op_new_weight = self.problem[op[0]-1][op[1]-1][m_to+1]

            R1 = lambda: R(l, alpha(m_to, pos-1))
            def R2():
                r = R(l, op)
                return r - op_old_weight if r is not None else None
            Q1 = lambda: Q(l, alpha(m_to, pos))
            def Q2():
                q = Q(l, op)
                return q - op_old_weight if q is not None else None 

            # 0 w maxach po to, żeby mieć jakąś liczbę, gdy drogi nie istnieją
            if pos == 0:
                return max(R2(), 0) + max(Q1(), Q2(), 0) + op_new_weight

            len_m_to = len(self.solution[m_to])
            if m_to == m_from:
                len_m_to -= 1
            if pos == len_m_to:
                return max(R1(), R2(), 0) + max(Q2(), 0) + op_new_weight

            maxR = max(R1(), R2(), 0)
            maxQ = max(Q1(), Q2(), 0)

            return maxR + maxQ + op_new_weight # a może tu ma być op_new_weight?
        
        return max([LB(l) for l in xrange(len(self.solution))])