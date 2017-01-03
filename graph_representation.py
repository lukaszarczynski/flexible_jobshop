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
                self.graph[(cycle_idx + 1,) + k[1:]] = deepcopy(v)
                successors = []
                for successor in v.successors:
                    successors.append((cycle_idx + 2,) + successor[1:])
                v.successors = successors

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
        print Q
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

        for v in self.topologica_order[self.topologica_order.index(v1)+1:]:
            if (not justH1) or v[0] == 0:
                preds = [p for p in self.graph[v].predecessors if v in temp_longest]
                if preds:
                    best_pred = max(preds, key=lambda p: temp_longest[p] + self.graph[p].weight)
                    temp_longest[v] = temp_longest[best_pred] + self.graph[best_pred].weight
                if v == v2:
                    if v2 in temp_longest:
                        return temp_longest[v2]
                    return None

if __name__ == "__main__":
    n, m, problem = load_problem("instances/Barnes_mt10c1.fjs")

    solution = load_solution("solutions/Barnes_mt10c1.txt")

    graph = Graph(n, m, problem, solution)