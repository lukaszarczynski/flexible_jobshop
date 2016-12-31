# -*- coding: UTF-8 -*-

from copy import deepcopy
from parsing import load_problem, load_solution
from collections import deque


class Node(object):
    def __init__(self, weight, neighbors):
        self.weight = weight
        self.successors = neighbors  # lista trójek (numer cyklu, zadanie, operacja), 
                                     # numer cyklu 0, jeśli ten sam; 1 jeśli następny
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
        for cycle_idx in xrange(m):
            for k, v in self.subgraph.iteritems():
                self.graph[(cycle_idx + 1,) + k[1:]] = deepcopy(v)

        for machine in solution:
            for cycle_idx in xrange(m):
                self.graph[(cycle_idx,) + machine[-1]].successors.append((1,) + machine[0])

        for k, v in self.graph.iteritems():
            for successor in v.successors:
                self.graph[(k[0]+successor[0],) + successor[1:]].indegree += 1

        self.vertices = self.graph.keys()

    def create_subgraph(self):
        for machine_idx, machine in enumerate(self.solution):
            for pair_idx, (task, operation) in enumerate(machine):
                weights = self.problem[task - 1][operation - 1]  # indeksować od 0?
                weight = None
                for (_machine, _weight) in weights:  # słownik {maszyna: waga} zamiast zbioru par?
                    if machine_idx + 1 == _machine:
                        weight = _weight
                if weight is None:
                    raise Exception("Incorrect solution")
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
                ##
                u = (v[0]+u[0],) + u[1:]
                ##
                if indegs_cp[u] == 1:
                    Q.appendleft(u)
                indegs_cp[u] -= 1
        return result

if __name__ == "__main__":
    n, m, problem = load_problem("instances/Barnes_mt10c1.fjs")

    solution = load_solution("solutions/Barnes_mt10c1.txt")

    graph = Graph(n, m, problem, solution)