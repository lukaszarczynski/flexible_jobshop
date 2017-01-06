# -*- coding: UTF-8 -*-

from graph_representation import Graph
import time


def search_with_lower_bound(n, m, problem, initial_solution, num_iter):
    G = Graph(n, m, problem, initial_solution)
    G.topological_sort()
    cost_time = 0.
    
    for i in xrange(num_iter):
        N = G.generate_neighborhood()
        
        t0 = time.time()
        best_move = min(N, key=G.lower_bound)
        cost_time += time.time()-t0
        
        G.make_a_move(best_move)
        
        t0 = time.time()
        G.topological_sort()
        cost_time += time.time()-t0
        
    return G.solution, cost_time