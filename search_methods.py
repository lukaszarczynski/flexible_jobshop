# -*- coding: UTF-8 -*-
from random import random

from graph_representation import Graph
import time

from parsing import load_solution, load_problem


def search_for_solution(n, m, problem, initial_solution, num_iter, cost_function):
    G = Graph(n, m, problem, initial_solution)
    cost_time = 0.

    for i in xrange(num_iter):
        neighborhood = G.generate_neighborhood()

        t0 = time.time()
        best_move = min(neighborhood, key=cost_function)
        cost_time += time.time() - t0

        G.make_a_move(best_move)

        t0 = time.time()
        G.topological_sort()
        cost_time += time.time() - t0

    return G.solution, cost_time, G.min_cycle_time()

def prepare_initial_solution(n, m, problem, optimal_solution, num_iter):
    graph = Graph(n, m, problem, optimal_solution)
    optimal_time = graph.min_cycle_time()
    initial_solution, _, initial_cycle_time = search_for_solution(n, m, problem, optimal_solution, num_iter, lambda x: random())
    print "Initial solution is ", float(initial_cycle_time) / optimal_time, " times worse than optimal/best known solution."
    return initial_solution, initial_cycle_time

if __name__ == "__main__":
    n, m, problem = load_problem("instances/Hurink_edata_car2.fjs")

    solution = load_solution("solutions/Hurink_edata_car2.txt")

    graph = Graph(n, m, problem, solution)

    optimal_time = graph.min_cycle_time()

    initial_solution, initial_cycle_time = prepare_initial_solution(n, m, problem, solution, 5)

    result, execution_time, cycle_time = search_for_solution(n, m, problem, initial_solution, 10, cost_function=graph.lower_bound)


    print "Solution found is ", float(cycle_time) / optimal_time, " times worse than optimal/best known solution."
    print "Initial solution was ", float(initial_cycle_time) / cycle_time, " times worse than found solution."
    print optimal_time, initial_cycle_time, cycle_time