# -*- coding: UTF-8 -*-
from copy import deepcopy
from collections import deque
from random import random, choice

import time

from parsing import load_problem, load_solution


class Node(object):
    def __init__(self, weight):
        self.weight = weight
        self.successors = []  # lista trójek (numer cyklu, zadanie, operacja)
        self.predecessors = []
        self.next_tech = None
        self.prev_tech = None
        self.next_mach = None
        self.prev_mach = None


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
                new_v_name = (cycle_idx + 1,) + k[1:]
                self.graph[new_v_name] = deepcopy(v)
                new_v = self.graph[new_v_name]
                if new_v.next_tech is not None:
                    new_v.next_tech = (cycle_idx + 1,) + new_v.next_tech[1:]
                if new_v.next_mach is not None:
                    new_v.next_mach = (cycle_idx + 1,) + new_v.next_mach[1:]

        for machine in self.solution:
            for cycle_idx in xrange(m):
                self.graph[(cycle_idx,) + machine[-1]].next_mach = ((cycle_idx + 1,) + machine[0])

        for k, v in self.graph.iteritems():
            if v.next_tech is not None:
                self.graph[v.next_tech].prev_tech = k
            if v.next_mach is not None:
                self.graph[v.next_mach].prev_mach = k

        self.vertices = self.graph.keys()
        self.recalculate_succs_and_preds(self.vertices)
        self.topological_sort()

    def recalculate_succs_and_preds(self, vertices):
        for v_name in vertices:
            v = self.graph[v_name]
            v.successors = {u for u in [v.next_tech, v.next_mach] if u is not None}
            v.predecessors = {u for u in [v.prev_tech, v.prev_mach] if u is not None}

    def create_subgraph(self):
        for machine_idx, machine in enumerate(self.solution):
            for pair_idx, (task, operation) in enumerate(machine):
                weights = self.problem[task - 1][operation - 1]  # indeksować od 0?

                if machine_idx + 1 not in weights:
                    raise Exception("Incorrect solution")
                weight = weights[machine_idx + 1]

                self.subgraph[(0, task, operation)] = Node(weight)
                v = self.subgraph[(0, task, operation)]

                if len(machine) > pair_idx + 1:
                    v.next_mach = ((0,) + machine[pair_idx + 1])
                if len(self.problem[task - 1]) > operation:
                    v.next_tech = (0, task, operation + 1)

    # format ruchu taki jak w lower_bound
    # nie sprawdza czy ruch jest poprawny! odpalać tylko na ruchach
    # wygenerowanych przez generate_neighborhood
    def make_a_move(self, (i, m_to, pos)):
        m_from = i[0]
        op = i[1:]

        for cycle_idx in xrange(self.m + 1):
            v_name = (cycle_idx,) + i[1:]
            v = self.graph[v_name]
            old_v_next_mach = v.next_mach
            old_v_prev_mach = v.prev_mach

            if v.prev_mach is not None:
                self.graph[v.prev_mach].next_mach = v.next_mach
            if v.next_mach is not None:
                self.graph[v.next_mach].prev_mach = v.prev_mach

            v.weight = self.problem[op[0] - 1][op[1] - 1][m_to + 1]

            _, new_v_next_mach, _, new_v_prev_mach = self.new_succs_preds((i, m_to, pos), cycle=cycle_idx,
                                                                          only_one_cycle=False)

            if new_v_next_mach is not None:
                self.graph[new_v_next_mach].prev_mach = v_name
                v.next_mach = new_v_next_mach
            else:
                v.next_mach = None

            if new_v_prev_mach is not None:
                self.graph[new_v_prev_mach].next_mach = v_name
                v.prev_mach = new_v_prev_mach
            else:
                v.prev_mach = None

            to_recalc = [u for u in [v_name, new_v_next_mach, new_v_prev_mach,
                                     old_v_next_mach, old_v_prev_mach] if u is not None]
            self.recalculate_succs_and_preds(to_recalc)

        self.solution[m_from].remove(op)
        self.solution[m_to].insert(pos, op)

    def find_opposite_move(self, (m_operation, machine, position)):
        new_m_operation = (machine,) + m_operation[1:]
        return (new_m_operation, m_operation[0], self.solution[m_operation[0]].index(m_operation[1:]))

    def topological_sort(self):
        indegs_cp = {v: len(self.graph[v].predecessors) for v in self.vertices}
        Q = deque([v for v, deg in indegs_cp.items() if deg == 0])
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
        temp_longest = {v1: self.graph[v1].weight}

        if v1 == v2:
            return temp_longest[v1]

        for v in self.topological_order[self.topological_order.index(v1) + 1:]:
            if (not justH1) or v[0] == 0:
                preds = [p for p in self.graph[v].predecessors if p in temp_longest]
                if preds:
                    best_pred = max(preds, key=lambda p: temp_longest[p])
                    temp_longest[v] = temp_longest[best_pred] + self.graph[v].weight
                if v == v2:
                    if v2 in temp_longest:
                        return temp_longest[v2]
                    return None

    def new_succs_preds(self, (i, k, s), cycle=0, only_one_cycle=True):
        next_tech = (cycle, i[1], i[2] + 1) if i[2] < len(self.problem[i[1] - 1]) else None
        prev_tech = (cycle, i[1], i[2] - 1) if i[2] > 1 else None
        old_i_machine_idx = self.solution[i[0]].index(i[1:])

        if k != i[0]:
            if s < len(self.solution[k]):
                new_next_m = (cycle,) + self.solution[k][s]
            else:
                if only_one_cycle or cycle == self.m:
                    new_next_m = None
                else:
                    new_next_m = (cycle + 1,) + self.solution[k][0]
        else:
            if s < old_i_machine_idx:
                new_next_m = (cycle,) + self.solution[k][s]
            elif s < len(self.solution[k]) - 1:
                new_next_m = (cycle,) + self.solution[k][s + 1]
            else:
                if only_one_cycle or cycle == self.m:
                    new_next_m = None
                elif self.solution[k][0] == i[1:]:
                    new_next_m = (cycle + 1,) + self.solution[k][1]
                else:
                    new_next_m = (cycle + 1,) + self.solution[k][0]

        if k != i[0]:
            if s > 0:
                new_prev_m = (cycle,) + self.solution[k][s - 1]
            else:
                if only_one_cycle or cycle == 0:
                    new_prev_m = None
                else:
                    new_prev_m = (cycle - 1,) + self.solution[k][-1]
        else:
            if s > old_i_machine_idx:
                new_prev_m = (cycle,) + self.solution[k][s]
            elif s > 0:
                new_prev_m = (cycle,) + self.solution[k][s - 1]
            else:
                if only_one_cycle or cycle == 0:
                    new_prev_m = None
                elif self.solution[k][-1] == i[1:]:
                    new_prev_m = (cycle - 1,) + self.solution[k][-2]
                else:
                    new_prev_m = (cycle - 1,) + self.solution[k][-1]

        return next_tech, new_next_m, prev_tech, new_prev_m

    def generate_neighborhood(self):
        # ruch jest zły wtw. gdy któraś z tych ścieżek istnieje w starym grafie:
        # następnik technologiczny -> nowy poprzednik maszynowy (path1)
        # nowy następnik maszynowy -> poprzednik technologiczny (path2)
        def valid_move(move):
            next_tech, new_next_m, prev_tech, new_prev_m = self.new_succs_preds(move)

            path1_exists = self.longest_path_length(next_tech, new_prev_m, justH1=True) is not None
            path2_exists = self.longest_path_length(new_next_m, prev_tech, justH1=True) is not None

            return (not path1_exists) and (not path2_exists)

        moves = []

        for i in [(x[0],) + op for x in enumerate(self.solution) for op in x[1]]:
            for k in [x - 1 for x in self.problem[i[1] - 1][i[2] - 1].keys()]:
                for s in xrange(len(self.solution[k]) + 1) if k != i[0] else xrange(len(self.solution[k])):
                    move = (i, k, s)
                    if valid_move(move):
                        moves.append(move)
        return moves

    def min_cycle_time(self, move=None):
        if move is not None:
            old_topological_order = self.topological_order
            reverse_move = self.find_opposite_move(move)
            self.make_a_move(move)
            self.topological_sort()

        critical_path_len = 0.

        for machine in self.solution:
            vertex = machine[0]
        #for task in xrange(self.n):
        #    vertex = (task + 1, 1)
            for cycle_idx in xrange(1, self.m + 1):
                longest_path = self.longest_path_length((0,) + vertex, (cycle_idx,) + vertex)
                if longest_path is None:
                    path_len = float("-inf")
                else:
                    longest_path -= self.graph[(cycle_idx,) + vertex].weight
                    path_len = longest_path / float(cycle_idx)
                if path_len > critical_path_len:
                    critical_path_len = path_len

        if move is not None:
            self.make_a_move(reverse_move)
            self.topological_order = old_topological_order

        return critical_path_len
    
    def lower_bound_experimental(self, move):
        op = move[0][1:]
        v_name = (0,) + op
        
        op_idx = self.topological_order.index(v_name)
        del self.topological_order[op_idx]
        
        rev_move = self.find_opposite_move(move)
        self.make_a_move(move)
        
        v = self.graph[v_name]
        
        def LB(l):
            if self.solution[l][0] == op:
                R1 = None
                R2 = None
            else:
                R1 = self.longest_path_length((0,) + self.solution[l][0], v.prev_mach, justH1=True)
                R2 = self.longest_path_length((0,) + self.solution[l][0], v.prev_tech, justH1=True)

            if self.solution[l][-1] == op:
                Q1 = None
                Q2 = None
            else:
                Q1 = self.longest_path_length(v.next_mach, (0,) + self.solution[l][-1], justH1=True)
                Q2 = self.longest_path_length(v.next_tech, (0,) + self.solution[l][-1], justH1=True)
            
            maxR = max(R1, R2, 0)
            maxQ = max(Q1, Q2, 0)
            
            longest_with_op = maxR + maxQ + v.weight
            
            if self.solution[l][0] == op or self.solution[l][-1] == op:
                longest_without_op = None
            else:
                longest_without_op = self.longest_path_length((0,) + self.solution[l][0], 
                                                              (0,) + self.solution[l][-1], justH1=True)

            return max(longest_without_op, longest_with_op)

        result = max([LB(l) for l in xrange(self.m)])
        
        self.make_a_move(rev_move)
        self.topological_order.insert(op_idx, v_name)
        
        return result
    
    # tutaj numery maszyn idą od 0
    def lower_bound(self, move):
        op = move[0][1:]
        v_name = (0,) + op
        
        op_idx = self.topological_order.index(v_name)
        del self.topological_order[op_idx]
        
        rev_move = self.find_opposite_move(move)
        self.make_a_move(move)
        
        v = self.graph[v_name]
        
        def LB(l):
            if self.solution[l][0] == op:
                R1 = None
                R2 = None
            else:
                R1 = self.longest_path_length((0,) + self.solution[l][0], v.prev_mach, justH1=True)
                R2 = self.longest_path_length((0,) + self.solution[l][0], v.prev_tech, justH1=True)

            if self.solution[l][-1] == op:
                Q1 = None
                Q2 = None
            else:
                Q1 = self.longest_path_length(v.next_mach, (0,) + self.solution[l][-1], justH1=True)
                Q2 = self.longest_path_length(v.next_tech, (0,) + self.solution[l][-1], justH1=True)
            
            maxR = max(R1, R2, 0)
            maxQ = max(Q1, Q2, 0)

            return maxR + maxQ + v.weight

        result = max([LB(l) for l in xrange(self.m)])
        
        self.make_a_move(rev_move)
        self.topological_order.insert(op_idx, v_name)
        
        return result
    
    def search_for_solution(self, num_iter, cost_function):
        cost_time = 0.
    
        print "searching..."
        #past_solutions = [deepcopy(self.solution)]
        #past_moves = []
        for i in xrange(num_iter):
            print i
            neighborhood = self.generate_neighborhood()
            #best_move = None
            #best_cost = float("inf")
            #best_sol = None

            t0 = time.time()
            best_move = min(neighborhood, key=cost_function)
            #for move in neighborhood:
            #    if not move in past_moves:
            #       new_sol = deepcopy(self.solution)
            #        new_sol[move[0][0]].remove(move[0][1:])
            #        new_sol[move[1]].insert(move[2], move[0][1:])
            #        if new_sol not in past_solutions:
            #            cost = cost_function(move)
            #            if cost < best_cost:
            #                best_cost = cost
            #                best_move = move
            #                best_sol = new_sol
            cost_time += time.time() - t0
            
            #past_solutions.append(best_sol)
            #past_moves.append(best_move)
            #past_moves.append(self.find_opposite_move(best_move))
            
            print best_move
            rev_move = self.find_opposite_move(best_move)
            print rev_move

            self.make_a_move(best_move)
            self.topological_sort()
            if rev_move == best_move:
                break
            
        print "found"

        return cost_time#, past_solutions

    def add_noise_to_solution(self, num_iter):
        for i in xrange(num_iter):
            random_move = choice(self.generate_neighborhood())
            self.make_a_move(random_move)
            self.topological_sort()
        
    
def get_random_starting_solution(n, m, problem):
    task_lens = dict(enumerate(map(len, problem)))
    solution = []

    for i in xrange(m):
        solution.append([])

    for op_num in xrange(max(task_lens.values())):
        tasks = [t for t in task_lens.keys() if task_lens[t] > op_num]
        for t in tasks:
            rand_machine = choice(problem[t][op_num].keys())
            solution[rand_machine-1].append((t+1, op_num+1))
            
    return solution


if __name__ == "__main__":
    n, m, problem = load_problem("instances/Barnes_setb4c9.fjs")

    solution = load_solution("solutions/Barnes_setb4c9.txt")

    graph = Graph(n, m, problem, solution)

    optimal_time = graph.min_cycle_time()
    print "optimal_time =", optimal_time

    neighborhood = graph.generate_neighborhood()
    move = neighborhood[0]
    opposite_move = graph.find_opposite_move(move)
    graph.make_a_move(move)

    lower_bound = graph.lower_bound(opposite_move)
    print "lower_bound =", lower_bound

    #graph.get_random_starting_solution()
    graph.add_noise_to_solution(num_iter=50)
    initial_cycle_time = graph.min_cycle_time()
    print "initial_cycle_time =", initial_cycle_time
    print "Initial solution is ", float(initial_cycle_time) / optimal_time, " times worse than optimal/best known solution."

    initial_cycle_time = graph.min_cycle_time()
    
    graph1 = graph
    graph2 = deepcopy(graph)

    execution_time_lb = graph1.search_for_solution(10, cost_function=graph1.lower_bound)

    cycle_time_lb = graph1.min_cycle_time()
    print "execution_time_lb =", execution_time_lb
    print "cycle_time_lb =", cycle_time_lb

    print "Solution found with lower bound is ", float(cycle_time_lb) / optimal_time, " times worse than optimal/best known solution."
    print "Initial solution was ", float(initial_cycle_time) / cycle_time_lb, " times worse than solution found with lower bound."

    
    execution_time_prec = graph2.search_for_solution(10, cost_function=graph2.min_cycle_time)

    cycle_time_prec = graph2.min_cycle_time()
    print "execution_time_prec =", execution_time_prec
    print "cycle_time_prec =", cycle_time_prec

    print "Solution found is ", float(cycle_time_prec) / optimal_time, " times worse than optimal/best known solution."
    print "Initial solution was ", float(initial_cycle_time) / cycle_time_prec, " times worse than found solution."