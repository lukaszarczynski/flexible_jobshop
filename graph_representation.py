# -*- coding: UTF-8 -*-
import sys, traceback
from copy import deepcopy
from collections import deque

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

    def find_opposite_move(self, (operation, machine, position)):
        operation_position = None
        for index, row in enumerate(self.solution):
            if operation[1:] in row:
                operation_position = (index, row.index(operation[1:]))
        return (operation,) + operation_position

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
                    best_pred = max(preds, key=lambda p: temp_longest[p] + self.graph[p].weight)
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
        topological_order = None
        if move is not None:
            reverse_move = self.find_opposite_move(move)
            self.make_a_move(move)
            self.topological_sort()
            topological_order = self.topological_order

        critical_path_len = 0

        for machine in self.solution:
            vertex = machine[0]
            for cycle_idx in xrange(1, self.m):
                path_len = self.longest_path_length((0,) + vertex, (cycle_idx,) + vertex) / cycle_idx
                if path_len > critical_path_len:
                    critical_path_len = path_len

        if move is not None:
            self.make_a_move(reverse_move)
            self.topological_order = topological_order

        return critical_path_len

    # m_from, m_to - skąd zabieramy, gdzie wsadzamy
    # op - zabierana operacja, para (nr_maszyny, nr_operacji)
    # pos - pozycja do wsadzenia op na maszynę m_to
    # tutaj numery maszyn idą od 0
    def lower_bound(self, (i, m_to, pos)):
        m_from = i[0]
        op = i[1:]

        def alpha(k, i):
            assert i >= 0
            if k != m_from or self.solution[k].index(op) > i:
                return self.solution[k][i]
            # print k, i
            return self.solution[k][i + 1]

        # jeśli longest_path nie istnieje, to powinien zwrócić None
        # te drogi poniżej są zawarte w H1 (pierwszej składowej G), więc
        # można by je liczyć trochę szybciej (tylko w H1)

        def R(k, opr):
            return self.longest_path_length((0,) + alpha(k, 0), (0,) + opr, justH1=True)

        def Q(k, opr):
            last_op_k_idx = len(self.solution[k]) - 1
            if k == m_from:
                last_op_k_idx -= 1
            return self.longest_path_length((0,) + opr, (0,) + alpha(k, last_op_k_idx), justH1=True)

        def LB(l):
            op_old_weight = self.graph[(0,) + op].weight
            op_new_weight = self.problem[op[0] - 1][op[1] - 1][m_to + 1]

            R1 = lambda: R(l, alpha(m_to, pos - 1))

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

            return maxR + maxQ + op_new_weight  # a może tu ma być op_new_weight?

        return max([LB(l) for l in xrange(len(self.solution))])


if __name__ == "__main__":
    n, m, problem = load_problem("instances/Hurink_edata_car2.fjs")

    solution = load_solution("solutions/Hurink_edata_car2.txt")

    graph = Graph(n, m, problem, solution)

    neighbors = graph.generate_neighborhood()
    move = ((1, 8, 2), 1, 6)

    print graph.min_cycle_time()
    print graph.min_cycle_time(move)
    print graph.lower_bound(move)

    print len(neighbors)
    min_cycle_times = map(graph.min_cycle_time, neighbors[:26])
    lower_bounds = map(graph.lower_bound, neighbors[:26])
    lower_bound_holds = reduce(lambda x, y: x == y,
                               [min_cycle_times[i] >= lower_bounds[i] for i, _ in enumerate(min_cycle_times)])
    print lower_bound_holds

    move = neighbors[27]
    opposite_move = graph.find_opposite_move(move)
    try:
        graph.make_a_move(move)
        graph.make_a_move(opposite_move)
    except ValueError as inst:
        print "Making move failed: ", type(inst), inst
        print '-' * 60
        traceback.print_exc(file=sys.stdout)
        print '-' * 60
