# -*- coding: UTF-8 -*-
# n - liczba zadań, m - liczba maszyn

def load_problem(filename):
    num_lines = 0
    problem = []
    
    with open(filename, 'r') as f:
        for line in f:
            if num_lines == 0:
                n, m = map(int, line.split()[:-1])                    
            else:
                numbers = map(int, line.split())
                num_op = numbers[0]
                numbers = numbers[1:]
                problem.append([])
                idx = 0
                
                while idx < len(numbers):
                    k = numbers[idx]
                    ops = {tuple(numbers[idx+2*i+1:idx+2*i+3]) for i in xrange(k)}
                    problem[-1].append(ops)
                    idx += 2*k + 1
                
                assert len(problem[-1]) == num_op
                
            num_lines += 1
            
    assert num_lines == n+1
    return n, m, problem


# m jest tylko do sprawdzenia, może n też powinienem sprawdzać, ale pewnie oba są niepotrzebne

def load_solution(filename, m=None):
    solution = []
    with open(filename, 'r') as f:
        temp = f.readline().split(';')[:-1]
        temp = map(lambda x: map(int, x.split(',')), temp)
        for t in temp:
            l = len(t)
            assert not l % 2
            ops = [tuple(t[2*i:2*i+2]) for i in xrange(l/2)]
            solution.append(ops)
    
    if m is not None:
        assert len(solution) == m
    return solution
