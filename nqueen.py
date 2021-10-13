from time import time
from itertools import permutations
import matplotlib.pyplot as plt
import pickle
from os import path, environ

timeCapture = {
    'Recursive': [],
    'Iterative': []
}

def timer(func):
    def wrap_func(*args, **kwargs):
        global timeCapture
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        delta_t = t2-t1
        print(f'Function {func.__name__!r} N = {args[0]} executed in {delta_t}s')
        timeCapture[func.__name__].append((args[0], delta_t))
        return result
    return wrap_func

@timer
def Iterative(N):

    def put_queen(x,y):
        nonlocal N
        if table[y][x] == 0:
            for i in range(N):
                table[y][i] = 1
                table[i][x] = 1
                table[y][x] = 2
                if y+i <= N-1 and x+i <= N-1:
                    table[y+i][x+i] = 1
                if y-i >= 0 and x+i <= N-1:
                    table[y-i][x+i] = 1
                if y+i <= N-1 and x-i >= 0:
                    table[y+i][x-i] = 1
                if y-i >= 0 and x-i >= 0:
                    table[y-i][x-i] = 1
            return True
        else:
            return False

    table = [[0]*N for _ in range(N)]    
    perms = permutations([i for i in range(N)])


    num_comb = 0

    for perm in perms:
        is_solution = True
        for i in range(N):
            is_solution = is_solution and put_queen(perm[i], i)
        if is_solution:
            # print_table()
            num_comb += 1
        table = [[0] * N for _ in range(N)]
    
    print(f'number of solutions : {num_comb}')

@timer
def Recursive(N):
    numSol = 0  

    b = N*[-1] 
    colFree = N*[1] 
    upFree = (2*N - 1)*[1] 		
    downFree = (2*N - 1)*[1]    		

    def putQueen(r, b, colFree, upFree, downFree):
        nonlocal N
        nonlocal numSol
        for c in range(N): 
            if colFree[c] and upFree[r+c] and downFree[r-c+N-1]:
                b[r] = c
                colFree[c] = upFree[r+c] = downFree[r-c+N-1] = 0
                if r == N-1:
                    numSol += 1
                else:
                    putQueen(r+1, b, colFree, upFree, downFree)
                colFree[c] = upFree[r+c] = downFree[r-c+N-1] = 1                                                            
    putQueen(0, b, colFree, upFree, downFree)
    print(f'number of solutions : {numSol}')

def nqueen():
    for n in range(4, 11):
        Recursive(n)
        Iterative(n)

    # saving data
    print('saving time capture to TimeCapture.pkl')
    with open('TimeCapture.pkl', 'wb') as file:
        pickle.dump(timeCapture, file)

def plot_graph():
    with open('TimeCapture.pkl', 'rb') as file:
        timeCapture = pickle.load(file)

    recursive = list(map(lambda x: x[1], timeCapture['Recursive']))
    iterative = list(map(lambda x: x[1], timeCapture['Iterative']))
    N = [i for i in range(4, 11)]
    plt.xlabel('Number of Queens')
    plt.ylabel('Time (seconds)')
    plt.plot(N, recursive, 'r--', label='Recursive')
    plt.plot(N, iterative, 'g--', label='Iterative')
    plt.legend(loc='upper left')
    plt.show()

def suppress_qt_warnings():
    environ["QT_DEVICE_PIXEL_RATIO"] = "0"
    environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    environ["QT_SCALE_FACTOR"] = "1"

if __name__ == '__main__':
    suppress_qt_warnings()
    if not path.isfile('./TimeCapture.pkl'):
        nqueen()
    plot_graph()