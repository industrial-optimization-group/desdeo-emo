import timeit

mysetup = "import numpy as np \n" \
          "arr = np.zeros((600,600))"
mycode1 = """
def place_prey1(arr):
    for i in range(1, 10000):

        free_space = np.transpose(np.nonzero(arr == 0))
        x, y = free_space[np.random.randint(free_space.shape[0])]
        arr[y][x] = i
"""

mycode2 = """
def place_prey2(arr):
    for i in range(1, 10000):

        x = np.random.randint(arr.shape[1])
        y = np.random.randint(arr.shape[0])

        while arr[y][x] != 0:
            x = np.random.randint(arr.shape[1])
            y = np.random.randint(arr.shape[0])

        # +1 to offset zero index individual
        arr[y][x] = i
"""
mycode3 = """
def place_prey3(arr):

"""
print(timeit.timeit(setup=mysetup, stmt=mycode1, number=10000000))
print(timeit.timeit(setup=mysetup, stmt=mycode2, number=10000000))