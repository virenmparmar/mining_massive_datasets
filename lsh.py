# Authors: Jessica Su, Wanzi Zhou, Pratyaksh Sharma, Dylan Liu, Ansh Shukla

import numpy as np
import random
import time
import pdb
import unittest
from PIL import Image
import matplotlib.pyplot as plt

# Finds the L1 distance between two vectors
# u and v are 1-dimensional np.array objects
# TODO: Implement this
def l1(u, v):
    return np.sum(np.abs(np.subtract(u,v)))

# Loads the data into a np array, where each row corresponds to
# an image patch -- this step is sort of slow.
# Each row in the data is an image, and there are 400 columns.
def load_data(filename):
    return np.genfromtxt(filename, delimiter=',')

# Creates a hash function from a list of dimensions and thresholds.
def create_function(dimensions, thresholds):
    def f(v):
        boolarray = [v[dimensions[i]] >= thresholds[i] for i in range(len(dimensions))]
        return "".join(map(str, map(int, boolarray)))
    return f

# Creates the LSH functions (functions that compute L K-bit hash keys).
# Each function selects k dimensions (i.e. column indices of the image matrix)
# at random, and then chooses a random threshold for each dimension, between 0 and
# 255.  For any image, if its value on a given dimension is greater than or equal to
# the randomly chosen threshold, we set that bit to 1.  Each hash function returns
# a length-k bit string of the form "0101010001101001...", and the L hash functions 
# will produce L such bit strings for each image.
def create_functions(k, L, num_dimensions=400, min_threshold=0, max_threshold=255):
    functions = []
    for i in range(L):
        dimensions = np.random.randint(low = 0, 
                                   high = num_dimensions,
                                   size = k)
        thresholds = np.random.randint(low = min_threshold, 
                                   high = max_threshold + 1, 
                                   size = k)

        functions.append(create_function(dimensions, thresholds))
    return functions

# Hashes an individual vector (i.e. image).  This produces an array with L
# entries, where each entry is a string of k bits.
def hash_vector(functions, v):
    return np.array([f(v) for f in functions])

# Hashes the data in A, where each row is a datapoint, using the L
# functions in "functions."
def hash_data(functions, A):
    return np.array(list(map(lambda v: hash_vector(functions, v), A)))

# Retrieve all of the points that hash to one of the same buckets 
# as the query point.  Do not do any random sampling (unlike what the first
# part of this problem prescribes).
# Don't retrieve a point if it is the same point as the query point.
def get_candidates(hashed_A, hashed_point, query_index):
    return filter(lambda i: i != query_index and \
        any(hashed_point == hashed_A[i]), range(len(hashed_A)))

# Sets up the LSH.  You should try to call this function as few times as 
# possible, since it is expensive.
# A: The dataset.
# Return the LSH functions and hashed data structure.
def lsh_setup(A, k = 24, L = 10):
    functions = create_functions(k = k, L = L)
    hashed_A = hash_data(functions, A)
    return (functions, hashed_A)

# Run the entire LSH algorithm
def lsh_search(A, hashed_A, functions, query_index, num_neighbors = 10):
    hashed_point = hash_vector(functions, A[query_index, :])
    candidate_row_nums = get_candidates(hashed_A, hashed_point, query_index)
    
    distances = map(lambda r: (r, l1(A[r], A[query_index])), candidate_row_nums)
    best_neighbors = sorted(distances, key=lambda t: t[1])[:num_neighbors]

    return [t[0] for t in best_neighbors]

# Plots images at the specified rows and saves them each to files.
def plot(A, row_nums, base_filename):
    for row_num in row_nums:
        patch = np.reshape(A[row_num, :], [20, 20])
        im = Image.fromarray(patch)
        if im.mode != 'RGB':
            im = im.convert('RGB')
        im.save(base_filename + "-" + str(row_num) + ".png")

# Finds the nearest neighbors to a given vector, using linear search.
def linear_search(A, query_index, num_neighbors):
    distances = {}
    for i in range(A.shape[0]):
        if(i==query_index):
            continue
        distances.update({i: l1(A[i], A[query_index])})
    best_neighbors = sorted(distances.items(), key=lambda t:t[1])[:num_neighbors]
    return [t[0] for t in best_neighbors]


'''
This error function calculates the error as a function of average of all the search errors.
A single search error is calculated by the sum of manhatan distances of datapoints found by LSH 
divided by the sum of manhattan distances of datapoints found by linear search
'''
def error_function(A, lsh_neighbors, linear_neighbors):
    sumation = 0.0
    for key in lsh_neighbors:
        numerator = sum(list(map(lambda x, y: l1(A[x], A[y]), np.array(lsh_neighbors[key]), np.full(len(lsh_neighbors[key]), key))))
        denominator = sum(list(map(lambda x, y: l1(A[x], A[y]), np.array(linear_neighbors[key]), np.full(len(lsh_neighbors[key]), key))))
        sumation = sumation + (numerator/denominator)
    return sumation/len(lsh_neighbors)
#### TESTS #####

'''
The following code executes to print out the average time taken by 
both the searching algorithms: LSH and Linear Search.
'''
def finding_average_time(A, query_indexes):
    # First we setup the call the LSH setup function for the given values of L and k 
    lsh_functions, lsh_hashed_data = lsh_setup(data, L=10, k=24)
    similar_images = {}
    search_time = []
    for index in query_indexes:
        start_time = time.time()
        similar_imag_for_index = lsh_search(data, lsh_hashed_data,lsh_functions, index, num_neighbors=3)
        search_time.append(time.time()-start_time)
        similar_images.update({index:similar_imag_for_index})
    avg_lsh_time = sum(search_time)/len(search_time)
    print(f"The average search time for Locality Sensitive Hashing is {avg_lsh_time}")
    
    # Running the same queries for linear search
    similar_images_by_linear = {}
    linear_search_time = []
    for index in query_indexes:
        start_time = time.time()
        similar_imag_for_index = linear_search(data, index, num_neighbors =3)
        linear_search_time.append(time.time() - start_time)
        similar_images_by_linear.update({index:similar_imag_for_index})
    avg_linear_serch_time = sum(linear_search_time)/len(linear_search_time)
    print(f"The average search time for Linear Serch is  {avg_linear_serch_time}")

def calculate_error_values(data, query_indexes):
    L = np.array([10, 12, 14, 16, 18, 20])
    k = np.array([16, 18, 20, 22, 24])

    #Here, i am calculating errors as a function of L
    errors_l = []
    for l in L:
        similar_images_using_lsh = {}
        similar_images_using_linear = {}
        lsh_functions, lsh_hashed_data = lsh_setup(data, L=l, k=24)
        for index in query_indexes:
            similar_imag_for_index = lsh_search(data, lsh_hashed_data,lsh_functions, index, num_neighbors=3)
            similar_images_using_lsh.update({index:similar_imag_for_index})
            similar_imag_for_index = linear_search(data, index, num_neighbors=3)
            similar_images_using_linear.update({index:similar_imag_for_index})
        errors_l.append(error_function(data, similar_images_using_lsh, similar_images_using_linear))
    #plotting the error function
    plt.plot(L, errors_l)
    plt.ylabel('Error')
    plt.xlabel('L')
    plt.title('Error as a function of L')
    plt.show()
    # Calculating error as a function of k
    errors_k = []
    for kval in k:
        similar_images_using_lsh = {}
        similar_images_using_linear = {}
        lsh_functions, lsh_hashed_data = lsh_setup(data, L=10, k=kval)
        for index in query_indexes:
            similar_imag_for_index = lsh_search(data, lsh_hashed_data,lsh_functions, index, num_neighbors=3)
            similar_images_using_lsh.update({index:similar_imag_for_index})
            similar_imag_for_index = linear_search(data, index, num_neighbors=3)
            similar_images_using_linear.update({index:similar_imag_for_index})
        errors_k.append(error_function(data, similar_images_using_lsh, similar_images_using_linear))
    #plotting the error function
    plt.plot(k,errors_k)
    plt.ylabel('Error')
    plt.xlabel('k')
    plt.title('Error as a function of k')
    plt.show()

def find_10_siilar_imgs(data,query_index, L, k):
    lsh_functions, lsh_hashed_data = lsh_setup(data, L=10, k=24)
    similar_imag_for_index_lsh = lsh_search(data, lsh_hashed_data,lsh_functions, query_index, num_neighbors=10)
    similar_imag_for_index_linear = linear_search(data, query_index, num_neighbors=10)
    print("The 10 most similar by lsh are as follows {}".format(similar_imag_for_index_lsh))
    print("The 10 most similar by linear are as follows {}".format(similar_imag_for_index_linear))
    plot(data, similar_imag_for_index_lsh, 'lsh')
    plot(data, similar_imag_for_index_linear, 'linear')
    plot(data, [query_index], 'orig')


class TestLSH(unittest.TestCase):
    def test_l1(self):
        u = np.array([1, 2, 3, 4])
        v = np.array([2, 3, 2, 3])
        self.assertEqual(l1(u, v), 4)

    def test_hash_data(self):
        f1 = lambda v: sum(v)
        f2 = lambda v: sum([x * x for x in v])
        A = np.array([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(f1(A[0,:]), 6)
        self.assertEqual(f2(A[0,:]), 14)

        functions = [f1, f2]
        self.assertTrue(np.array_equal(hash_vector(functions, A[0, :]), np.array([6, 14])))
        self.assertTrue(np.array_equal(hash_data(functions, A), np.array([[6, 14], [15, 77]])))

    ### TODO: Write your tests here (they won't be graded, 
    ### but you may find them helpful)

'''
The main function which calls routines for all the questions asked in the assignment.
'''
if __name__ == '__main__':
#   unittest.main() ### TODO: Uncomment this to run tests
    query_indexes = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])

    data = load_data('patches.csv')
    
    finding_average_time(data, query_indexes)

    calculate_error_values(data, query_indexes)

    find_10_siilar_imgs(data,100, 24, 10)
