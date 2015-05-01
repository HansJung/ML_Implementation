'''
From the list, extracting two possible pair
'''

from itertools import combinations

l = [1,2,3]
A = []
for subset in combinations(l,2):
    print subset
