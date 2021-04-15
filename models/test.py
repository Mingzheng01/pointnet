import tensorflow as tf
import numpy as np

def updateList(list1):
    list1 += [10]
n = [5, 6]
print(id(n))                  # 140312184155336
updateList(n)
print(n)                      # [5, 6, 10]
print(id(n))                  # 140312184155336
