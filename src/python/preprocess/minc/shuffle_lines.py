import random
lines = open('train_balance.txt').readlines()
random.shuffle(lines)
open('train_balance_shuffle.txt', 'w').writelines(lines)
