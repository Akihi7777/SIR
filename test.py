import random


class test():
    def __init__(self,i=7):
        self.i=i

    def o(self,j):
        j+=1
        print(j)

if __name__ == '__main__':
    t=test()
    for i in range(10):
        t.o(t.i)


