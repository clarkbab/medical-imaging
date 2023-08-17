class Test:
    def __init__(self):
        self.__a = [1, 2, 3, 4, 5]
    
    def __getitem__(self, index):
        index, other = index
        return other * self.__a[index]

t = Test()
print(t[1, 2])
