
class Farther(object):  # 这里的object可写可不写，不写就默认为object
    def __init__(self, name:str, age:int):
        self.name = name
        self.age = age
 
    def put_1(self)->None:
        print(self.name, self.age)
 
 
class Me(Farther):
    def __init__(self, name, age, university):
        super().__init__(name, age)   #不同之处
        self.university = university
 
    def put_1(self):
        super().put_1()
        print(self.university)
 
 
class Brother(Farther):
    def __init__(self, name, age, grad):
        super().__init__(name, age)


if __name__ == "__main__":
    sws_1 = Me('张三', 20, '大二')
    sws_1.put_1()

    sws_2 = Brother('张二', 22, '大四')
    sws_2.put_1()