
#整数
a = 100
#字符串
b = 'i am ok'
#数组 数组创建不定长
classmates = ['a','b',b,a]
#取数组最后一个元素
print(classmates[-1])
#在数组插入元素
classmates.insert(1,"adsfasgd")
#删除指定位置元素
classmates.pop(1)
#删除末尾元素
classmates.pop()
#替换指定元素
classmates[2]='a a a a a '
#数组插入数组
classmates.append(['a','n'])

#元组 tuple,数组创建定长
tuple1 = (a,b,'aaaaaa')
#元组 tuple,消除歧义，只创建一个元素的数组
tuple2 = (1,)
#元组 tuple,内部指向数组，所以 ['A','B']的值可变
tuple3 = ('a', 'b', ['A', 'B'])




print(tuple1)
print(tuple2)
print(classmates)