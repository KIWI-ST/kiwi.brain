
#整数
a = 100
#字符串
b = 'i am ok'
#字符串转化为数值
c='200'
d= int(c)

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
count = len(tuple1)
#条件
if count >= 3:
    print(tuple1[2])
elif d>100:
    print(d)
#循环
for element in tuple3:
    print(element)
#range,快速生产一系列数组
range1 = range(0,100,2)
list1 = list(range1)
e=0
for element2 in list1:
    e+=element2
#dict 键值对
dic = {
    "a":a,
    "b":b
}    
print(dic['a'])
#set,不重复的集合,支持或与非操作
set1 = set([1,2,3,4])
set2 = set([2,3,56,7,8])
print(set1&set2)
print(set1|set2) 








import tensorflow as tf

hello = tf.constant("")

aa = tf.constant


print(tuple1)
print(tuple2)
print(classmates)