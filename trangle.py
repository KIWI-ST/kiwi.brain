# 杨辉三角形代码
# 很赞的实现了
# 1.数组切片用法
# 2.生成器用法

def yanghui():
    L=[1, 0]
    while 1:
        yield L[:-1:]
        L = [1] + [L[i]+L[i+1] for i in range(len(L)-1)] + [0]
    return 'done'
triangles = yanghui()
n = 0
for t in triangles:
    print(t)
    n = n + 1
    if n == 10:
        break