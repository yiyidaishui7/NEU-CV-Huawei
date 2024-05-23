# exp1-1
n = int(input("请输入要求斐波那契数列前几项："))
def fun(n):
    f1 = 0
    f2 = 1
    for i in range(n):
        f1,f2 = f2,f1+f2
        print(str(f1)+" ",end="")

print("------斐波那契前"+str(n)+"项------")
fun(n)