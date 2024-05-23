# exp1-2
import numpy as np
import time
A = np.arange(1000)
B = np.arange(1000, 2000)
sum = 0
start_time = time.time()

for a, b in zip(A, B):
    sum += a * b

end_time = time.time()
run_time = end_time - start_time
print(f"sum = {sum}")
print(f"运行时间：{run_time} 秒")