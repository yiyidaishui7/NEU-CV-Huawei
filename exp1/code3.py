#exp1-3
import numpy as np
import time

A = np.arange(1000)
B = np.arange(1000, 2000)

start_time = time.time()

sum = np.dot(A,B)

end_time = time.time()  # 获取当前时间
run_time = end_time - start_time  # 计算运行时间
print(f"sum = ：{sum}")
print(f"运行时间：{run_time} 秒")