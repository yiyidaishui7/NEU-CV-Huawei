
#exp1-4
import cv2
import numpy as np
image = cv2.imread("usagi.png")
# 左旋90
# 获取图像的尺寸
(height, width) = image.shape[:2]

# 创建旋转矩阵
rotation_mat = cv2.getRotationMatrix2D((width / 2, height / 2), -90, 1.0)

# 旋转图像
rotated = cv2.warpAffine(image, rotation_mat, (width, height))

# 保存旋转后的图像
cv2.imwrite('usagi_rotate.png', rotated)

# 垂直翻转
flipped_0 = cv2.flip(image, 0)
cv2.imwrite("usagi_ub.png", flipped_0)
# 水平翻转
flipped_1 = cv2.flip(image, 1)
cv2.imwrite("usagi_lr.png", flipped_1)
# 添加噪音
noise = np.random.normal(0, 50, image.shape)
noisy = image + noise
cv2.imwrite("usagi_noise.png", noisy)
# 增亮图像
value = 50
brightened = cv2.add(image, value)
cv2.imwrite("usagi_light.png", brightened)
import matplotlib.pyplot as plt

all_images = [cv2.imread('usagi.png'),
              cv2.imread('usagi_rotate.png'),
              cv2.imread('usagi_ub.png'),
              cv2.imread('usagi_lr.png'),
              cv2.imread('usagi_noise.png'),
              cv2.imread('usagi_light.png')]

file_names = ['usagi.png',
              'usagi_rotate.png',
              'usagi_ub.png',
              'usagi_lr.png',
              'usagi_noise.png',
              'usagi_light.png']


fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 将每张图像添加到子图中
for i, ax in enumerate(axes.flat):
    # 将 BGR 图像转换为 RGB
    image_rgb = cv2.cvtColor(all_images[i], cv2.COLOR_BGR2RGB)
    ax.imshow(image_rgb)
    # 在图片下方添加文件名
    ax.set_title(file_names[i], fontsize=12, pad=10)
    ax.axis('off')  # 隐藏坐标轴

# 调整子图间的间距
plt.subplots_adjust(wspace=0.2, hspace=0.3)

# 显示图像
plt.show()
