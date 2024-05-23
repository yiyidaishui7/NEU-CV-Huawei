# #选做1
import argparse
def print_isosceles_triangle(height=5):
    for i in range(1, height + 1):
        print(' ' * (height - i) + str(i) * (2 * i - 1))

parser = argparse.ArgumentParser(description='生成等腰三角形')
parser.add_argument('--height', type=int, default=5, help='三角形的高度 (必须是大于0的整数， 默认： 5)')
args = parser.parse_args()
print_isosceles_triangle(args.height)
