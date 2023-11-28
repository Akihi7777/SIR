# 创建一个示例列表
my_list = [1, 2, 3, 4, 5, 5, 2, 5, 5]

# 判断最后五个元素是否相同
if my_list[-5:] == [my_list[-1]] * 5:
    print("The last five elements are the same.")
else:
    print("The last five elements are not the same.")