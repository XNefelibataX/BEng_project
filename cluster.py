import random

# 打开并读取 label.txt 文件
with open("label.txt", "r", encoding="utf-8") as file:
    lines = file.readlines()

# 生成随机索引
total_lines = len(lines)
test_indices = set(random.sample(range(total_lines), total_lines // 5))

# 分别获取测试数据和训练数据，保持顺序
test_data = [lines[i] for i in range(total_lines) if i in test_indices]
train_data = [lines[i] for i in range(total_lines) if i not in test_indices]

# 清空并写入 label_train.txt 和 label_test.txt
with open("label_train.txt", "w", encoding="utf-8") as train_file:
    train_file.writelines(train_data)

with open("label_test.txt", "w", encoding="utf-8") as test_file:
    test_file.writelines(test_data)

print("数据已随机分割完成，分别保存到 label_train.txt 和 label_test.txt 中。")
