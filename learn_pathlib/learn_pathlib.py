# %%
from pathlib import Path

# %%
# 从字符串创建Path对象
current_dir = Path(".")  # 当前目录
home_dir = Path.home()  # 用户目录
abo_dir = Path("/home")  # 绝对路径
rel_dir = Path("~")
doc_dir = home_dir / "Documents"  # 使用/操作符拼接路径
# %%
# 文件与目录操作
p = Path("learn_pathlib.py")
print(f"{p.exists()=}")  # 判断文件是否存在
print(f"{p.is_file()=}")  # 判断是否为文件
print(f"{p.is_dir()=}")  # 判断是否为目录

# %%
# 创建与删除目录
new_dir = Path("new_dir")
new_dir.mkdir()  # 创建目录
print(f"{new_dir.exists()=}")
new_dir.rmdir()  # 删除目录
print(f"{new_dir.exists()=}")
# %%
# 遍历目录
for p in home_dir.iterdir():
    # 打印非隐藏文件
    if not p.name.startswith("."):
        print(p)

# %%
# 递归遍历目录
doc_dir = Path.home() / "Documents"
for p in doc_dir.rglob("*.pdf"):  # 递归遍历所有pdf文件
    # 打印非隐藏文件
    if not p.name.startswith("."):
        print(p)
# %%
# 读取和写入文件
file_path = Path(__file__)
content = file_path.read_text()  # 读取文件内容
print(f"content = \n{content}")

new_text = Path("new_text.txt")
new_text.write_text(content)  # 写入文件内容
# %%
# 获取文件信息
file_path = Path(__file__)
print(f"{file_path.name=}")  # 文件名
print(f"{file_path.suffix=}")  # 文件后缀
print(f"{file_path.stem=}")  # 文件名（不含后缀）
print(f"{file_path.parent=}")  # 父目录
print(f"{file_path.absolute()=}")  # 绝对路径

# %%
# 重命名
new_text = Path("new_text.txt")
new_text.rename("new_text.py")

# %%
# 计算相对路径
py_file = Path(__file__)
parent = py_file.parent
py_file.relative_to(parent)

# %%
