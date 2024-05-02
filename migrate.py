import os
import re

def is_local_module(module_name, project_root):
    # 检查模块的文件是否存在于项目的根目录下
    module_path = os.path.join(project_root, module_name.replace('.', os.sep))
    return os.path.exists(module_path + '.py') or os.path.exists(os.path.join(module_path, '__init__.py'))

def convert_import_line(line, project_root):
    # 匹配绝对导入语句
    match = re.match(r'(from|import) (\S+)', line)
    if match:
        import_type, module_name = match.groups()
        # Q: import_type 是什么
        # A: import_type 是 'from' 或 'import'
        if is_local_module(module_name, project_root):
            # 如果是本地模块，将绝对导入转换为相对导入
            if import_type == 'import':
                return 'from . import ' + module_name
            else:
                return line.replace(module_name, 'ddspsvc.' + module_name)
    return line

def convert_imports_in_file(file_path, project_root):
    with open(file_path, 'r', encoding="utf-8") as file:
        lines = file.readlines()
    # 转换导入语句
    lines = [convert_import_line(line, project_root) for line in lines]
    with open(file_path, 'w', encoding="utf-8") as file:
        file.writelines(lines)
        
def convert_project_imports(project_root):
    for root, _, files in os.walk(project_root):
        for file in files:
            if file.endswith('.py'):
                convert_imports_in_file(os.path.join(root, file), project_root)
                
convert_project_imports('ddspsvc')