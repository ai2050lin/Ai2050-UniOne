"""
修复 FiberNetPanel.jsx: 移除所有 shadcn UI 组件依赖,
替换为原生 HTML + Tailwind CSS 实现。
"""

filepath = r"d:\develop\TransformerLens-main\frontend\src\components\FiberNetPanel.jsx"

with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

# 1. 删除 shadcn UI 导入
content = content.replace('import { Button } from "@/components/ui/button";\r\n', '')
content = content.replace('import { Button } from "@/components/ui/button";\n', '')
content = content.replace('import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";\r\n', '')
content = content.replace('import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";\n', '')
content = content.replace('import { Input } from "@/components/ui/input";\r\n', '')
content = content.replace('import { Input } from "@/components/ui/input";\n', '')
content = content.replace('import { Badge } from "@/components/ui/badge";\r\n', '')
content = content.replace('import { Badge } from "@/components/ui/badge";\n', '')

# 2. 在 import 区域之后，组件定义之前，添加 shimmed 组件
shim_code = '''
// === UI Component Shims (替代 shadcn UI) ===
const Button = ({ children, onClick, disabled, className = '', variant = 'default', size = 'default', ...props }) => {
  const baseStyle = 'inline-flex items-center justify-center rounded-md font-medium transition-colors focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer';
  const sizeStyle = size === 'sm' ? 'h-8 px-3 text-xs' : 'h-10 px-4 text-sm';
  const variantStyle = variant === 'ghost' ? 'bg-transparent hover:bg-slate-800 text-slate-400' : variant === 'secondary' ? 'bg-slate-800 text-slate-200' : '';
  return <button onClick={onClick} disabled={disabled} className={`${baseStyle} ${sizeStyle} ${variantStyle} ${className}`} {...props}>{children}</button>;
};
const Card = ({ children, className = '', ...props }) => <div className={`rounded-xl border bg-slate-950 text-slate-50 shadow ${className}`} {...props}>{children}</div>;
const CardHeader = ({ children, className = '', ...props }) => <div className={`flex flex-col space-y-1.5 p-6 ${className}`} {...props}>{children}</div>;
const CardTitle = ({ children, className = '', ...props }) => <div className={`font-semibold leading-none tracking-tight ${className}`} {...props}>{children}</div>;
const CardContent = ({ children, className = '', ...props }) => <div className={`p-6 pt-0 ${className}`} {...props}>{children}</div>;
const Input = ({ className = '', ...props }) => <input className={`flex h-10 w-full rounded-md border border-slate-800 bg-slate-950 px-3 py-2 text-sm text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 ${className}`} {...props} />;
const Badge = ({ children, className = '', ...props }) => <span className={`inline-flex items-center rounded-full border border-slate-700 px-2.5 py-0.5 text-xs font-semibold ${className}`} {...props}>{children}</span>;

'''

# 找到 API_BASE 定义行，在它之前插入 shim
api_base_line = "const API_BASE ="
pos = content.find(api_base_line)
if pos > 0:
    content = content[:pos] + shim_code + content[pos:]
    print("UI shims injected before API_BASE")
else:
    print("WARNING: Could not find API_BASE line")

with open(filepath, "w", encoding="utf-8") as f:
    f.write(content)

# 验证
with open(filepath, "r", encoding="utf-8") as f:
    verify = f.read()

has_no_shadcn = '@/components/ui' not in verify
has_shims = 'UI Component Shims' in verify
print(f"shadcn imports removed: {has_no_shadcn}")
print(f"UI shims present: {has_shims}")
print(f"Total lines: {len(verify.splitlines())}")
print("Done!")
