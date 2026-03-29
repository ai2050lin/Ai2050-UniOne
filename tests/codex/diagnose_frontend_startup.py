"""
前端客户端启动诊断脚本
检查前端环境配置和常见问题
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """运行命令并返回结果"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "命令执行超时"
    except Exception as e:
        return False, "", str(e)


def check_nodejs():
    """检查Node.js安装"""
    print("[检查1] Node.js环境:")
    success, stdout, stderr = run_command("node --version", "检查Node.js版本")
    if success:
        version = stdout.strip()
        print(f"  [OK] Node.js版本: {version}")
        # 检查版本是否过低
        major_version = int(version.split('.')[0].replace('v', ''))
        if major_version < 16:
            print(f"  [WARN] Node.js版本过低，建议升级到16+")
        return True
    else:
        print(f"  [FAIL] Node.js未安装或不可用")
        print(f"  错误: {stderr}")
        return False


def check_npm():
    """检查npm安装"""
    print("\n[检查2] npm环境:")
    success, stdout, stderr = run_command("npm --version", "检查npm版本")
    if success:
        version = stdout.strip()
        print(f"  [OK] npm版本: {version}")
        return True
    else:
        print(f"  [FAIL] npm未安装或不可用")
        print(f"  错误: {stderr}")
        return False


def check_node_modules():
    """检查node_modules是否存在"""
    print("\n[检查3] node_modules目录:")
    frontend_path = Path("d:/develop/TransformerLens-main/frontend")
    node_modules_path = frontend_path / "node_modules"
    
    if node_modules_path.exists():
        print(f"  [OK] node_modules目录存在")
        # 检查一些关键包
        key_packages = [
            "react",
            "react-dom",
            "vite",
            "@react-three/fiber",
            "@react-three/drei",
            "three"
        ]
        
        all_exist = True
        for pkg in key_packages:
            pkg_path = node_modules_path / pkg
            if pkg_path.exists():
                print(f"  [OK] {pkg} 已安装")
            else:
                print(f"  [FAIL] {pkg} 未安装")
                all_exist = False
        
        return all_exist
    else:
        print(f"  [FAIL] node_modules目录不存在")
        print(f"  提示: 需要运行 'npm install' 安装依赖")
        return False


def check_package_json():
    """检查package.json是否存在"""
    print("\n[检查4] package.json文件:")
    frontend_path = Path("d:/develop/TransformerLens-main/frontend")
    package_json_path = frontend_path / "package.json"
    
    if package_json_path.exists():
        print(f"  [OK] package.json存在")
        # 检查scripts
        import json
        try:
            with open(package_json_path, 'r', encoding='utf-8') as f:
                package_data = json.load(f)
            
            if 'scripts' in package_data:
                scripts = package_data['scripts']
                print(f"  [OK] 可用的脚本:")
                for name, script in scripts.items():
                    print(f"    - npm run {name}: {script}")
            
            return True
        except Exception as e:
            print(f"  [FAIL] 无法解析package.json: {str(e)}")
            return False
    else:
        print(f"  [FAIL] package.json不存在")
        return False


def check_vite_config():
    """检查vite.config.js是否存在"""
    print("\n[检查5] Vite配置:")
    frontend_path = Path("d:/develop/TransformerLens-main/frontend")
    vite_config_path = frontend_path / "vite.config.js"
    
    if vite_config_path.exists():
        print(f"  [OK] vite.config.js存在")
        content = vite_config_path.read_text(encoding='utf-8')
        
        # 检查端口配置
        if '5173' in content:
            print(f"  [OK] 端口配置: 5173")
        else:
            print(f"  [WARN] 端口配置可能不标准")
        
        # 检查host配置
        if '0.0.0.0' in content or 'localhost' in content:
            print(f"  [OK] host配置存在")
        
        return True
    else:
        print(f"  [WARN] vite.config.js不存在")
        return False


def check_port_conflict():
    """检查端口冲突"""
    print("\n[检查6] 端口占用检查:")
    success, stdout, stderr = run_command(
        "netstat -ano | findstr \":5173\"",
        "检查5173端口"
    )
    
    if success and stdout.strip():
        lines = stdout.strip().split('\n')
        if len(lines) > 0:
            print(f"  [WARN] 端口5173已被占用")
            for line in lines[:3]:  # 只显示前3个
                parts = line.split()
                if len(parts) >= 5:
                    local_addr = parts[1]
                    pid = parts[-1]
                    print(f"    地址: {local_addr}, PID: {pid}")
            print(f"  提示: 使用 'taskkill /F /PID <PID>' 终止进程")
            return False
    else:
        print(f"  [OK] 端口5173未被占用")
        return True


def check_index_html():
    """检查index.html是否存在"""
    print("\n[检查7] 入口文件:")
    frontend_path = Path("d:/develop/TransformerLens-main/frontend")
    index_html_path = frontend_path / "index.html"
    src_main_path = frontend_path / "src" / "main.jsx"
    
    if index_html_path.exists():
        print(f"  [OK] index.html存在")
    else:
        print(f"  [FAIL] index.html不存在")
        return False
    
    if src_main_path.exists():
        print(f"  [OK] src/main.jsx存在")
    else:
        print(f"  [FAIL] src/main.jsx不存在")
        return False
    
    return True


def check_eslint():
    """检查ESLint配置"""
    print("\n[检查8] ESLint配置:")
    frontend_path = Path("d:/develop/TransformerLens-main/frontend")
    
    eslint_config_files = [
        "eslint.config.js",
        ".eslintrc.js",
        ".eslintrc.json"
    ]
    
    found = False
    for config_file in eslint_config_files:
        config_path = frontend_path / config_file
        if config_path.exists():
            print(f"  [OK] {config_file}存在")
            found = True
            break
    
    if not found:
        print(f"  [WARN] 未找到ESLint配置文件")
    
    return True


def provide_solutions():
    """提供解决方案"""
    print("\n" + "="*60)
    print("解决方案和建议")
    print("="*60)
    
    print("\n1. 如果node_modules不存在:")
    print("   cd frontend")
    print("   npm install")
    
    print("\n2. 如果端口被占用:")
    print("   netstat -ano | findstr \":5173\"")
    print("   taskkill /F /PID <进程ID>")
    
    print("\n3. 启动前端服务:")
    print("   cd frontend")
    print("   npm run dev")
    
    print("\n4. 如果依赖有问题，可以尝试:")
    print("   cd frontend")
    print("   rm -rf node_modules package-lock.json")
    print("   npm install")
    
    print("\n5. 如果Node.js版本过低:")
    print("   从 https://nodejs.org/ 下载最新LTS版本")
    
    print("\n6. 访问应用:")
    print("   打开浏览器访问: http://localhost:5173")


def main():
    """主函数"""
    print("="*60)
    print("前端客户端启动诊断")
    print("="*60)
    
    checks = [
        check_nodejs,
        check_npm,
        check_node_modules,
        check_package_json,
        check_vite_config,
        check_port_conflict,
        check_index_html,
        check_eslint,
    ]
    
    passed = 0
    failed = 0
    
    for check_func in checks:
        try:
            result = check_func()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  [ERROR] 检查失败: {str(e)}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"诊断结果: {passed} 通过, {failed} 失败/警告")
    print("="*60)
    
    provide_solutions()
    
    if failed == 0:
        print("\n[成功] 所有检查通过！前端应该可以正常启动。")
        return 0
    else:
        print(f"\n[提示] 发现{failed}个问题，请按照上述解决方案处理。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
