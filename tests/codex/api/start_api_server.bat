@echo off
chcp 65001 >nul
echo ========================================
echo 启动AGI研究数据API服务
echo ========================================
echo.

cd /d "%~dp0"
python server.py

pause
