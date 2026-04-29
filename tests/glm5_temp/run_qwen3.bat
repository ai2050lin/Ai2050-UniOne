@echo off
cd /d d:\Ai2050\TransformerLens-Project
echo [%time%] Starting qwen3 CCLXXV > tests\glm5_temp\cclxxv_bat_log.txt
python -u tests\glm5\cclxxv_v3.py --model qwen3 >> tests\glm5_temp\cclxxv_bat_log.txt 2>&1
echo [%time%] Finished >> tests\glm5_temp\cclxxv_bat_log.txt
