@echo off
cd /d d:\Ai2050\TransformerLens-Project
echo [%date% %time%] Starting DeepSeek7B CCLXXV (venv python) >> d:\Ai2050\TransformerLens-Project\tests\glm5_temp\cclxxv_ds7b_log4.txt
.venv\Scripts\python.exe tests\glm5\phase_cclxxv_up_direction.py --model deepseek7b >> d:\Ai2050\TransformerLens-Project\tests\glm5_temp\cclxxv_ds7b_log4.txt 2>&1
echo [%date% %time%] DeepSeek7B done >> d:\Ai2050\TransformerLens-Project\tests\glm5_temp\cclxxv_ds7b_log4.txt
