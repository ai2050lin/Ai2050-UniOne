@echo off
cd /d d:\Ai2050\TransformerLens-Project
echo [%date% %time%] Starting DeepSeek7B CCLXXV (fixed) >> d:\Ai2050\TransformerLens-Project\tests\glm5_temp\cclxxv_ds7b_log2.txt
python tests\glm5\cclxxv_v3.py --model deepseek7b >> d:\Ai2050\TransformerLens-Project\tests\glm5_temp\cclxxv_ds7b_log2.txt 2>&1
echo [%date% %time%] DeepSeek7B done >> d:\Ai2050\TransformerLens-Project\tests\glm5_temp\cclxxv_ds7b_log2.txt
