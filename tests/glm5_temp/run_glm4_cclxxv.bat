@echo off
cd /d d:\Ai2050\TransformerLens-Project
echo [%date% %time%] Starting GLM4 CCLXXV >> d:\Ai2050\TransformerLens-Project\tests\glm5_temp\cclxxv_glm4_log.txt
python tests\glm5\cclxxv_v3.py --model glm4 >> d:\Ai2050\TransformerLens-Project\tests\glm5_temp\cclxxv_glm4_log.txt 2>&1
echo [%date% %time%] GLM4 done >> d:\Ai2050\TransformerLens-Project\tests\glm5_temp\cclxxv_glm4_log.txt
