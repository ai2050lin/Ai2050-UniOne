@echo off
cd /d d:\Ai2050\TransformerLens-Project
python tests/glm5/cclxxv_v3.py --model deepseek7b 2>&1 | tee tests/glm5_temp/cclxxv_ds7b_v3_log.txt
echo Done!
