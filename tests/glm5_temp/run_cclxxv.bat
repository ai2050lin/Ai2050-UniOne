@echo off
set PYTHONUNBUFFERED=1
cd /d d:\Ai2050\TransformerLens-Project
python -u tests/glm5/cclxxv_run_all.py
echo DONE >> tests\glm5_temp\cclxxv_run3.txt
