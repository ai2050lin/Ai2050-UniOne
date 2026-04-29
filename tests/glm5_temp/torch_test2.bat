@echo off
set PYTHONUNBUFFERED=1
echo [%time%] Start > d:\Ai2050\TransformerLens-Project\tests\glm5_temp\torch2.txt
python -u -c "import time; t0=time.time(); print(f't0={t0}', flush=True); import torch; print(f'torch imported in {time.time()-t0:.1f}s', flush=True); print(f'CUDA: {torch.cuda.is_available()}', flush=True); print('DONE', flush=True)" >> d:\Ai2050\TransformerLens-Project\tests\glm5_temp\torch2.txt 2>&1
echo [%time%] End >> d:\Ai2050\TransformerLens-Project\tests\glm5_temp\torch2.txt
