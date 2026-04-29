@echo off
set PYTHONUNBUFFERED=1
cd /d d:\Ai2050\TransformerLens-Project
echo [%time%] Starting torch test > tests\glm5_temp\torch_test.txt
python -u -c "import sys; sys.stdout.reconfigure(encoding='utf-8', errors='replace'); print('step1', flush=True); import torch; print('step2', flush=True); print(f'CUDA: {torch.cuda.is_available()}', flush=True); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}', flush=True); x = torch.randn(10,10).cuda() if torch.cuda.is_available() else None; print('step3: tensor OK', flush=True); del x; torch.cuda.empty_cache(); print('DONE', flush=True)" >> tests\glm5_temp\torch_test.txt 2>&1
echo [%time%] Finished >> tests\glm5_temp\torch_test.txt
