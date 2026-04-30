@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8

echo === CCLXXXIV: Top PC Decoding ===
echo [%date% %time%] Starting Qwen3...

cd /d d:\Ai2050\TransformerLens-Project
python tests/glm5/cclxxxiv_top_pc_decoding.py qwen3

echo [%date% %time%] Qwen3 done. Starting GLM4...
python tests/glm5/cclxxxiv_top_pc_decoding.py glm4

echo [%date% %time%] GLM4 done. Starting DeepSeek7B...
python tests/glm5/cclxxxiv_top_pc_decoding.py deepseek7b

echo [%date% %time%] All models done!
