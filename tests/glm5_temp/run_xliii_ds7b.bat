@echo off
cd /d D:\develop\TransformerLens-main
echo Killing Ollama...
taskkill /F /IM ollama.exe 2>nul
timeout /t 2 /nobreak >nul
echo Running Phase XLIII DeepSeek7B test...
python -u tests\glm5\phase_xliii_ffn_weight_causal_chain.py --model deepseek7b
echo Test completed! Exit code: %ERRORLEVEL%
