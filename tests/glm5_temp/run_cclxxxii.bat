@echo off
chcp 65001 >nul 2>&1
echo === CCLXXXII Large Category Test ===
echo Starting at %date% %time%

echo.
echo === Running Qwen3 ===
cd /d d:\Ai2050\TransformerLens-Project
python tests/glm5/cclxxxii_large_category.py qwen3
if %ERRORLEVEL% NEQ 0 (
    echo Qwen3 FAILED with error %ERRORLEVEL%
) else (
    echo Qwen3 DONE
)

echo.
echo === Running GLM4 ===
cd /d d:\Ai2050\TransformerLens-Project
python tests/glm5/cclxxxii_large_category.py glm4
if %ERRORLEVEL% NEQ 0 (
    echo GLM4 FAILED with error %ERRORLEVEL%
) else (
    echo GLM4 DONE
)

echo.
echo === Running DeepSeek7B ===
cd /d d:\Ai2050\TransformerLens-Project
python tests/glm5/cclxxxii_large_category.py deepseek7b
if %ERRORLEVEL% NEQ 0 (
    echo DeepSeek7B FAILED with error %ERRORLEVEL%
) else (
    echo DeepSeek7B DONE
)

echo.
echo === ALL MODELS DONE ===
echo Finished at %date% %time%
