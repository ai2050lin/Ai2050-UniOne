@echo off
cd /d d:\Ai2050\TransformerLens-Project
python -u tests/glm5/phase_ccxxxiii_semantic_gravity.py --model deepseek7b --n_pairs 40 --layers last > results\causal_fiber\deepseek7b_ccxxxiii\run_console.log 2>&1
echo DONE_DEEPSEEK7B >> results\causal_fiber\deepseek7b_ccxxxiii\run_console.log

python -u tests/glm5/phase_ccxxxiii_semantic_gravity.py --model qwen3 --n_pairs 40 --layers last > results\causal_fiber\qwen3_ccxxxiii\run_console.log 2>&1
echo DONE_QWEN3 >> results\causal_fiber\qwen3_ccxxxiii\run_console.log

python -u tests/glm5/phase_ccxxxiii_semantic_gravity.py --model glm4 --n_pairs 40 --layers last > results\causal_fiber\glm4_ccxxxiii\run_console.log 2>&1
echo DONE_GLM4 >> results\causal_fiber\glm4_ccxxxiii\run_console.log

echo ALL_DONE > results\causal_fiber\all_ccxxxiii_done.txt
