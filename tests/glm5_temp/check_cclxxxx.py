import time
time.sleep(180)
lines = open(r'd:\Ai2050\TransformerLens-Project\tests\glm5_temp\cclxxxx_log.txt', encoding='utf-8').readlines()
print(f'Total: {len(lines)} lines')
for l in lines[-5:]:
    print(l.strip())
