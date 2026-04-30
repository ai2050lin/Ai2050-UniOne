import time, os

LOG = r"d:\Ai2050\TransformerLens-Project\tests\glm5_temp\cclxxx_log.txt"
t0 = time.time()
prev_size = -1
last_line = ""

while time.time() - t0 < 1800:
    try:
        s = os.path.getsize(LOG)
        f = open(LOG, "r", encoding="utf-8")
        lines = f.readlines()
        f.close()
        last = lines[-1].rstrip() if lines else ""
        if last != last_line or s != prev_size:
            elapsed = time.time() - t0
            print(f"[{elapsed:.0f}s] {last}", flush=True)
            last_line = last
            prev_size = s
        if "All done" in last:
            print("COMPLETE!", flush=True)
            break
    except Exception as e:
        print(f"Error: {e}", flush=True)
    time.sleep(15)

print("Timeout or done. Last 5 lines:", flush=True)
try:
    f = open(LOG, "r", encoding="utf-8")
    lines = f.readlines()
    f.close()
    for l in lines[-5:]:
        print(l.rstrip(), flush=True)
except:
    pass
