import numpy as np
import time
import os
import json

class TopologicalNoiseEngine:
    def __init__(self, dim=10000):
        self.D = dim
        self.vocab = {}
        self.W = np.zeros((dim, dim), dtype=np.float32)

    def bind_concept(self, word):
        vec = np.random.choice([-1.0, 1.0], size=self.D).astype(np.float32)
        self.vocab[word] = vec / np.sqrt(self.D)
        return self.vocab[word]

    def _random_bipolar(self):
        vec = np.random.choice([-1.0, 1.0], size=self.D).astype(np.float32)
        return vec / np.sqrt(self.D)

    def hrr_bind(self, x, y):
        return np.real(np.fft.ifft(np.fft.fft(x) * np.fft.fft(y))).astype(np.float32)

    def hrr_unbind(self, bound_trace, known_key_vec):
        y_inv = np.zeros_like(known_key_vec)
        y_inv[0] = known_key_vec[0]
        y_inv[1:] = known_key_vec[1:][::-1]
        return self.hrr_bind(bound_trace, y_inv)

    def memorize(self, vec):
        self.W += np.outer(vec, vec)
        np.fill_diagonal(self.W, 0)

    def resonate(self, noisy_cue, max_steps=15):
        state = np.copy(noisy_cue)
        for i in range(max_steps):
            new_state = np.dot(self.W, state)
            new_state = np.sign(new_state)
            new_state[new_state == 0] = 1.0
            new_state = new_state / np.sqrt(self.D)
            sim = np.dot(new_state, state) * self.D
            if sim >= (self.D * 0.999): 
                return new_state, i+1
            state = new_state
        return state, max_steps

    def decode(self, vec):
        best_sim = -999.0
        best_word = "??"
        for w, v in self.vocab.items():
            sim = np.dot(v, vec) * self.D
            if sim > best_sim:
                best_sim = sim
                best_word = w
        return best_word, float(best_sim)

def run_noise_horizon_test():
    print(f"\n=========================================================")
    print(f" M3: NOISE CATASTROPHE HORIZON TEST (Phase Transition)")
    print(f"=========================================================\n")
    
    engine = TopologicalNoiseEngine(dim=10000)
    words = ['A', 'B', 'VERB1', 'VERB2', 'X', 'Y', 'Z']
    for w in words: engine.bind_concept(w)
    
    print("[*] Loading 50 traces into the memory matrix to introduce interference...")
    target_trace = engine.hrr_bind(engine.vocab['VERB1'], engine.vocab['A']) + engine.vocab['X']
    target_trace /= np.linalg.norm(target_trace)
    engine.memorize(target_trace)
    
    for i in range(49):
        t = engine.hrr_bind(engine.vocab['VERB2'], engine._random_bipolar() if i%2==0 else engine.vocab['B']) + engine._random_bipolar()
        t /= np.linalg.norm(t)
        engine.memorize(t)
        
    noise_levels = [0.10, 0.30, 0.50, 0.70, 0.85, 0.90, 0.95]
    print("[*] Commencing Noise Injection & Attractor Phase Transition Sweep:\n")
    
    results = {}
    
    for noise in noise_levels:
        noisy_trace = target_trace.copy()
        flip = np.random.rand(10000) < noise
        noisy_trace[flip] *= -1
        
        t0 = time.time()
        recovered, steps = engine.resonate(noisy_trace, max_steps=15)
        
        # Test if X is recovered directly (VSA superposition natively handles it since VERB1*A is not in vocab)
        ans, conf = engine.decode(recovered)
        
        status = "[HIT]" if ans == 'X' else "[CRASH: Hallucination]"
        print(f"    -> Noise: {noise*100:2.0f}% | Steps: {steps:2d} | Guessed: {ans:2s} (Conf: {conf:6.1f}) | {status} | {(time.time()-t0)*1000:.1f}ms")
        
        results[f"Noise_{int(noise*100)}"] = {
            "Status": "Recovered" if ans == 'X' else "Hallucination/Collapse",
            "Steps": steps,
            "Accuracy": 1 if ans == 'X' else 0
        }
        
    os.makedirs("tempdata", exist_ok=True)
    with open("tempdata/phase10_m3_noise_report.json", "w") as f:
        json.dump({
            "Phase": "10 M3 - Noise Catastrophe Horizon",
            "Results": results
        }, f, indent=2)

if __name__ == "__main__":
    run_noise_horizon_test()
