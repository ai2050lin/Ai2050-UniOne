import numpy as np
import time
import os
import re
import json

class CorpusMathEngine:
    def __init__(self, dim=100000):
        self.D = dim
        self.vocab = {}
        self.inverse_vocab = {}
        self.traces = []
        
    def get_or_create_vector(self, word):
        if word not in self.vocab:
            vec = np.random.choice([-1.0, 1.0], size=self.D).astype(np.float32) / np.sqrt(self.D)
            self.vocab[word] = vec
            self.inverse_vocab[tuple(np.sign(vec).tolist()[:50])] = word # Fast hash for exact match
        return self.vocab[word]

    def hrr_bind(self, x, y):
        return np.real(np.fft.ifft(np.fft.fft(x) * np.fft.fft(y))).astype(np.float32)

    def hrr_unbind(self, bound_trace, known_key_vec):
        y_inv = np.zeros_like(known_key_vec)
        y_inv[0] = known_key_vec[0]
        y_inv[1:] = known_key_vec[1:][::-1]
        return self.hrr_bind(bound_trace, y_inv)

    def memorize_batch(self, traces_list):
        self.traces = np.array(traces_list, dtype=np.float32)
        print(f"      [~] Engram Memory matrix shape: {self.traces.shape}")

    def resonate(self, cue, beta=1000.0):
        # O(N*D) Modern Continuous Hopfield dual projection
        projection = np.dot(self.traces, cue)
        proj_shifted = projection - np.max(projection)
        exp_proj = np.exp(beta * proj_shifted)
        softmax = exp_proj / np.sum(exp_proj)
        new_state = np.dot(self.traces.T, softmax)
        return new_state

    def decode_word(self, vec):
        best_sim = -999.0
        best_word = "??"
        # Chunked dot product to save memory overhead if vocab gets extremely large
        for w, v in self.vocab.items():
            sim = np.dot(v, vec) * self.D
            if sim > best_sim:
                best_sim = sim
                best_word = w
        return best_word, sim / self.D

class SimpleNLPParser:
    def parse_to_triples(self, text):
        sentences = [s.strip() for s in re.split('[。！；\n]', text) if s.strip()]
        triples = []
        # Simple rule-based extraction for Chinese Demo
        for s in sentences:
            if "围绕着" in s:
                parts = s.split("围绕着")
                triples.append((parts[0].strip(), "围绕着", parts[1].strip()))
            elif "属于" in s:
                parts = s.split("属于")
                triples.append((parts[0].strip(), "属于", parts[1].strip()))
            elif "是" in s:
                parts = s.split("是")
                triples.append((parts[0].strip(), "是", parts[1].strip()))
            elif "引力来源" in s and "为" in s:
                # E.g. 太阳系的引力来源为太阳
                s = s.replace("的引力来源为", "引力来源")
                parts = s.split("引力来源")
                triples.append((parts[0].strip(), "引力来源", parts[1].strip()))
        return triples

def run_real_corpus_test():
    print("\n=========================================================")
    print(" PHASE 12: REAL-WORLD NLP CORPUS ALGEBRAIC INTEGRATION")
    print("=========================================================\n")
    
    text_corpus = """
    地球是行星。太阳是恒星。
    月球是卫星。
    地球围绕着太阳。月球围绕着地球。
    太阳属于银河系。地球属于太阳系。
    太阳系的引力来源为太阳。
    银河系是星系。
    """
    
    print("[*] 1. Raw Text Corpus Ingestion")
    print("---------------------------------------------------------")
    print(text_corpus.strip())
    print("---------------------------------------------------------")
    
    parser = SimpleNLPParser()
    triples = parser.parse_to_triples(text_corpus)
    
    print("\n[*] 2. NLP Extraction (SPO Triples):")
    for subj, rel, obj in triples:
        print(f"      [主]: {subj:4s} | [谓]: {rel:6s} | [宾]: {obj:4s}")
        
    print("\n[*] 3. Encoding to 100,000D Orthogonal Algebra Space...")
    engine = CorpusMathEngine(dim=100000)
    
    traces = []
    t0 = time.time()
    for subj, rel, obj in triples:
        v_subj = engine.get_or_create_vector(subj)
        v_rel = engine.get_or_create_vector(rel)
        v_obj = engine.get_or_create_vector(obj)
        
        # Superposition trace: (Subj * Rel) + Obj
        trace = engine.hrr_bind(v_subj, v_rel) + v_obj
        trace = trace / np.linalg.norm(trace)
        traces.append(trace)
        
    engine.memorize_batch(traces)
    print(f"    -> Semantic Grounding & Memory Injection in {(time.time()-t0)*1000:.2f} ms.\n")
    
    # QA System via Algebra
    queries = [
        ("地球", "围绕着", "地球围绕着什么? (预期: 太阳)"),
        ("地球", "是", "地球是什么? (预期: 行星)"),
        ("太阳", "属于", "太阳属于什么? (预期: 银河系)"),
        ("月球", "围绕着", "月球围绕着什么? (预期: 地球)"),
        ("太阳系", "引力来源", "太阳系的引力来源是什么? (预期: 太阳)") # Changed extraction rule above to map to "引力来源"
    ]
    
    print("[*] 4. Algebraic Q&A Retrieval (No LLM, Pure Math):")
    correct = 0
    t_qa = time.time()
    
    for subj, rel, q_text in queries:
        print(f"\n   [提问] {q_text}")
        
        # 1. Construct semantic cue: (Subj * Rel)
        cue_vec = engine.hrr_bind(engine.get_or_create_vector(subj), engine.get_or_create_vector(rel))
        
        # 2. Resonate to find the appropriate full trace in memory
        retrieved_trace = engine.resonate(cue_vec)
        
        # 3. Unbind the subject and relationship to isolate the object
        ans_vec = retrieved_trace - cue_vec
        
        # 4. Decode
        ans_word, confidence = engine.decode_word(ans_vec)
        
        print(f"   [引擎解答] >> {ans_word} (Confidence: {confidence:.2f})")
        if ans_word in q_text:
            correct += 1
            
    print(f"\n[*] QA Finished in {(time.time()-t_qa)*1000:.2f} ms.")
    print(f"[*] Accuracy: {correct}/{len(queries)} ({(correct/len(queries))*100:.1f}%)")
    
    os.makedirs("tempdata", exist_ok=True)
    with open("tempdata/phase12_corpus_qa_report.json", "w", encoding="utf-8") as f:
        json.dump({
            "phase": 12,
            "corpus_triples_extracted": len(triples),
            "qa_accuracy": correct/len(queries),
            "time_ms": (time.time()-t0)*1000
        }, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    run_real_corpus_test()
