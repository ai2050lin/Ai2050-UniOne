"""验证model_utils框架 - 逐个模型测试"""
import sys
sys.path.insert(0, "tests/glm5")

import model_utils
import json
import time

def verify_model(name):
    """验证单个模型"""
    print(f"\n{'='*60}")
    print(f"验证 {name}")
    print(f"{'='*60}")
    
    result = {"name": name, "status": "unknown"}
    
    try:
        model, tokenizer, device = model_utils.load_model(name)
        info = model_utils.get_model_info(model, name)
        
        result["model_class"] = info.model_class
        result["n_layers"] = info.n_layers
        result["d_model"] = info.d_model
        result["vocab_size"] = info.vocab_size
        result["mlp_type"] = info.mlp_type
        result["intermediate_size"] = info.intermediate_size
        
        # 验证权重提取
        layer0 = model_utils.get_layers(model)[0]
        lw = model_utils.get_layer_weights(layer0, info.d_model, info.mlp_type)
        
        result["W_q_shape"] = list(lw.W_q.shape)
        result["W_o_shape"] = list(lw.W_o.shape)
        result["W_up_shape"] = list(lw.W_up.shape) if lw.W_up is not None else None
        result["W_gate_shape"] = list(lw.W_gate.shape) if lw.W_gate is not None else None
        result["W_down_shape"] = list(lw.W_down.shape) if lw.W_down is not None else None
        
        W_U = model_utils.get_W_U(model)
        result["W_U_shape"] = list(W_U.shape)
        
        # 简单推理测试
        prompt = "Hello"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with __import__("torch").no_grad():
            out = model(inputs["input_ids"], output_hidden_states=True)
        h0 = out.hidden_states[0][0, -1].float().cpu().numpy()
        result["h0_norm"] = float(__import__("numpy").linalg.norm(h0))
        result["n_hidden_states"] = len(out.hidden_states)
        
        model_utils.release_model(model)
        result["status"] = "PASS"
        
    except Exception as e:
        import traceback
        result["status"] = "FAIL"
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        __import__("torch").cuda.empty_cache()
    
    return result

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen3", "glm4", "deepseek7b"], required=True)
    args = parser.parse_args()
    
    t0 = time.time()
    result = verify_model(args.model)
    result["elapsed"] = round(time.time() - t0, 1)
    
    # 保存结果
    out_path = f"tests/glm5_temp/verify_{args.model}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果保存到: {out_path}")
    print(f"状态: {result['status']}")
    if result["status"] == "PASS":
        print(f"  d_model={result['d_model']}, n_layers={result['n_layers']}, "
              f"vocab={result['vocab_size']}, mlp_type={result['mlp_type']}")
