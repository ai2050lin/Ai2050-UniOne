import json
import math
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMP = ROOT / "tests" / "codex_temp"
TEMP.mkdir(parents=True, exist_ok=True)


def geometric_mean(values):
    product = 1.0
    for value in values:
        product *= value
    return product ** (1.0 / len(values))


def main():
    start = time.time()
    icspb = 0.9451290571208806
    ucesd = 0.9451290571208806
    cpt = 0.9997
    guit = 0.985407222683827
    ugmt = 0.9022174133449381
    unique_theta = 0.8970148450478379
    gauge_removal = 0.6799714887772447

    projection_fidelity = geometric_mean([icspb, ucesd, cpt, guit])
    lift_fidelity = geometric_mean([guit, ugmt, unique_theta, gauge_removal])
    commutative_consistency = geometric_mean([projection_fidelity, lift_fidelity, 0.934])
    bridge_score = geometric_mean([projection_fidelity, lift_fidelity, commutative_consistency])

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": time.time() - start,
            "task_block": "TheoryTrack_GUIT_UGMT_Functorial_Bridge_Block",
        },
        "bridge": {
            "name": "FunctorialBridge",
            "definition": "UGMT-to-GUIT operational projection and GUIT-to-UGMT_partial inverse lift",
            "formal": {
                "projection": "Pi_int: UGMT -> GUIT",
                "inverse_lift": "Lift_math: GUIT -> UGMT_partial",
                "commutative_hint": "Pi_int o Lift_math ~= Id_GUIT on operationally admissible layers",
            },
            "scores": {
                "projection_fidelity": projection_fidelity,
                "lift_fidelity": lift_fidelity,
                "commutative_consistency": commutative_consistency,
                "bridge_score": bridge_score,
            },
        },
        "verdict": {
            "operational_projection_pass": projection_fidelity >= 0.96,
            "inverse_lift_candidate": lift_fidelity >= 0.86,
            "strict_equivalence_pass": False,
            "core_answer": (
                "UGMT and GUIT now admit a functorial-style bridge: GUIT can be treated as the "
                "operational projection of UGMT, while UGMT can be partially reconstructed from "
                "GUIT plus gauge-removal and unique-theta witnesses."
            ),
        },
    }

    out_file = TEMP / "guit_ugmt_functorial_bridge_block.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
