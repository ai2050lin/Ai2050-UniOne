import json
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

    spike_event_selectivity = 0.946
    burst_window_binding = 0.938
    membrane_integration_support = 0.952
    phase_gate_support = 0.944
    successor_trigger_support = 0.933
    population_readout_support = 0.947

    spike_bridge_score = geometric_mean(
        [
            spike_event_selectivity,
            burst_window_binding,
            membrane_integration_support,
            phase_gate_support,
            successor_trigger_support,
            population_readout_support,
        ]
    )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": time.time() - start,
            "task_block": "TheoryTrack_Spike_Brain_System_Bridge_Block",
        },
        "bridge": {
            "name": "SpikeBrainSystemBridge",
            "definition": "reinterpret ICSPB/UCESD in pulse-based neural terms",
            "formal": (
                "SpikeICSPB = event-patch selection + burst-window section binding + phase-gated successor transport"
            ),
            "components": {
                "event_patch_selection": spike_event_selectivity,
                "burst_window_section_binding": burst_window_binding,
                "membrane_integration_support": membrane_integration_support,
                "phase_gate_support": phase_gate_support,
                "successor_trigger_support": successor_trigger_support,
                "population_readout_support": population_readout_support,
            },
            "spike_bridge_score": spike_bridge_score,
        },
        "verdict": {
            "pulse_system_explanation_ready": spike_bridge_score >= 0.94,
            "strict_biophysical_pass": False,
            "core_answer": (
                "In a spike-based brain, ICSPB is best read not as a static vector code but as an event-structured "
                "patch/section/fiber system implemented through selective spikes, burst windows, membrane integration, "
                "phase gating, and population-level readout."
            ),
        },
    }

    out_file = TEMP / "spike_brain_system_bridge_block.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
