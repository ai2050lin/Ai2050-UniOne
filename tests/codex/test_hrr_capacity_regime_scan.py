import argparse
import json
import math
from datetime import datetime


def top1_error_upper_bound(d: int, m: int, n: int, c: float) -> float:
    # P(err) <= (N-1) * exp(-C*d/M)
    val = (n - 1) * math.exp(-c * d / m)
    return min(1.0, val)


def required_dimension(m: int, n: int, eps: float, c: float) -> float:
    # d >= (M/C) * (log(N-1) + log(1/eps))
    return (m / c) * (math.log(n - 1) + math.log(1.0 / eps))


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan HRR capacity regime from theoretical bounds.")
    parser.add_argument("--n", type=int, default=151936, help="Candidate size N")
    parser.add_argument("--d", type=int, default=4096, help="Current dimension d")
    parser.add_argument("--m-list", type=str, default="5,20,50,100,200,400", help="Comma-separated M list")
    parser.add_argument("--c-list", type=str, default="0.05,0.08,0.10", help="Comma-separated C list")
    parser.add_argument("--eps-list", type=str, default="1e-1,1e-2,1e-3", help="Comma-separated eps list")
    parser.add_argument("--json-out", type=str, default="", help="Output path")
    args = parser.parse_args()

    m_list = [int(x.strip()) for x in args.m_list.split(",") if x.strip()]
    c_list = [float(x.strip()) for x in args.c_list.split(",") if x.strip()]
    eps_list = [float(x.strip()) for x in args.eps_list.split(",") if x.strip()]

    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "N": args.n,
            "d": args.d,
            "M_list": m_list,
            "C_list": c_list,
            "eps_list": eps_list,
        },
        "error_bound_scan": [],
        "required_dimension_scan": [],
    }

    for c in c_list:
        group = {"C": c, "rows": []}
        for m in m_list:
            p = top1_error_upper_bound(args.d, m, args.n, c)
            group["rows"].append({"M": m, "p_err_upper": p})
        result["error_bound_scan"].append(group)

    for c in c_list:
        group = {"C": c, "rows": []}
        for eps in eps_list:
            row = {"eps": eps, "required_d": []}
            for m in m_list:
                row["required_d"].append({"M": m, "d_min": required_dimension(m, args.n, eps, c)})
            group["rows"].append(row)
        result["required_dimension_scan"].append(group)

    text_lines = []
    text_lines.append("=== HRR Error Upper Bound Scan ===")
    for group in result["error_bound_scan"]:
        text_lines.append(f"C={group['C']}")
        for row in group["rows"]:
            text_lines.append(f"  M={row['M']:>4d} -> p_err_upper={row['p_err_upper']:.6g}")

    text_lines.append("\n=== Required Dimension Scan ===")
    for group in result["required_dimension_scan"]:
        text_lines.append(f"C={group['C']}")
        for row in group["rows"]:
            eps = row["eps"]
            line = ", ".join([f"M={x['M']}:d>={int(round(x['d_min']))}" for x in row["required_d"]])
            text_lines.append(f"  eps={eps}: {line}")

    print("\n".join(text_lines))

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nSaved JSON: {args.json_out}")


if __name__ == "__main__":
    main()
