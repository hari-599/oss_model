import argparse
import statistics
import time

from assistant import ChatAssistant


HF_SPACE_HARDWARE = {
    "CPU Basic": {"hourly_cost": 0.00, "notes": "Free Space hardware"},
    "CPU Upgrade": {"hourly_cost": 0.03, "notes": "8 vCPU, 32 GB RAM"},
    "Nvidia T4 small": {"hourly_cost": 0.40, "notes": "16 GB VRAM"},
    "Nvidia T4 medium": {"hourly_cost": 0.60, "notes": "16 GB VRAM"},
    "1x Nvidia L4": {"hourly_cost": 0.80, "notes": "24 GB VRAM"},
    "Nvidia A10G small": {"hourly_cost": 1.00, "notes": "24 GB VRAM"},
    "Nvidia A100 large": {"hourly_cost": 2.50, "notes": "80 GB VRAM"},
}

DEFAULT_PROMPTS = [
    "Give me three ideas for a personal productivity assistant.",
    "Summarize why open-source models are useful for student projects.",
    "Write a short friendly reply to someone asking for deployment help.",
    "Explain latency and throughput in simple terms.",
    "Create a quick checklist for launching a Hugging Face Space.",
]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = round((len(sorted_values) - 1) * pct)
    return sorted_values[index]


def estimate_cost_per_1k_chats(hourly_cost: float, avg_latency: float) -> float:
    if hourly_cost == 0:
        return 0.0
    if avg_latency <= 0:
        return 0.0
    requests_per_hour = 3600 / avg_latency
    return (hourly_cost / requests_per_hour) * 1000


def run_benchmark(prompts: list[str]) -> dict[str, float]:
    start_load = time.perf_counter()
    assistant = ChatAssistant()
    load_seconds = time.perf_counter() - start_load

    latencies = []
    output_tokens = []

    for prompt in prompts:
        start = time.perf_counter()
        response = assistant.generate_response(prompt)
        latency = time.perf_counter() - start

        latencies.append(latency)
        output_tokens.append(len(assistant.tokenizer.encode(response)))

    total_tokens = sum(output_tokens)
    total_latency = sum(latencies)

    return {
        "load_seconds": load_seconds,
        "avg_latency": statistics.mean(latencies),
        "p50_latency": statistics.median(latencies),
        "p95_latency": percentile(latencies, 0.95),
        "tokens_per_second": total_tokens / total_latency if total_latency else 0.0,
    }


def build_table(metrics: dict[str, float]) -> str:
    lines = [
        "| Deployment Option | Hourly Cost | Cold Load | Avg Latency | p95 Latency | Tokens/sec | Est. Cost / 1K Chats | Notes |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]

    for name, hardware in HF_SPACE_HARDWARE.items():
        hourly_cost = hardware["hourly_cost"]
        cost_per_1k = estimate_cost_per_1k_chats(
            hourly_cost,
            metrics["avg_latency"],
        )
        hourly_text = "Free" if hourly_cost == 0 else f"${hourly_cost:.2f}/hr"
        cost_text = "Free" if hourly_cost == 0 else f"${cost_per_1k:.2f}"
        lines.append(
            "| "
            f"{name} | "
            f"{hourly_text} | "
            f"{metrics['load_seconds']:.2f}s | "
            f"{metrics['avg_latency']:.2f}s | "
            f"{metrics['p95_latency']:.2f}s | "
            f"{metrics['tokens_per_second']:.2f} | "
            f"{cost_text} | "
            f"{hardware['notes']} |"
        )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Luna OSS assistant latency and estimate Space cost.",
    )
    parser.add_argument(
        "--prompts",
        nargs="*",
        default=DEFAULT_PROMPTS,
        help="Prompts to use for the benchmark.",
    )
    args = parser.parse_args()

    metrics = run_benchmark(args.prompts)
    print(build_table(metrics))


if __name__ == "__main__":
    main()
