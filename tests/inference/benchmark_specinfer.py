import argparse
import json
import os
import time
from types import SimpleNamespace

import flexflow.serve as ff


def load_configs(config_file: str) -> dict:
    if not os.path.isfile(config_file):
        raise FileNotFoundError(f"Config file {config_file} not found")
    with open(config_file, "r") as f:
        return json.load(f)


def create_llm_and_ssms(configs: dict):
    configs_ns = SimpleNamespace(**configs)
    ff_data_type = ff.DataType.DT_FLOAT if configs_ns.full_precision else ff.DataType.DT_HALF
    llm = ff.LLM(
        configs_ns.llm_model,
        data_type=ff_data_type,
        cache_path=configs_ns.cache_path,
        refresh_cache=configs_ns.refresh_cache,
        output_file=configs_ns.output_file,
    )
    ssms = []
    for ssm_cfg in configs_ns.ssms:
        ssm_ns = SimpleNamespace(**ssm_cfg)
        ssm_dtype = ff.DataType.DT_FLOAT if ssm_ns.full_precision else ff.DataType.DT_HALF
        ssm = ff.SSM(
            ssm_ns.ssm_model,
            data_type=ssm_dtype,
            cache_path=ssm_ns.cache_path,
            refresh_cache=ssm_ns.refresh_cache,
            output_file=configs_ns.output_file,
        )
        ssms.append(ssm)
    return llm, ssms


def compile_models(llm, ssms, configs: dict):
    gen_cfg = ff.GenerationConfig(do_sample=False, temperature=0.9, topp=0.8, topk=1)
    for ssm in ssms:
        ssm.compile(
            gen_cfg,
            max_requests_per_batch=configs.get("max_requests_per_batch", 4),
            max_seq_length=configs.get("max_seq_length", 256),
            max_tokens_per_batch=configs.get("max_tokens_per_batch", 64),
            num_kv_cache_slots=configs.get("num_kv_cache_slots", -1),
        )
    llm.compile(
        gen_cfg,
        max_requests_per_batch=configs.get("max_requests_per_batch", 4),
        max_seq_length=configs.get("max_seq_length", 256),
        max_tokens_per_batch=configs.get("max_tokens_per_batch", 64),
        num_kv_cache_slots=configs.get("num_kv_cache_slots", -1),
        ssms=ssms,
    )


def run_benchmark(config_file: str, prompt_file: str, results_file: str):
    configs = load_configs(config_file)
    configs["prompt"] = prompt_file
    configs["output_file"] = results_file
    ff.init(configs)
    llm, ssms = create_llm_and_ssms(configs)
    compile_models(llm, ssms, configs)
    llm.start_server()
    with open(prompt_file, "r") as f:
        prompts = json.load(f)
    output = []
    latencies = []
    for idx, prompt in enumerate(prompts):
        start = time.time()
        res = llm.generate(prompt, max_length=configs.get("max_length", -1))[0]
        latency = time.time() - start
        latencies.append(latency)
        output.append({
            "req_idx": idx,
            "prompt": prompt,
            "response": res.output_text,
            "output_tokens": ",".join(str(tok) for tok in res.output_tokens),
            "num_decoding_steps": len(res.output_tokens),
            "latency": latency,
        })
    llm.stop_server()
    summary = {
        "num_requests": len(prompts),
        "average_latency": sum(latencies) / len(latencies) if latencies else 0.0,
        "total_tokens": sum(o["num_decoding_steps"] for o in output),
        "requests": output,
    }
    with open(results_file, "w") as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Benchmark speculative inference")
    parser.add_argument("--config-file", required=True, help="Path to JSON config file")
    parser.add_argument("--prompt-file", required=True, help="Path to JSON file with prompts")
    parser.add_argument("--results-file", required=True, help="File to save benchmark results")
    args = parser.parse_args()
    run_benchmark(args.config_file, args.prompt_file, args.results_file)


if __name__ == "__main__":
    main()
