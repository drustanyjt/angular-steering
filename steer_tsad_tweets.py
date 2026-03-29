import os
from vllm import LLM, SamplingParams
from vllm_angular_steering import AngularSteering
import pandas as pd
import argparse

ANCHOR_PROMPT = "Rephrase this tweet: "

def load_tweets(csv_path, sentiment, n=10):
    df = pd.read_csv(csv_path, encoding="latin1")
    tweets = df[df["sentiment"] == sentiment]["text"].astype(str).tolist()
    return tweets[:n]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--config", type=str, required=True, help="Path to steering config .npy file")
    parser.add_argument("--csv", type=str, default="tsad/test.csv")
    parser.add_argument("--sentiment", type=str, default="positive", choices=["positive", "negative"])
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--target_degree", type=float, default=180.0)
    parser.add_argument("--adaptive_mode", type=int, default=1)
    parser.add_argument("--prompt_only", action="store_true")
    args = parser.parse_args()

    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    llm = LLM(model=args.model, enforce_eager=True, gpu_memory_utilization=0.75)
    steering = AngularSteering(llm)
    steering.load_config_from_file(args.config)
    steering.apply_steering(target_degree=args.target_degree, adaptive_mode=args.adaptive_mode, prompt_only=args.prompt_only)

    tweets = load_tweets(args.csv, args.sentiment, args.n)
    prompts = [ANCHOR_PROMPT + tweet for tweet in tweets]
    outputs = llm.generate(prompts, SamplingParams(temperature=0.7, max_tokens=64))
    for orig, prompt, output in zip(tweets, prompts, outputs):
        print(f"Original: {orig}\nPrompt: {prompt}\nSteered Output: {output.outputs[0].text}\n{'-'*40}")
    print("finishing")

if __name__ == "__main__":
    main()
