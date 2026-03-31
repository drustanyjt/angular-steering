import os
from vllm import LLM, SamplingParams
from vllm_angular_steering import AngularSteering

def main():
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    # Initialize vLLM (enforce_eager=True is REQUIRED)
    llm = LLM(model="Qwen/Qwen2.5-7B-Instruct", enforce_eager=True, gpu_memory_utilization=0.75)
    # Load and apply steering (using available config file)
    steering = AngularSteering(llm)
    steering.load_config_from_file("output/STMT/steering_config-en-max_norm_23_post-pca_0.npy")
    steering.apply_steering(target_degree=180, adaptive_mode=1)

    # Example prompts
    prompts = [
        "I'm very upset!",
        "This is really frustrating.",
        "What on earth went down here?"
    ]
    outputs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=64))
    for prompt, output in zip(prompts, outputs):
        print(f"Prompt: {prompt}\nSteered Output: {output.outputs[0].text}\n{'-'*40}")
    print("finishing")

if __name__ == "__main__":
    main()
