import os
from vllm import LLM, SamplingParams
from vllm_angular_steering_sentiment import AngularSteering

def main():
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    # Initialize vLLM (enforce_eager=True is REQUIRED)
    llm = LLM(model="Qwen/Qwen2.5-3B-Instruct", enforce_eager=True, gpu_memory_utilization=0.75)
    # Load and apply steering (using available config file)
    for degree in [0, 180]:
        print("NOW STEERING FOR DEGREE ", degree)

        steering = AngularSteering(llm)
        steering.load_config_from_file("output/Qwen2.5-3B-Instruct/STMT/steering_config-en-max_sim_28_mid-pca_0.npy")
        steering.apply_steering(target_degree=degree, adaptive_mode=0)

        # Example prompts
        prompts = [
            "I'm very upset!",
            "Think of a random person. Describe the person to me, including his mood.",
            "What are your thoughts about the movie JAWS?"
            "How would you rate the average human's happiness from 1 to 10, if you had to give it a rating?"
        ]
        outputs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=64))
        for prompt, output in zip(prompts, outputs):
            print(f"Prompt: {prompt}\nSteered Output: {output.outputs[0].text}\n{'-'*40}")
    print("finishing")

if __name__ == "__main__":
    main()
