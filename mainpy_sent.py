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
        steering.load_config_from_file("output/Qwen2.5-3B-Instruct/STMT/steering_config-en-max_sim_27_mid-pca_0.npy")
        steering.apply_steering(target_degree=degree, adaptive_mode=0)

        # Example prompts
        prompts = [
            # "I'm very upset!",

            "Think of a random person. Describe the person to me, including his mood.",

            "Write a brief overview of a family sitting down for dinner.",

            "Describe the atmosphere of a typical Monday morning in a busy city.",

            "How would you rate the average human's happiness from 1 to 10, if you had to give it a rating? You can use your subjective perspective, don't need to keep it neutral.",

            "Analyze the situation described in this tweet and restate the core event in a single sentence: 'My bike was put on hold...should have known that.... argh total bummer'",

            "Analyze the situation described in this tweet and restate the core event in a single sentence: 'Watching Body of Lies...good film'"

        ]
        outputs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=80))
        for prompt, output in zip(prompts, outputs):
            print(f"Prompt: {prompt}\nSteered Output: {output.outputs[0].text}\n{'-'*40}")
    print("finishing")

if __name__ == "__main__":
    main()
