import numpy as np
from autogen import AssistantAgent, UserProxyAgent
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# 1️⃣ Check for GPU Availability
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2️⃣ Load Falcon-7B Model with 4-bit Quantization & Proper Offloading
model_name = "tiiuae/falcon-7b-instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  
    bnb_4bit_quant_type="nf4",  
    bnb_4bit_compute_dtype=torch.float16,  
    llm_int8_enable_fp32_cpu_offload=True  # Enables CPU offloading
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"  # Auto-assign layers between GPU/CPU
)

# 3️⃣ Define Falcon-7B Agent for Budget Insights
class FalconAgent:
    def __init__(self):
        self.name = "GenAI_Falcon_Agent"

    def generate_reply(self, query):
        prompt = (
            f"Analyze the following budget allocation, explain why this split was chosen, "
            f"and how it will be beneficial for maximizing marketing performance:\n\n{query}\n\n"
            f"Provide a clear explanation with supporting insights."
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_length=250, do_sample=True, temperature=0.7)
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        return response

genai_agent = FalconAgent()

# 4️⃣ Define ML-Based Budget Allocation Agent
class BudgetAllocationAgent:
    def __init__(self, total_budget, channels, performance_metrics):
        self.total_budget = total_budget
        self.channels = channels
        self.performance_metrics = performance_metrics

    def allocate_budget(self):
        weights = np.array(self.performance_metrics) / sum(self.performance_metrics)
        budget_allocation = weights * self.total_budget
        return {self.channels[i]: round(budget_allocation[i], 2) for i in range(len(self.channels))}

# 5️⃣ Define User Interaction Agent
user_agent = UserProxyAgent(name="User", human_input_mode="ALWAYS")

# 6️⃣ Get User Input
total_budget = float(input("Enter total marketing budget (Rs.): "))
num_channels = int(input("Enter number of marketing channels: "))

channels = []
performance_metrics = []
for _ in range(num_channels):
    channel_name = input("Enter channel name: ")
    metric = float(input(f"Enter performance metric for {channel_name} (ROI, CTR, etc.): "))
    channels.append(channel_name)
    performance_metrics.append(metric)

# 7️⃣ Allocate Budget Using ML-Based Agent
ml_agent = BudgetAllocationAgent(total_budget, channels, performance_metrics)
budget_allocations = ml_agent.allocate_budget()

# 8️⃣ Generate Insights Using Falcon-7B
question = f"Here is the allocated budget:\n{budget_allocations}"
genai_response = genai_agent.generate_reply(question)

# 9️⃣ Display Results
print("\n🔹 **Budget Allocation:**")
for channel, budget in budget_allocations.items():
    print(f"- {channel}: Rs. {budget:.2f}")

print("\n🔹 **Falcon-7B Insights on Budget Allocation:**")
print(genai_response)
