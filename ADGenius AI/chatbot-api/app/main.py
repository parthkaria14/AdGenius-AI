from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.llms import HuggingFacePipeline
from langchain.agents import initialize_agent, Tool
from langchain.tools import tool
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import pandas as pd
import torch

# Load datasets
ad_performance_data = pd.read_csv("/code/data/ad_performance.csv")
customer_behavior_data = pd.read_csv("/code/data/marketing_campaign_dataset.csv")

# Ensure column names are properly formatted
ad_performance_data.columns = ad_performance_data.columns.str.strip()
customer_behavior_data.columns = customer_behavior_data.columns.str.strip()

# Initialize FastAPI
app = FastAPI()

class ProductRequest(BaseModel):
    product: str

# Enable 4-bit Quantization using bitsandbytes
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  
    bnb_4bit_quant_type="nf4",  
    bnb_4bit_compute_dtype=torch.float16,  
    llm_int8_enable_fp32_cpu_offload=True  
)

# Load Open-Source LLM (Falcon-7B, Mistral-7B, or Llama-2) in 4-bit mode
model_name = "tiiuae/falcon-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"  
)

# Wrap model in a LangChain-compatible pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)  # Removed `device` argument
llm = HuggingFacePipeline(pipeline=pipe)

# Define AI Agents
@tool
def analyze_ad_performance(product: str):
    """Analyzes ad performance data to identify the highest ROI ad types."""
    performance_summary = ad_performance_data.groupby("Ad Type").agg({
        "Conversion_Rate": "mean",
        "CTR": "mean"
    }).reset_index()

    allocation = (performance_summary["Conversion_Rate"] / performance_summary["Conversion_Rate"].sum()) * 100
    performance_summary["Budget Allocation"] = allocation

    best_ad_type = performance_summary.sort_values("Conversion_Rate", ascending=False).iloc[0]

    return {
        "best_ad_type": best_ad_type["Ad Type"],
        "conversion_rate": round(best_ad_type["Conversion_Rate"], 4),
        "CTR": round(best_ad_type["CTR"], 4),
        "budget_allocation": round(best_ad_type["Budget Allocation"], 2)
    }

@tool
def analyze_customer_behavior(product: str):
    """Examines customer behavior trends to determine the most engaging marketing channels."""
    if "Channel_Used" not in customer_behavior_data.columns:
        return {"error": "Column 'Channel_Used' not found in dataset"}
    
    behavior_summary = customer_behavior_data.groupby("Channel_Used").agg({
        "Clicks": "sum",
        "Conversion_Rate": "mean"
    }).reset_index()

    allocation = (behavior_summary["Clicks"] / behavior_summary["Clicks"].sum()) * 100
    behavior_summary["Budget Allocation"] = allocation

    best_channel = behavior_summary.sort_values("Clicks", ascending=False).iloc[0]

    return {
        "best_ad_placement": best_channel["Channel_Used"],
        "clicks": int(best_channel["Clicks"]),
        "avg_conversion_rate": round(best_channel["Conversion_Rate"], 4),
        "budget_allocation": round(best_channel["Budget Allocation"], 2)
    }

@tool
def generate_marketing_strategy(product: str):
    """Generates AI-driven marketing insights, ad copy, and audience segmentation for the product."""
    prompt = f"""
    Generate a digital marketing strategy for {product}. Provide:
    - Target audience segmentation (Age, Interests, Demographics).
    - Platform recommendations (Google Ads, Facebook, Instagram, LinkedIn).
    - Sample ad copy for social media.
    """

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7)
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    return response

# Map generic ad types to specific online channels
ad_type_to_channel = {
    "Native": "Google Ads",
    "Email": "Email Marketing",
    "Other Channels": "Social Media Ads (Facebook, Instagram, LinkedIn)"
}

# Create LangChain Agent
tools = [analyze_ad_performance, analyze_customer_behavior, generate_marketing_strategy]
marketing_agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

@app.post("/multi_agent_analysis")
async def multi_agent_analysis(request: ProductRequest):
    try:
        product = request.product

        # Run agents
        ad_performance_results = analyze_ad_performance(product)
        customer_behavior_results = analyze_customer_behavior(product)
        marketing_strategy = generate_marketing_strategy(product)

        # Handle potential errors in customer behavior analysis
        if "error" in customer_behavior_results:
            raise HTTPException(status_code=500, detail=customer_behavior_results["error"])

        # Map ad types to specific online channels
        best_ad_type = ad_type_to_channel.get(ad_performance_results["best_ad_type"], ad_performance_results["best_ad_type"])
        best_ad_placement = ad_type_to_channel.get(customer_behavior_results["best_ad_placement"], customer_behavior_results["best_ad_placement"])

        # Determine final budget allocation dynamically
        total_allocation = ad_performance_results["budget_allocation"] + customer_behavior_results["budget_allocation"]
        remaining_allocation = max(0, 100 - total_allocation)  # Ensure it doesn't go negative

        # Replace "Other Channels" with a more specific name
        final_budget_allocation = {
            best_ad_type: ad_performance_results["budget_allocation"],
            best_ad_placement: customer_behavior_results["budget_allocation"],
            "Display Ads": remaining_allocation  # Replaced "Other Channels" with "Display Ads"
        }

        # Generate insights report
        insights_report = f"""
        **Marketing Insights Report for {product}**

        ### Budget Allocation:
        - **{best_ad_type}:** {ad_performance_results["budget_allocation"]}%
        - **{best_ad_placement}:** {customer_behavior_results["budget_allocation"]}%
        - **Display Ads:** {remaining_allocation}%

        ### Performance Metrics:
        - **Best Performing Ad Type:** {best_ad_type}
          - Conversion Rate: {ad_performance_results["conversion_rate"]}
          - CTR: {ad_performance_results["CTR"]}
        - **Most Engaging Marketing Channel:** {best_ad_placement}
          - Total Clicks: {customer_behavior_results["clicks"]}
          - Avg Conversion Rate: {customer_behavior_results["avg_conversion_rate"]}

        ### AI-Generated Marketing Strategy:
        {marketing_strategy}
        """

        return {
            "budget_allocation": final_budget_allocation,
            "insights_report": insights_report
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))