import httpx
<<<<<<< HEAD
=======
import numpy as np
>>>>>>> aad18d5bc2355b5acf1895fc334ab7c22b5a675d
import torch
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Initialize FastAPI
app = FastAPI()

# Load Falcon-7B Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "tiiuae/falcon-7b-instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  
    bnb_4bit_quant_type="nf4",  
    bnb_4bit_compute_dtype=torch.float16,  
    llm_int8_enable_fp32_cpu_offload=True  
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"  
)

# Request Model
class ProductRequest(BaseModel):
    product: str

# Function to search platform-specific marketing budget allocation
async def search_ad_allocation(product):
    queries = [
        f"Marketing spend distribution for {product} across Google Ads, Facebook Ads, LinkedIn Ads, YouTube Ads, TV Ads, SEO, and Email.",
        f"How do companies allocate advertising budgets for {product} across different platforms?",
        f"Breakdown of advertising expenses for {product} in percentage across Google, Meta, TV, LinkedIn, and Email."
    ]
    
    for query in queries:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={"query": query},
                headers={"Authorization": "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"}  # Replace with your API key
            )

            if response.status_code == 200:
                data = response.json()
                if 'results' in data and data['results']:
                    return data['results']
    
    return None  # No data found

# Function to extract specific ad platform budget allocation
def extract_ad_allocation(market_data):
    categories = {
        "Google Ads": 0, "Facebook Ads": 0, "YouTube Ads": 0, 
        "Instagram Ads": 0, "TV Ads": 0, "SEO": 0, "Email Marketing": 0
    }
    
    for item in market_data:
        snippet = item.get('snippet', '').lower()
        score = item.get('score', 1.0)

        if "%" in snippet:
            if "google ads" in snippet:
                categories["Google Ads"] += score
            if "facebook ads" in snippet or "meta ads" in snippet:
                categories["Facebook Ads"] += score
            if "youtube ads" in snippet:
                categories["YouTube Ads"] += score
            if "linkedin ads" in snippet:
                categories["LinkedIn Ads"] += score
            if "tv ads" in snippet:
                categories["TV Ads"] += score
            if "seo" in snippet:
                categories["SEO"] += score
            if "email marketing" in snippet:
                categories["Email Marketing"] += score

    total_score = sum(categories.values())
    if total_score == 0:
        return None  # No clear allocation found

    for key in categories:
        categories[key] = round((categories[key] / total_score) * 100, 2)

    return categories

# Function to predict missing allocations with Falcon-7B
def generate_falcon_prediction(product):
    query = f"Predict the ideal ad budget distribution (in percentages) for {product}. Categories: Google Ads, Facebook Ads, YouTube Ads, LinkedIn Ads, TV Ads, SEO, Email Marketing."
    inputs = tokenizer(query, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=150, do_sample=True, temperature=0.7)
    response = tokenizer.decode(output[0], skip_special_tokens=True)

<<<<<<< HEAD
    # Extract percentages using regex
    matches = re.findall(r"(\d+)%", response)
    percentages = [int(m) for m in matches[:7]]  # Take only 7 values

    if len(percentages) == 7:
        total = sum(percentages)
        if total != 100:
            # Normalize to 100%
            percentages = [(p / total) * 100 for p in percentages]
            percentages = [round(p, 2) for p in percentages]  # Round to 2 decimal places

        return {
            "Google Ads": percentages[0],
            "Instagram Ads": percentages[1],
            "YouTube Ads": percentages[2],
            "Facebook Ads": percentages[3],
            "TV Ads": percentages[4],
            "SEO": percentages[5],
            "Email Marketing": percentages[6]
        }
    else:
        # Return None instead of default values if Falcon fails to provide valid output
        return None

# Dynamic insights function based on allocation
def generate_dynamic_insights(budget_allocation):
    insights = "**Ad Budget Distribution Insights**\n\n"
    
    platform_insights = {
        "Google Ads": "Effective for capturing high-intent users. Google Ads often drives high CTR for users actively searching for specific products or services.",
        "Facebook Ads": "Great for targeting specific audiences and remarketing to warm leads. Provides broad reach and strong visual formats.",
        "Instagram Ads": "Perfect for visually-driven campaigns, especially for younger audiences. Works well for influencer marketing and lifestyle brands.",
        "YouTube Ads": "Highly engaging through video content. Great for storytelling and longer-form ads, offering high engagement.",
        "TV Ads": "Mass reach during prime-time and seasonal events. Effective for building awareness in larger markets, especially for mainstream products.",
        "SEO": "Drives organic growth over time. Helps with visibility in search results without the need for direct paid ads.",
        "Email Marketing": "Personalized communication directly with customers. Effective for nurturing leads and converting existing customers."
    }

    for platform, percentage in budget_allocation.items():
        if platform in platform_insights:
            insights += f"- **{platform}**: Allocating {percentage}% to {platform}.\n"
            insights += f"  - {platform_insights[platform]}\n\n"

    # Add performance considerations based on allocation
    insights += "**Performance Metrics Considerations**:\n"
    
    if "Google Ads" in budget_allocation and budget_allocation["Google Ads"] > 20:
        insights += "- **CTR**: Higher Google Ads budgets typically result in a significant increase in CTR for high-intent searches.\n"
    
    if "Facebook Ads" in budget_allocation and budget_allocation["Facebook Ads"] > 15:
        insights += "- **Conversions**: Facebook Ads are great for targeting warm leads and delivering personalized content to boost conversions.\n"

    if "YouTube Ads" in budget_allocation and budget_allocation["YouTube Ads"] > 10:
        insights += "- **Engagement**: Video content tends to generate high engagement rates, making YouTube Ads perfect for brand storytelling.\n"

    if "TV Ads" in budget_allocation and budget_allocation["TV Ads"] > 10:
        insights += "- **Mass Reach**: TV Ads can be used to build widespread brand awareness during prime-time slots.\n"

    if "SEO" in budget_allocation and budget_allocation["SEO"] > 5:
        insights += "- **Organic Growth**: Investing in SEO ensures long-term visibility and drives traffic from organic search.\n"

    if "Email Marketing" in budget_allocation and budget_allocation["Email Marketing"] > 5:
        insights += "- **Lead Nurturing**: Email marketing is one of the best ways to keep customers engaged and drive conversions.\n"

    return insights
=======
    # Extracting percentages using regex
    matches = re.findall(r"(\d+)%", response)
    if len(matches) >= 7:
        return {
            "Google Ads": int(matches[0]),
            "Instagram Ads": int(matches[1]),
            "YouTube Ads": int(matches[2]),
            "Facebook Ads": int(matches[3]),
            "TV Ads": int(matches[4]),
            "SEO": int(matches[5]),
            "Email Marketing": int(matches[6])
        }
    else:
        return {  # Default split if Falcon fails
            "Google Ads": 30, "Facebook Ads": 20, "YouTube Ads": 15, 
            "LinkedIn Ads": 10, "TV Ads": 15, "SEO": 5, "Email Marketing": 5
        }

# Function to generate detailed insights with performance metrics
def generate_insights(budget_allocation):
    query = (
        f"Here is the ad budget split in percentages:\n{budget_allocation}\n"
        "Explain why this distribution is effective using key performance metrics: "
        "CTR (Click-Through Rate), Conversion Rate, Click Rate, Impressions, "
        "Customer Time on the Product, and other relevant stats."
    )
    inputs = tokenizer(query, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=250, do_sample=True, temperature=0.7)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response
>>>>>>> aad18d5bc2355b5acf1895fc334ab7c22b5a675d

# FastAPI Endpoint
@app.post("/allocate_budget")
async def allocate_budget_endpoint(request: ProductRequest):
    try:
        # Step 1: Get market allocation data
        market_data = await search_ad_allocation(request.product)

        # Step 2: Extract platform-specific percentage-based budget split
        budget_allocation = extract_ad_allocation(market_data) if market_data else None

<<<<<<< HEAD
        # Step 3: If no valid allocation found, use Falcon-7B for prediction
        if not budget_allocation:
            budget_allocation = generate_falcon_prediction(request.product)

        # If Falcon fails to provide valid output, handle gracefully
        if budget_allocation is None:
            budget_allocation = {
                "Google Ads": 30, "Facebook Ads": 20, "YouTube Ads": 15,
                "LinkedIn Ads": 10, "TV Ads": 15, "SEO": 5, "Email Marketing": 5
            }
        
        # Step 4: Generate dynamic insights based on budget allocation
        insights = generate_dynamic_insights(budget_allocation)

        return {"budget_allocation_percentage": budget_allocation, "dynamic_insights": insights}
=======
        if not budget_allocation:
            budget_allocation = generate_falcon_prediction(request.product)

        # Step 3: Generate Falcon insights with CTR, Conversion Rates, etc.
        insights = generate_insights(budget_allocation)

        return {"budget_allocation_percentage": budget_allocation, "falcon_insights": insights}
>>>>>>> aad18d5bc2355b5acf1895fc334ab7c22b5a675d
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
