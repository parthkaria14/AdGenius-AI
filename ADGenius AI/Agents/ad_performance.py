import autogen # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import pandas as pd # type: ignore
import numpy as np
import seaborn as sns # type: ignore
import matplotlib.pyplot as plt # type: ignore
from sklearn.model_selection import train_test_split, GridSearchCV # type: ignore
from sklearn.ensemble import RandomForestRegressor # type: ignore
from sklearn.metrics import mean_absolute_error, r2_score # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore

# Set device for CUDA acceleration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Enable quantization using bitsandbytes
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)

# Load Falcon-7B model and tokenizer with quantization
tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")
model = AutoModelForCausalLM.from_pretrained(
    "tiiuae/falcon-7b-instruct", quantization_config=bnb_config
).to(device)

def analyze_ad_performance(report_text):
    """
    Feed the report to Falcon-7B for additional suggestions.
    """
    prompt = (
        f"Please review the following ad performance report and provide additional optimization suggestions:\n\n"
        f"{report_text}\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    outputs = model.generate(**inputs, max_length=1024)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Improved Ad Performance Agent without adding missing columns
class AdPerformanceAgent(autogen.AssistantAgent):
    def __init__(self, name):
        super().__init__(name=name)
    
    def evaluate_ads(self, df):
        # Report warnings only (do not create missing columns)
        expected_columns = [
            'Ad Type', 'Impressions', 'Clicks', 'Conversion Rate', 'CTR', 
            'DayOfWeek', 'Month', 'Spend', 'Conversions'
        ]
        for col in expected_columns:
            if col not in df.columns:
                print(f"Warning: Column '{col}' is missing from the dataset.")
        
        # Build Overall Metrics using only available columns.
        overall_metrics = "## Overall Metrics\n"
        
        if 'Spend' in df.columns:
            total_spend = df['Spend'].sum()
            overall_metrics += f"- Total Spend: ${total_spend:,.2f}\n"
        
        if 'Clicks' in df.columns:
            total_clicks = df['Clicks'].sum()
            overall_metrics += f"- Total Clicks: {total_clicks}\n"
        else:
            total_clicks = None  # used later for weighting
        
        if 'Conversions' in df.columns:
            total_conversions = df['Conversions'].sum()
            overall_metrics += f"- Total Conversions: {total_conversions}\n"
        
        if 'Impressions' in df.columns and 'Clicks' in df.columns:
            impressions_total = df['Impressions'].sum()
            if impressions_total > 0:
                overall_ctr = (df['Clicks'].sum() / impressions_total) * 100
                overall_metrics += f"- Overall CTR: {overall_ctr:.2f}%\n"
        
        if 'Conversion Rate' in df.columns and total_clicks is not None:
            # Calculate a weighted average conversion rate using clicks as weights.
            weighted_conv_rate = (df['Conversion Rate'] * df['Clicks']).sum() / total_clicks
            overall_metrics += f"- Overall Conversion Rate: {weighted_conv_rate:.2f}%\n"
        
        # Static Budget Recommendations remain unchanged.
        budget_recommendations = (
            "## Budget Recommendations\n"
            "1. Increase budget allocation for high-performing segments (e.g., age group 25-34).\n"
            "2. Optimize underperforming ads based on demographic insights.\n"
            "3. Test new creative variations for top-performing topics (e.g., Food).\n"
        )
        
        # Trend Analysis (only if data for Month and Clicks exists)
        trend_analysis = "## Trend Analysis\n"
        if {'Month', 'Clicks'}.issubset(df.columns):
            # Use monthly average CTR if Impressions exists, otherwise just sum clicks per month.
            if 'Impressions' in df.columns:
                monthly_ctr = (
                    df.groupby('Month')
                      .apply(lambda x: (x['Clicks'].sum() / (x['Impressions'].sum() + 1e-6)) * 100)
                      .reset_index(name='CTR')
                )
                overall_ctr_value = None
                if 'Impressions' in df.columns:
                    overall_ctr_value = (df['Clicks'].sum() / df['Impressions'].sum()) * 100
                if overall_ctr_value is not None:
                    december_data = monthly_ctr[monthly_ctr['Month'] == 12]
                    if not december_data.empty and december_data.iloc[0]['CTR'] > overall_ctr_value:
                        trend_analysis += "- Notable increase in CTR during the holiday season.\n"
                    else:
                        trend_analysis += "- No significant CTR boost detected during the holiday season.\n"
            else:
                # If we don't have Impressions, we can show click counts by month.
                monthly_clicks = df.groupby('Month')['Clicks'].sum().reset_index(name='Clicks')
                december_data = monthly_clicks[monthly_clicks['Month'] == 12]
                if not december_data.empty:
                    trend_analysis += "- December shows higher click counts compared to other months.\n"
                else:
                    trend_analysis += "- Insufficient monthly click data for trend analysis.\n"
        else:
            trend_analysis += "- Insufficient data for monthly trend analysis.\n"
        
        # Rural performance analysis only if Region exists.
        if 'Region' in df.columns and {'Clicks', 'Month'}.issubset(df.columns):
            rural_data = df[df['Region'].str.lower() == "rural"]
            if not rural_data.empty:
                if 'Impressions' in rural_data.columns:
                    summer_mask = rural_data['Month'].isin([6, 7, 8])
                    summer_clicks = rural_data[summer_mask]['Clicks'].sum()
                    summer_impressions = rural_data[summer_mask]['Impressions'].sum()
                    summer_ctr = (summer_clicks / (summer_impressions + 1e-6)) * 100 if summer_impressions > 0 else None

                    other_mask = ~rural_data['Month'].isin([6, 7, 8])
                    other_clicks = rural_data[other_mask]['Clicks'].sum()
                    other_impressions = rural_data[other_mask]['Impressions'].sum()
                    other_ctr = (other_clicks / (other_impressions + 1e-6)) * 100 if other_impressions > 0 else None

                    if summer_ctr is not None and other_ctr is not None:
                        if summer_ctr < other_ctr:
                            trend_analysis += "- Consistent performance drop in rural areas during summer months.\n"
                        else:
                            trend_analysis += "- Rural areas do not show a significant performance drop during summer months.\n"
                else:
                    # If no Impressions, simply compare clicks in summer vs. other months.
                    summer_clicks = rural_data[rural_data['Month'].isin([6, 7, 8])]['Clicks'].sum()
                    other_clicks = rural_data[~rural_data['Month'].isin([6, 7, 8])]['Clicks'].sum()
                    if summer_clicks < other_clicks:
                        trend_analysis += "- Rural areas see a drop in clicks during summer months.\n"
                    else:
                        trend_analysis += "- No significant drop in clicks in rural areas during summer months.\n"
            else:
                trend_analysis += "- No rural area data available for detailed trend analysis.\n"
        else:
            trend_analysis += "- 'Region' column or required data for rural trend analysis is missing.\n"
        
        # Combine the sections into one report.
        report = "\n".join([overall_metrics, budget_recommendations, trend_analysis])
        
        # Use the language model to enhance the report with further suggestions.
        enhanced_report = analyze_ad_performance(report)
        final_report = report + "\n\n---\nAI-Enhanced Suggestions:\n" + enhanced_report
        
        return final_report

# Preprocessing function that leaves the DataFrame intact.
def preprocess_data(df):
    df = df.copy()
    df.fillna(0, inplace=True)
    # Do not create any new columns (like Hour or CTR) that aren't already in the data.
    return df

# ML Model for Ad Performance Prediction – now without the 'Hour' column.
def train_performance_model(df):
    df = preprocess_data(df)
    # Use only features that exist; exclude 'Hour'
    features = [feat for feat in ['Clicks', 'CTR', 'DayOfWeek', 'Month'] if feat in df.columns]
    if not features:
        raise ValueError("None of the required features for modeling are available in the dataset.")
    
    target = 'Conversion Rate'
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' is missing from the dataset.")
    
    X = df[features]
    y = df[target]
    
    # Standardize data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
    model_cv = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, n_jobs=-1)
    model_cv.fit(X_train, y_train)
    
    predictions = model_cv.predict(X_test)
    error = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"Model MAE: {error}, R2 Score: {r2}")
    
    return model_cv.best_estimator_, X_test, y_test, predictions

# Visualization and Reporting
def generate_reports(df, model, X_test, y_test, predictions):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Actual Conversion Rate')
    plt.ylabel('Predicted Conversion Rate')
    plt.title('Actual vs Predicted Conversion Rate')
    plt.show()
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_names = [feat for feat in ['Clicks', 'CTR', 'DayOfWeek', 'Month'] if feat in df.columns]
    
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importances')
    plt.bar(range(len(feature_names)), importances[indices], align='center')
    plt.xticks(range(len(feature_names)), feature_names, rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.show()
    
    numeric_df = df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_df.corr(method='spearman')
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Spearman Correlation Matrix')
    plt.show()

# Example Usage
if __name__ == "__main__":
    file_path = "D:/Projects/Datazen/AdGeniusAI-datathon2025/Data/ad_performance.csv"
    df = pd.read_csv(file_path)
    
    # Convert 'Click Time' to datetime (if present) and extract available time features.
    if 'Click Time' in df.columns:
        df['Click Time'] = pd.to_datetime(df['Click Time'])
        if 'DayOfWeek' not in df.columns:
            df['DayOfWeek'] = df['Click Time'].dt.dayofweek
        if 'Month' not in df.columns:
            df['Month'] = df['Click Time'].dt.month

    df = preprocess_data(df)
    agent = AdPerformanceAgent("Ad Optimizer")
    insights = agent.evaluate_ads(df)
    print("AI-Powered Insights:\n", insights)
    
    try:
        trained_model, X_test, y_test, predictions = train_performance_model(df)
        print("Ad Performance Model Trained Successfully.")
        generate_reports(df, trained_model, X_test, y_test, predictions)
    except ValueError as e:
        print("Model training skipped:", e)
