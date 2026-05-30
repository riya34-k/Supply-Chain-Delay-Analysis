📌 Problem Statement

Delivery delays in supply chains directly impact customer satisfaction and business revenue. This project analyses supply chain data to identify patterns in delays, evaluate supplier performance, and understand how inventory levels and risk factors contribute to late deliveries.

📂 Dataset

Domain: Supply Chain / Logistics
Features include: Supplier details, order dates, delivery dates, inventory levels, risk scores, shipping modes, and delay flags
Target: Delivery delay (Yes / No) and delay duration

🛠️ Tools & Libraries

Category Tools
Language Python 3
Data Analysis Pandas, NumPy
Visualisation Matplotlib, Seaborn, Tableau
Machine Learning Scikit-learn 
Environment VS Code

🔍 Project Workflow

1. Data Cleaning & Preprocessing
Handled missing values in delivery date and inventory columns
Parsed and engineered date features (order month, delivery gap, day of week)
Created binary delay flag from actual vs expected delivery dates

2. Exploratory Data Analysis (EDA)
Insight             Finding
Overall delay rate High across most suppliers — not isolated to a few
Supplier reliability Delays are NOT strongly correlated with individual supplier risk scores
Inventory levels    Low inventory periods show higher delay frequency
Shipping mode       Some shipping methods consistently underperform on delivery time
Seasonal patterns   Certain months show spike in delays — likely demand surge periods

3. Machine Learning — Delay Prediction
Built a baseline classification model to predict whether an order would be delayed
Finding: Model accuracy was low (~55–60%), indicating delays are driven by a combination of factors rather than any single predictor
This itself is a key insight — delay is a multi-variable problem that requires systemic fixes, not targeting one variable

4. Tableau Dashboard
Created an interactive dashboard showing:
Supplier-wise delay breakdown
Monthly delay trends
Inventory level vs delay correlation
Shipping mode performance comparison

💡 Key Business Insights

No single root cause — delays result from combinations of low inventory + high demand + shipping mode, not one factor alone
Inventory management is the most actionable lever — stocking up before peak periods reduces delays
Supplier risk scores are not reliable predictors — actual performance tracking is needed
Certain shipping modes show consistently longer delivery gaps — consider renegotiating SLAs

📁 Project Structure
supply-chain-analysis/
│
├── data/
│   └── supply_chain_data.csv
│
├── notebooks/
│   └── supply_chain_analysis.ipynb
│
├── supply_chain_analysis.py
├── dashboard_screenshot.png
└── README.md

🚀 How to Run

# 1. Clone the repo
git clone https://github.com/riya34-k/supply-chain-analysis.git
# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn
# 3. Run the script
python supply_chain_analysis.py
