# 🧠 Talent Match Intelligence System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-ff4b4b?logo=streamlit)
![Supabase](https://img.shields.io/badge/Supabase-Connected-3fcf8e?logo=supabase)

An **AI-powered analytics dashboard** built with **Streamlit**, **Supabase**, and **A4F API**,  
designed to analyze and match employee talent profiles using the **TGV/TV (Talent Group Variable / Talent Variable)** framework.  

This tool helps HR teams and organizations identify the best-fit candidates for specific roles  
based on psychometric, competency, teamwork, and performance data.

---

## ⚠️ API Service Notice

> **ℹ️ A4F API Service Alert (Updated November 2025)**  
> Some users may experience **403 errors or slow responses** when using the **A4F AI endpoints**.  
> This is due to **ongoing performance issues** with the A4F Free Tier API.  
> You can monitor the real-time status at:  
> 🔗 [https://www.a4f.co/status](https://www.a4f.co/status)

If you encounter an error like:
AI API Error: 403
Failed to generate job profile. Please check API connection or try again.
Please note that this issue is temporary and related to A4F API availability, not the app configuration.

---

## ✨ Features

- 🤖 **AI-Generated Job Profiles** – Automatically creates detailed job descriptions using the A4F API  
- ⚡ **Dynamic Talent Matching Algorithm** – Real SQL-based matching powered by Supabase  
- 📊 **Comprehensive Dashboard** – Interactive visualizations for match distribution and candidate rankings  
- 🧱 **Benchmark-Based Matching** – Uses top performers (rating 5) as role baselines  
- 💾 **Exportable Results** – Download complete matching analysis in CSV format  

---

## 🧩 Tech Stack

| Layer | Technology |
|-------|-------------|
| Frontend | Streamlit |
| Backend | Supabase |
| Database | PostgreSQL (Supabase) |
| AI Engine | A4F API (`provider-3/gpt-5-nano`) |
| Visualization | Plotly |
| Config | python-dotenv |
| Language | Python 3.9 + |

---

## ⚙️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/talent-match-intelligence.git
cd talent-match-intelligence
```

### 2️⃣ Create and Activate Virtual Environment
```bash
python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

requirements.txt includes:
```bash
streamlit
pandas
plotly
requests
supabase
numpy
python-dotenv
```

### 4️⃣ Configure Environment Variables
⚠️ The .env file is not included in the repository — you must create it manually.

```bash
# Supabase Configuration
SUPABASE_URL=https://your-supabase-url.supabase.co
SUPABASE_KEY=your-supabase-service-role-key

# A4F API Configuration
A4F_API_KEY=your-a4f-api-key
A4F_MODEL=provider-3/yout-model
```

### ▶️ Running the App
```bash
streamlit run streamlit_app.py
```
After the setup is complete, launch the dashboard using: streamlit run streamlit_app.py
