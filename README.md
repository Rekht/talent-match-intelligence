# 🧠 Talent Match Intelligence System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-ff4b4b?logo=streamlit)
![Supabase](https://img.shields.io/badge/Supabase-Connected-3fcf8e?logo=supabase)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

An **AI-powered analytics dashboard** built with **Streamlit**, **Supabase**, and **A4F API**,  
designed to analyze and match employee talent profiles using the **TGV/TV (Talent Group Variable / Talent Variable)** framework.  

This tool helps HR teams and organizations identify the best-fit candidates for specific roles  
based on psychometric, competency, teamwork, and performance data.

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
