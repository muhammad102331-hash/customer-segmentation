# 🧭 Customer Segmentation Prediction App

**Developed by:** Syed Muhammad Ali  
**GitHub:** [github.com/muhammad102331-hash](https://github.com/muhammad102331-hash)  
**LinkedIn:** [linkedin.com/in/muhammad-ali-64613838b](https://www.linkedin.com/in/muhammad-ali-64613838b/)  
**Email:** muhammadshah36912@gmail.com  

---

## 📊 Project Overview

The **Customer Segmentation Prediction App** is a Machine Learning project developed by **Syed Muhammad Ali** to analyze and segment customers based on behavioral and financial features.  
By applying **K-Means Clustering**, this project helps businesses identify distinct customer groups, enabling **targeted marketing strategies**, **improved customer retention**, and **data-driven decision-making**.

This interactive web app, built with **Streamlit**, allows users to input customer details such as **age, income, spending, and recency**, and instantly predicts which segment or cluster the customer belongs to.

---

## 🧠 Key Features

- 🔹 Real-time customer segment prediction using a trained **K-Means** model  
- 🔹 Interactive **Streamlit** interface for seamless user experience  
- 🔹 Data preprocessing and scaling with **StandardScaler**  
- 🔹 Model persistence using **Joblib** for fast deployment  
- 🔹 Easily extendable for future customer analytics applications  

---

## 🧩 Technologies Used

- **Python 3.10+**  
- **Pandas**, **NumPy** – data manipulation  
- **Scikit-learn** – ML model and scaling  
- **Joblib** – model serialization  
- **Streamlit** – web app interface  

---

## ⚙️ Workflow Summary

1. **Data Preprocessing:**  
   Cleaned and standardized features such as age, income, total spending, and recency.

2. **Model Training:**  
   Used the **K-Means algorithm** to cluster customers based on their attributes.

3. **Model Evaluation:**  
   Applied the **Elbow Method** to determine the optimal number of clusters.

4. **Deployment:**  
   Integrated the model into a **Streamlit app** to make predictions interactively.

---

## 🚀 How to Run Locally

Follow these steps to run the project on your system:

```bash
# 1. Clone the repository
git clone https://github.com/muhammad102331-hash/customer-segmentation.git
cd customer-segmentation

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run segmentation.py
