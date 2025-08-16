# FOOD_WASTE_MIUUL
# FOOD BRIDGE  
*A Data-Driven Smart Application to Reduce Food Waste*  

## What Makes Us Different?  
- Donation integration  
- Advanced CRM/RFM analytics  
- Route optimization  
- Data science–based dynamic pricing  
- Corporate dashboard  

---

## 1. Introduction  
Globally, nearly one-third of produced food is wasted before it reaches consumers.  
Restaurants, hotels, catering companies, corporate cafeterias, and supermarkets often throw away surplus food at the end of the day.  

**FOOD BRIDGE** is a mobile/web platform designed to reduce food waste and support small businesses by heavily leveraging **data science techniques**.  

**Core principles:**  
- Businesses list surplus food items at the end of the day.  
- The app applies **real-time dynamic pricing** or redirects items for **donation**.  
- Users can purchase products or donate them directly to NGOs.  
- The system analyzes user behavior, purchase trends, and donation habits, then provides geo-based recommendations and optimizes logistics.  
- CRM and RFM analyses enable customer segmentation and personalized campaigns.  

---

## 2. Core Features  

- **Real-Time Pricing**  
  - Automatic price drops as closing time approaches.  
  - Techniques: XGBoost, LightGBM, Random Forest, Reinforcement Learning (planned).  

- **Predictive Recommendations**  
  - Collaborative Filtering, Matrix Factorization, Neural Collaborative Filtering.  

- **Donation Option**  
  - NGO API integration, donation trend analysis, CRM-based segmentation.  

- **Neighborhood Support**  
  - GeoPandas, Folium for regional analysis.  

- **Corporate Participation**  
  - Dashboards with RFM-based performance measurement.  

- **Categorized Packages**  
  - NLP-based text analysis, category filtering.  

- **Return/Cancellation Policies**  
  - User behavior analysis, credit systems.  

---

## 3. CRM & RFM Analytics  

- **Data Collected for CRM:**  
  - Purchased packages (product, price, date)  
  - Donation frequency  
  - Average basket value  
  - Location-based behavior  

- **RFM Segments & Strategies:**  
  - **Champion**: Frequent buyers, high spend → Exclusive discounts, priority alerts  
  - **At Risk**: Haven’t purchased in a long time → Reminder emails, bonus points  
  - **Potential Loyalist**: New but frequent buyers → Loyalty program suggestions  

- **Predictive Models:**  
  - CLV (Customer Lifetime Value): Gradient Boosting, Random Forest  
  - Churn Prediction: Logistic Regression, XGBoost  

---

## 4. Data Science Applications  

| Analysis               | Goal                               | Techniques |
|-------------------------|------------------------------------|------------|
| Demand Forecasting      | Prevent overproduction             | GradientBoosting, RandomForest, Feature Engineering |
| Dynamic Pricing         | Discounts based on demand/expiry   | XGBoost, planned RL |
| User Segmentation       | Understand buying/donation trends  | K-Means, DBSCAN, PCA |
| Recommendation System   | Suggest best packages              | TF-IDF + Cosine Similarity, planned CF |
| Geo Analysis & Routing  | Optimize donations/delivery routes | K-Means, DBSCAN, Hotspot Analysis |
| Carbon Footprint Calc.  | Show impact of rescued food        | Planned model |
| CRM & RFM               | Maximize customer value            | RFM Scoring, CLV, churn model (planned) |  

---

## 5. Data Sources  

- **Sales & POS Data**:  
  - [Food Demand Forecasting (Kaggle)](https://www.kaggle.com/datasets/kannanaikkal/food-demand-forecasting)  
  - [Simulated POS Data](https://www.kaggle.com/datasets/ganeshabbitota/pos-data-simulated-restaurant-data)  
  - [NYC Restaurant & Inspection Data](https://github.com/rspiro9/NYC-Restaurant-Yelp-and-Inspection-Analysis/blob/main/data.pickle)  

- **User Behavior Data**: For CRM, RFM, CLV, churn.  

- **Product & Category Data**:  
  - [Open Food Facts](https://world.openfoodfacts.org/data)  

- **Geographic & Logistics Data**:  
  - [OpenStreetMap](https://www.openstreetmap.org)  
  - [Food Delivery Dataset](https://www.kaggle.com/datasets/ahsan81/food-ordering-and-delivery-app-dataset)  

- **Donation & NGO Data**:  
  - [NYC Food Insecurity Data](https://github.com/NewYorkCityCouncil/food_insecurity/blob/main/data/input/EFAP_pdf_11_4_24.csv)  
  - [FAO Food Loss & Waste Platform](https://www.fao.org/platform-food-loss-waste/flw-data/en/)  

---

## 6. Expected Outcomes  

- **For Businesses**: Less waste, increased revenue, data-driven insights  
- **For Users**: Affordable meals, donation opportunities, personalized offers  
- **For Society**: Lower carbon footprint, stronger small businesses  
- **For Data Science Community**: Rich datasets and models  

---

## 7. Future Potential  

- NGO, logistics, and payment API integrations  
- International expansion  
- Gamification (badges, loyalty points)  
- Advanced CRM & RFM analyses  
- Improved dynamic pricing with RL  

---

## 📌 Project Vision  
**FOOD BRIDGE aims to build a sustainable ecosystem where technology, social responsibility, and data science intersect to fight food waste and support communities.**
# FOOD BRIDGE  
*A Data-Driven Smart Application to Reduce Food Waste*  

## What Makes Us Different?  
- Donation integration  
- Advanced CRM/RFM analytics  
- Route optimization  
- Data science–based dynamic pricing  
- Corporate dashboard  

---

## 1. Introduction  
Globally, nearly one-third of produced food is wasted before it reaches consumers.  
Restaurants, hotels, catering companies, corporate cafeterias, and supermarkets often throw away surplus food at the end of the day.  

**FOOD BRIDGE** is a mobile/web platform designed to reduce food waste and support small businesses by heavily leveraging **data science techniques**.  

**Core principles:**  
- Businesses list surplus food items at the end of the day.  
- The app applies **real-time dynamic pricing** or redirects items for **donation**.  
- Users can purchase products or donate them directly to NGOs.  
- The system analyzes user behavior, purchase trends, and donation habits, then provides geo-based recommendations and optimizes logistics.  
- CRM and RFM analyses enable customer segmentation and personalized campaigns.  

---

## 2. Core Features  

- **Real-Time Pricing**  
  - Automatic price drops as closing time approaches.  
  - Techniques: XGBoost, LightGBM, Random Forest, Reinforcement Learning (planned).  

- **Predictive Recommendations**  
  - Collaborative Filtering, Matrix Factorization, Neural Collaborative Filtering.  

- **Donation Option**  
  - NGO API integration, donation trend analysis, CRM-based segmentation.  

- **Neighborhood Support**  
  - GeoPandas, Folium for regional analysis.  

- **Corporate Participation**  
  - Dashboards with RFM-based performance measurement.  

- **Categorized Packages**  
  - NLP-based text analysis, category filtering.  

- **Return/Cancellation Policies**  
  - User behavior analysis, credit systems.  

---

## 3. CRM & RFM Analytics  

- **Data Collected for CRM:**  
  - Purchased packages (product, price, date)  
  - Donation frequency  
  - Average basket value  
  - Location-based behavior  

- **RFM Segments & Strategies:**  
  - **Champion**: Frequent buyers, high spend → Exclusive discounts, priority alerts  
  - **At Risk**: Haven’t purchased in a long time → Reminder emails, bonus points  
  - **Potential Loyalist**: New but frequent buyers → Loyalty program suggestions  

- **Predictive Models:**  
  - CLV (Customer Lifetime Value): Gradient Boosting, Random Forest  
  - Churn Prediction: Logistic Regression, XGBoost  

---

## 4. Data Science Applications  

| Analysis               | Goal                               | Techniques |
|-------------------------|------------------------------------|------------|
| Demand Forecasting      | Prevent overproduction             | GradientBoosting, RandomForest, Feature Engineering |
| Dynamic Pricing         | Discounts based on demand/expiry   | XGBoost, planned RL |
| User Segmentation       | Understand buying/donation trends  | K-Means, DBSCAN, PCA |
| Recommendation System   | Suggest best packages              | TF-IDF + Cosine Similarity, planned CF |
| Geo Analysis & Routing  | Optimize donations/delivery routes | K-Means, DBSCAN, Hotspot Analysis |
| Carbon Footprint Calc.  | Show impact of rescued food        | Planned model |
| CRM & RFM               | Maximize customer value            | RFM Scoring, CLV, churn model (planned) |  

---

## 5. Data Sources  

- **Sales & POS Data**:  
  - [Food Demand Forecasting (Kaggle)](https://www.kaggle.com/datasets/kannanaikkal/food-demand-forecasting)  
  - [Simulated POS Data](https://www.kaggle.com/datasets/ganeshabbitota/pos-data-simulated-restaurant-data)  
  - [NYC Restaurant & Inspection Data](https://github.com/rspiro9/NYC-Restaurant-Yelp-and-Inspection-Analysis/blob/main/data.pickle)  

- **User Behavior Data**: For CRM, RFM, CLV, churn.  

- **Product & Category Data**:  
  - [Open Food Facts](https://world.openfoodfacts.org/data)  

- **Geographic & Logistics Data**:  
  - [OpenStreetMap](https://www.openstreetmap.org)  
  - [Food Delivery Dataset](https://www.kaggle.com/datasets/ahsan81/food-ordering-and-delivery-app-dataset)  

- **Donation & NGO Data**:  
  - [NYC Food Insecurity Data](https://github.com/NewYorkCityCouncil/food_insecurity/blob/main/data/input/EFAP_pdf_11_4_24.csv)  
  - [FAO Food Loss & Waste Platform](https://www.fao.org/platform-food-loss-waste/flw-data/en/)  

---

## 6. Expected Outcomes  

- **For Businesses**: Less waste, increased revenue, data-driven insights  
- **For Users**: Affordable meals, donation opportunities, personalized offers  
- **For Society**: Lower carbon footprint, stronger small businesses  
- **For Data Science Community**: Rich datasets and models  

---

## 7. Future Potential  

- NGO, logistics, and payment API integrations  
- International expansion  
- Gamification (badges, loyalty points)  
- Advanced CRM & RFM analyses  
- Improved dynamic pricing with RL  

---
---

## 8. Pipeline & Team Experience  

This project was also developed within a **data pipeline structure**, ensuring that each module (demand forecasting, dynamic pricing, recommendations, donations, geographic analysis) was connected end-to-end in a reproducible workflow.  

Our team of **4 members** divided the responsibilities across data collection, modeling, backend integration, and analytics. This allowed us to simulate a real-world collaborative project environment.  

For most of us, this was a **new and transformative experience in Data Science** — combining machine learning, optimization, and social impact into a single product. Beyond technical skills, it helped us understand teamwork, pipeline design, and the importance of connecting algorithms with real-world societal challenges.  

---

## 🔗 References & Datasets  

- **Food Demand Forecasting Dataset (Kaggle):**  
  https://www.kaggle.com/datasets/kannanaikkal/food-demand-forecasting  

- **POS Data – Simulated Restaurant Data (Kaggle):**  
  https://www.kaggle.com/datasets/ganeshabbitota/pos-data-simulated-restaurant-data  

- **NYC Restaurant & Yelp Inspection Data:**  
  https://github.com/rspiro9/NYC-Restaurant-Yelp-and-Inspection-Analysis/blob/main/data.pickle  

- **Open Food Facts (Product & Category Data):**  
  https://world.openfoodfacts.org/data  

- **OpenStreetMap (Geographic Data):**  
  https://www.openstreetmap.org  

- **Food Ordering & Delivery App Dataset (Kaggle):**  
  https://www.kaggle.com/datasets/ahsan81/food-ordering-and-delivery-app-dataset  

- **NYC Food Insecurity Data:**  
  https://github.com/NewYorkCityCouncil/food_insecurity/blob/main/data/input/EFAP_pdf_11_4_24.csv  

- **FAO Food Loss & Waste Data Platform:**  
  https://www.fao.org/platform-food-loss-waste/flw-data/en/  

---

## 📌 Project Vision  
**FOOD BRIDGE aims to build a sustainable ecosystem where technology, social responsibility, and data science intersect to fight food waste and support communities.**


## 📌 Project Vision  
**FOOD BRIDGE aims to build a sustainable ecosystem where technology, social responsibility, and data science intersect to fight food waste and support communities.**
