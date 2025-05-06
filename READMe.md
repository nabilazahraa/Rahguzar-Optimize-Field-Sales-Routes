
# Rahguzar: Optimizing Field Sales Routes

> This project is part of the **Final Year Project (Kaavish)** at **Habib University**, completed by students of the Dhanani School of Science and Engineering, Spring 2025.

**Rahguzar** is an intelligent, web-based platform that automates the creation of Permanent Journey Plans (PJPs) for field sales teams in the FMCG sector. It uses a hybrid algorithmic pipeline combining clustering, evolutionary scheduling, and real-road route optimization to generate efficient and balanced journey plans.

---

## 📌 Project Overview

Traditional PJP planning methods often result in:

- High travel costs
- Unbalanced workloads
- Manual inefficiencies

**Rahguzar** solves these problems by:

- Automating store clustering using MST + K-Means
- Scheduling using Evolutionary Algorithms with real constraints
- Routing using OR-Tools TSP + Dockerized OSRM
- Providing a full-featured interactive map and dashboard interface

---

### Web Application
![Home](/RAHGUZAR%20(KAAVSIH%20REPORT)/images/StartScreen.png)
![Route Planning Interface](/RAHGUZAR%20(KAAVSIH%20REPORT)/images/Map.png)
![Configurations](/RAHGUZAR%20(KAAVSIH%20REPORT)/images/Configure.png)
![Configurations](/RAHGUZAR%20(KAAVSIH%20REPORT)/images/plan.png)
![Configurations](/RAHGUZAR%20(KAAVSIH%20REPORT)/images/Cluster map.png)
![Store Management](/RAHGUZAR%20(KAAVSIH%20REPORT)/images/stores.png)

![Performance Dashboard](/RAHGUZAR%20(KAAVSIH%20REPORT)/images/dashboard.png)

---

## 🔧 Key Features

- **Three-Phase Optimization**: Clustering → Scheduling → Routing
- **Interactive Map Interface**: Dynamic cluster editing, manual overrides
- **Constraint-Aware Planning**: Shift limits, visit frequency, travel time
- **Dashboard KPIs**: Visual performance monitoring
- **Cloud-Based Architecture**: Deployed on AWS EC2 + RDS
- **Secure API Access**: JWT authentication

---

## 🏗️ System Architecture

- **Frontend**: React + Leaflet.js
- **Backend**: Flask + OR-Tools + Evolutionary Algorithm
- **Routing Engine**: Dockerized OSRM on AWS EC2
- **Database**: PostgreSQL on AWS RDS
- **Deployment**: Gunicorn + Nginx on Ubuntu Server

---

## 🚀 Deployment Instructions

### 1. Frontend (React)
```bash
cd frontend
npm install
npm run build
```

### 2. Backend Flask
```bash
cd backend
pip install -r requirements.txt
gunicorn app:app
```
### 3. Routing Engine (OSRM)
```bash
docker run -t -i -p 5002:5000 osrm/osrm-backend osrm-routed /data/map.osrm
```

### 4. PostgreSQL (RDS or Local)
Use schema.sql to initialize database tables.

---

## 🧠 Tech Stack

- **Frontend**: React, Leaflet.js, Axios  
- **Backend**: Flask, Python, OR-Tools, Evolutionary Algorithms  
- **Database**: PostgreSQL  
- **Routing Engine**: Dockerized OSRM  
- **Cloud Deployment**: AWS EC2, AWS RDS  
- **Infrastructure**: JWT Authentication, Nginx, Gunicorn  

---

## 👨‍💻 Team

- **Nabila Zahra** 
- **Muhammad Youshay** 
- **Rabia Shahab** 
- **Iqra Azfar**

---

## 🙏 Acknowledgements

Special thanks to:

- **Syeda Saleha Raza** – Faculty Mentor, Habib University  
- **Fatima Alvi** – Industry Mentor, SalesFlo Pvt Ltd  
- **SalesFlo Pvt Ltd** – Dataset and domain guidance  
