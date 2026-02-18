# Real-Time Fraud Detection API

## Overview
A production-oriented fraud detection system that serves a trained ML model via REST API and logs predictions into PostgreSQL.

## Features
- XGBoost fraud classifier
- RESTful API (Flask)
- PostgreSQL integration
- Dockerized deployment
- Logging
- Input validation
- Unit tests

## API Endpoints

### Health Check
GET /health

### Predict Fraud
POST /predict

{
    "features": [...]
}

Response:
{
    "fraud": 0
}

## Setup

1. Clone repo
2. Create virtual environment
3. Install dependencies
4. Set .env variables
5. Run:

python run.py

## Docker

docker build -t fraud-api .
docker run -p 5000:5000 fraud-api
