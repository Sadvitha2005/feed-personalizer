# 📬 Feed Personalization Engine

This project is a FastAPI-based microservice that ranks posts for users using a machine learning model trained on user and post interaction features.

---

## 🚀 Features

- Post ranking based on user profile, tags, timing, and buddy connections.
- Machine learning-powered scoring (LightGBM).
- FastAPI interface with Swagger UI (`/docs`) for testing.
- Dockerized for easy deployment.
- Retrainable with new data.

---

## 🔧 Project Setup

### 🖥️ Local Setup (Without Docker)

#### 📂 1. Clone the Repository
```bash
git clone https://github.com/yourusername/feed-personalizer.git
cd feed-personalizer
```
#### ⚙️ 2. Manual Setup
Follow these steps if you're setting up manually on your local machine:
##### Step 1: Create a Virtual Environment
```bash
python -m venv myenv
```
- To activate in <b>Git Bash:</b>
```bash
source myenv/Scripts/activate
```
- To activate in <b>Windows Powershell:</b>
```powershell
myenv\Scripts\activate
```
##### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```
##### Step 3: Run the Server
```bash
uvicorn main:app --reload --port 8000
```
##### Step 4: Access API
- Swagger UI: http://localhost:8000/docs
- Root message: http://localhost:8000
### 🐳 Docker Setup
<b>1. Ensure Docker is installed and running</b>

Download: https://www.docker.com/products/docker-desktop

<b>2. Build Docker image</b>
```bash
docker build -t feed-personalizer .
```
<b>3. Run the Docker container</b>
```bash
docker run -p 8000:8000 feed-personalizer
```
<b>4. Access the service</b>
- http://localhost:8000
- http://localhost:8000/docs

---
## 📡 Sample API Call

#### Endpoint: ```/rank-feed``` [POST]
##### Sample Request
```json
{
  "user_id": "stu_9999",
  "user_profile": {
    "branches_of_interest": ["AI", "DS"],
    "tags_followed": ["python", "ml"],
    "buddies": ["stu_1010", "stu_2020"],
    "active_hours": ["07:00-09:00", "20:00-23:00"]
  },
  "posts": [
    {
      "post_id": "p1",
      "author_id": "stu_1010",
      "tags": ["ml"],
      "content_type": "text",
      "karma": 90,
      "created_at": "2025-05-27T07:30:00Z"
    }
  ]
}
```
##### Sample Response
```json
{
  "user_id": "stu_9999",
  "ranked_posts": [
    {
      "post_id": "p1",
      "score": 0.91
    }
  ],
  "status": "ranked"
}
```
---
## 🔁 Retrain the Model
