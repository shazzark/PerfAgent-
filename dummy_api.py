from fastapi import FastAPI
import time

app = FastAPI()

# 1. A fast, healthy endpoint
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "I am fast and healthy!"}

# 2. A slow, bottlenecked endpoint (We will use this to test the bot)
@app.get("/slow")
def slow_endpoint():
    # Simulating a slow database query or heavy computation
    time.sleep(1.5) 
    return {"status": "ok", "message": "Wow, that took a long time."}

# To run this: py -m uvicorn dummy_api:app --reload