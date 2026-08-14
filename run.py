import uvicorn

if __name__ == "__main__":
    print("==================================================================")
    print("🚀 Starting PPE Monitoring System (AI Safety Analytics)")
    print("🌐 API Server: http://localhost:8000")
    print("🖥️ Web Dashboard: http://localhost:8000/app/")
    print("==================================================================")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
