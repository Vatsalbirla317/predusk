from fastapi import FastAPI
import uvicorn
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.get("/")
async def read_root():
    logger.info("Root endpoint called")
    return {"Hello": "World"}

if __name__ == "__main__":
    logger.info("Starting FastAPI application...")
    uvicorn.run(
        "run_app:app",
        host="127.0.0.1",
        port=8000,
        log_level="debug",
        reload=True
    )
