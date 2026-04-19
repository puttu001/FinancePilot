from fastapi import FastAPI
from api.routers import router
from utils.logging import set_logger
from utils.config import load_config
logger = set_logger(__name__)
config = load_config()

app = FastAPI(title='FinancePilot API')
app.include_router(router)
@app.get('/')
def root()->dict:
    logger.info("Root endpoint accessed")
    return {'message':'FinancePilot bcakend running'}




if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
    