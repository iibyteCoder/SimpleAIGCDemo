from fastapi import FastAPI, APIRouter
from starlette.middleware.cors import CORSMiddleware

from app.api.api import router

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)
app.include_router(router=router)

