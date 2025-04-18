import sys 
import os

import certifi

ca = certifi.where()

from houseprediction.exceptation.exceptation import housepredException
from houseprediction.logging.logger import logging

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request, Response

from uvicorn import run as app_run
from fastapi.responses import Response
from fastapi.responses import RedirectResponse
import pandas as pd

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="./templates")

@app.get("/", tagas=["automation"])
async def index():
    return RedirectResponse(url ="/docs")

@app.train("/train")
async def train()