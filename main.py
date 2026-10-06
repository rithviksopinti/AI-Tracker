import io
import os
import asyncio
import certifi
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv
from PIL import Image
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from motor.motor_asyncio import AsyncIOMotorClient

# 1. Load Environment Variables
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
mongo_uri = os.getenv("MONGO_URI")

if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable missing in .env file.")
if not mongo_uri:
    raise ValueError("MONGO_URI environment variable missing in .env file.")

# 2. App & Database Setup
app = FastAPI(title="AI Budget Tracker API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect to MongoDB Atlas using certifi SSL certificates
try:
    mongo_client = AsyncIOMotorClient(mongo_uri, tlsCAFile=certifi.where())
    db = mongo_client.budget_tracker
    expenses_collection = db.expenses
except Exception as e:
    print(f"MongoDB Initialization Error: {e}")

# Gemini AI Client Setup
client = genai.Client(api_key=api_key)

# 3. Data Schemas
class ExpenseAnalysis(BaseModel):
    item_name: str = Field(description="Name or title of the item/expense identified.")
    price: float = Field(description="Estimated or actual price in USD (numeric value only).")
    category: str = Field(
        description="Category such as Food & Dining, Electronics, Entertainment, Groceries, Utilities, Transportation, Shopping, or Other."
    )
    confidence_notes: str = Field(
        description="Brief 1-sentence reasoning for the identified item, price, or category."
    )

# 4. API Endpoints
@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/api/analyze-expense", response_model=ExpenseAnalysis)
async def analyze_expense(
    text_prompt: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    if not text_prompt and not file:
        raise HTTPException(
            status_code=400, 
            detail="Please provide at least a text description or an image file."
        )

    contents = []
    
    # Handle Image Upload
    if file:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
        
        image_bytes = await file.read()
        pil_image = Image.open(io.BytesIO(image_bytes))
        contents.append(pil_image)

    # Handle Text Input
    if text_prompt:
        contents.append(f"User description/context: {text_prompt}")

    system_instruction = (
        "You are an expert expense parser for a budget tracking application. "
        "Analyze the provided image and/or text context to identify what item was purchased, "
        "its price, and its expense category. If price is not explicitly shown, estimate a fair market price."
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Generate structured response from Gemini
            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    response_schema=ExpenseAnalysis,
                ),
            )

            structured_data = ExpenseAnalysis.model_validate_json(response.text)

            # SAVE TO MONGODB ATLAS
            expense_doc = structured_data.model_dump()
            expense_doc["created_at"] = datetime.utcnow().isoformat()

            print("Attempting to save expense to MongoDB Atlas...")
            result = await expenses_collection.insert_one(expense_doc)
            print(f"✅ SUCCESS: Saved to MongoDB with Document ID: {result.inserted_id}")

            return structured_data

        except Exception as e:
            print(f"❌ DATABASE/AI ERROR: {type(e).__name__} - {str(e)}")
            if "503" in str(e) and attempt < max_retries - 1:
                await asyncio.sleep(2 * (attempt + 1))
                continue
            raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/api/expenses")
async def get_expenses():
    expenses = []
    try:
        async for doc in expenses_collection.find({}, {"_id": 0}).sort("created_at", -1):
            expenses.append(doc)
        return expenses
    except Exception as e:
        print(f"❌ FETCH ERROR: {e}")
        raise HTTPException(status_code=500, detail=f"Database Fetch Error: {str(e)}")