from typing import Optional, List, Dict
from pydantic import BaseModel
from fastapi import UploadFile, File, Form

class FlightHours(BaseModel):
    lower: int
    upper: int

class QueryRequest(BaseModel):
    query: str

class ShapeDetectionResponse(BaseModel):
    success: bool
    message: str
    image_url: Optional[str] = None
    csv_url: Optional[str] = None
    csv_data: Optional[list] = None
    shapes_detected: Optional[int] = None

class ConversationMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class QueryRequestFile(BaseModel):
    query: str
    file_name: str
    pb_number: str
    conversation_history: Optional[List[ConversationMessage]] = None
    session_id: Optional[str] = None  # Session ID for ephemeral FAISS memory
    
class NamesReq(BaseModel):
    pb_number: str

class GetRows(BaseModel):
    pb_number: str
    filename: str

class ExcelFileInput:
    def __init__(
        self,
        file: UploadFile = File(...),
        pb_number: str = Form(...),
        is_scanned: bool = Form(False),
        session_id: Optional[str] = Form(None)
    ):
        self.file = file
        self.pb_number = pb_number
        self.is_scanned = is_scanned
        self.session_id = session_id