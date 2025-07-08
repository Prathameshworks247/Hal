from pydantic import BaseModel
from fastapi import UploadFile, File, Form

class FlightHours(BaseModel):
    lower: int
    upper: int

class QueryRequest(BaseModel):
    query: str


class QueryRequestFile(BaseModel):
    query: str
    file_name: str
    pb_number: str
    
class NamesReq(BaseModel):
    pb_number: str

class GetRows(BaseModel):
    pb_number: str
    filename: str

class ExcelFileInput:
    def __init__(
        self,
        file: UploadFile = File(...),
        pb_number: str = Form(...)
    ):
        self.file = file
        self.pb_number = pb_number