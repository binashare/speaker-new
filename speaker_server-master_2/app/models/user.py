from pydantic import BaseModel, Field
from typing import Optional

class UserSchema(BaseModel):
    TenDT: str = Field(...)
    CMND: str = Field(...)
    Phone: str = Field(...)
    QuocTich: str = Field(...)
    QueQuan: str = Field(...)
    NgaySinh: str = Field(...)
    GioiTinh: str = Field(...)
    pathFile: str = Field(...)
    # feature: list = Field(...)
    # class Config:
    #     schema_extra = {
    #         "example": {
    #             "TenDT": "Trần Nhật Trường",
    #             "CMND" : "0961012528",
    #             "Phone": "0961012528",
    #             "QuocTich": "0961012528",
    #             "QueQuan": "0961012528",
    #             "NgaySinh": "0961012528",
    #             "GioiTinh": "0961012528",
    #             "pathFile": "0961012528",
    #             "feature" : []
    #         }
    #     }

class UserLogin(BaseModel):
    CMND: str = Field(...)
    pathFile: str = Field(...)

    class Config:
        schema_extra = {
            "example": {
                "CMND" : "0961012528",
                "pathFile" : "sdfdsf",
            }
        }
class UserUpdate(BaseModel):
    token_code: str = Field(...)
    class Config:
        schema_extra = {
            "example": {
                "token_code" : "",
            }
        }
def ResponseModel(data, message):
    return {
        "data": data,
        "code": 200,
        "message": message,
    }

def ErrorResponseModel(code, message):
    return {"code": code, "message": message}