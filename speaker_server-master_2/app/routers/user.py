from fastapi import APIRouter,Depends, HTTPException,Header,Body, UploadFile, Form, File
from fastapi.encoders import jsonable_encoder

from typing import List, Optional
from server.dependencies import get_token_header
from server.database import add_user, login, update_user, user_helper,FindUserbyCMND
from models.user import UserSchema, UserLogin,UserUpdate, ResponseModel, ErrorResponseModel

from core.voice_utils import extra_feature,compare_similarity

import jwt
from decouple import config

SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

router = APIRouter(
    prefix="/user",
    tags=["user"],
)

@router.get("/add_Speaker")
async def add_Speaker(audio: str, CMND: str):
    try:
        print("da vao day")
        result = extra_feature(audio=audio)
        result = result.detach().cpu().numpy().tolist()
        print(result)
        userbyCMND = await FindUserbyCMND({
            "CMND": CMND
        })
        print(userbyCMND)
        if userbyCMND is None:
            return ErrorResponseModel(404,"So CMND chua duoc dang ky")
        userbyCMND['Feature'] = result
        result_update = await update_user(userbyCMND)
        if result_update:
            return ResponseModel({},"Cập nhật thành công đặc trưng")
        else:
            return ErrorResponseModel(102,"Cập nhật thất bại")
    except Exception as err:
        print('err',err)
        return ErrorResponseModel(400,"Cập nhật thất bại")


@router.post("/login", tags=["user"])
async def user_login(user: UserLogin = Body(...)):
    user_login = jsonable_encoder(user)   
    result = await login(user_login)

    if result is None:
        return ErrorResponseModel(202, "Số điện thoại chưa được đăng ký")

    jwt_token = jwt.encode(user_helper(result),SECRET_KEY)
    result['token_code'] = jwt_token.decode('utf-8')

    result_update = await update_user(result)
    if result_update: 
        return ResponseModel({
            "id_token" : jwt_token
        }, "Login successfully.")
    else : 
        return ResponseModel({
        }, "Login fail")
    
