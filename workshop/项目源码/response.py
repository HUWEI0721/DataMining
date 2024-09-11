from fastapi import status
from fastapi.responses import JSONResponse, Response
from typing import Union
import json
from json import JSONEncoder as JSONR
import datetime

class CommonResponse:
    @staticmethod
    def success(data, message: str = "success") -> JSONResponse:
        cls = Myncoder
        content = json.dumps({
            "code": 200,
            "message": message,
            "data": data
        }, cls=cls, ensure_ascii=False)
        return Response(content=content, status_code=status.HTTP_200_OK, media_type="application/json")

    @staticmethod
    def error(code: int, message: str) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "code": code,
                "message": message,
                "data": None
            }
        )

    @staticmethod
    def custom(status_code: int, code: int, message: str, data: Union[dict, list, str] = None) -> JSONResponse:
        return JSONResponse(
            status_code=status_code,
            content={
                "code": code,
                "message": message,
                "data": data
            }
        )

    @staticmethod
    def response(status_code: int, code: int, message: str, data: Union[dict, list, str] = None) -> JSONResponse:
        return JSONResponse(
            status_code=status_code,
            content={
                "code": code,
                "message": message,
                "data": data
            }
        )


class Myncoder(JSONR):
    def default(self, o):
        if isinstance(o, list):
            return [self.default(i) for i in o]
        if isinstance(o, dict):
            return {k: self.default(v) for k, v in o.items()}
        return super().default(o)