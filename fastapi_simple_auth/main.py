from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.security.utils import get_authorization_scheme_param
from starlette.middleware.base import BaseHTTPMiddleware
import base64

app = FastAPI()

USERNAME = "admin"
PASSWORD = "12345"

PROTECTED_PATHS = ["/docs", "/redoc", "/openapi.json"]

class BasicAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        for path in PROTECTED_PATHS:
            if request.url.path.startswith(path):
                auth = request.headers.get("Authorization")
                scheme, credentials = get_authorization_scheme_param(auth)

                if scheme.lower() != "basic" or not credentials:
                    return Response(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        headers={"WWW-Authenticate": "Basic"},
                        content="Unauthorized",
                    )

                try:
                    decoded = base64.b64decode(credentials).decode("utf-8")
                    username, password = decoded.split(":", 1)
                except Exception:
                    return Response(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        headers={"WWW-Authenticate": "Basic"},
                        content="Invalid auth format",
                    )

                if username != USERNAME or password != PASSWORD:
                    return Response(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        headers={"WWW-Authenticate": "Basic"},
                        content="Invalid credentials",
                    )
                break  # Auth passed for protected path

        return await call_next(request)

app.add_middleware(BasicAuthMiddleware)

