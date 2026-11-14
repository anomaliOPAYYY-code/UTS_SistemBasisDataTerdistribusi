import os
import joblib
import requests
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, func
from sqlalchemy.orm import declarative_base, sessionmaker

app = FastAPI(title="SBDT Distributed Inference API")
security = HTTPBearer()

PRIMARY_DB_URL = os.getenv("PRIMARY_DB_URL", "postgresql://postgres:secret@pg-primary:5432/appdb")
REPLICA_DB_URL = os.getenv("REPLICA_DB_URL", "postgresql://postgres:secret@pg-replica:5432/appdb")
KEYCLOAK_JWKS_URL = os.getenv("KEYCLOAK_JWKS_URL", "http://keycloak:8080/realms/sbdt-realm/protocol/openid-connect/certs")
KEYCLOAK_AUDIENCE = os.getenv("KEYCLOAK_AUDIENCE", "account")

primary_engine = create_engine(PRIMARY_DB_URL, pool_pre_ping=True)
replica_engine = create_engine(REPLICA_DB_URL, pool_pre_ping=True)

WriteSession = sessionmaker(bind=primary_engine, autocommit=False, autoflush=False)
ReadSession = sessionmaker(bind=replica_engine, autocommit=False, autoflush=False)

Base = declarative_base()

class InferenceRecord(Base):
    __tablename__ = "inference_records"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    age = Column(Float)
    blood_pressure = Column(Float)
    prediction = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

@app.on_event("startup")
def startup_event():
    Base.metadata.create_all(bind=primary_engine)

model = joblib.load("model.pkl")

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        jwks = requests.get(KEYCLOAK_JWKS_URL, timeout=5).json()
        header = jwt.get_unverified_header(token)
        rsa_key = next(
            (
                {
                    "kty": k["kty"],
                    "kid": k["kid"],
                    "use": k["use"],
                    "n": k["n"],
                    "e": k["e"],
                }
                for k in jwks.get("keys", [])
                if k["kid"] == header["kid"]
            ),
            None,
        )
        if not rsa_key:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Public key JWKS tidak cocok")

        return jwt.decode(
            token,
            rsa_key,
            algorithms=["RS256"],
            audience=KEYCLOAK_AUDIENCE,
            options={"verify_iss": False},
        )
    except JWTError as err:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Token tidak valid: {err}")

class InferenceInput(BaseModel):
    age: float
    blood_pressure: float

@app.post("/predict")
def predict(payload: InferenceInput, user=Depends(verify_token)):
    result = int(model.predict([[payload.age, payload.blood_pressure]])[0])
    db = WriteSession()
    try:
        row = InferenceRecord(
            user_id=user.get("sub", "unknown"),
            age=payload.age,
            blood_pressure=payload.blood_pressure,
            prediction=result,
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return {
            "database_node": "PRIMARY (WRITE ROUTING)",
            "record_id": row.id,
            "prediction": result,
        }
    finally:
        db.close()

@app.get("/history")
def history(user=Depends(verify_token)):
    db = ReadSession()
    try:
        records = db.query(InferenceRecord).order_by(InferenceRecord.id.desc()).limit(10).all()
        return {
            "database_node": "READ-REPLICA (READ ROUTING)",
            "count": len(records),
            "data": [
                {
                    "id": r.id,
                    "user_id": r.user_id,
                    "age": r.age,
                    "blood_pressure": r.blood_pressure,
                    "prediction": r.prediction,
                    "created_at": str(r.created_at),
                }
                for r in records
            ],
        }
    finally:
        db.close()