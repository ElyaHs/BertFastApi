from fastapi import FastAPI
from pydantic import BaseModel, Field
import predict


class Request(BaseModel):
    first_statement: str
    second_statement: str


app = FastAPI()


@app.get("/")
def root():
    return "Hello world"


@app.post("/bert")
def hello(request: Request):
    pred = predict.predict_inference(request.first_statement, request.second_statement)
    good_prediction_message = "The sentences are similar in meaning"
    bad_prediction_message = "The sentences are not similar in meaning"
    return {"message": {good_prediction_message if pred else bad_prediction_message}}

