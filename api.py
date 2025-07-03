from fastapi import FastAPI  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from transformers import RobertaForSequenceClassification  # type: ignore
from transformers import RobertaTokenizer  # type: ignore
from transformers import pipeline  # type: ignore
import re
from pydantic import BaseModel
import tensorflow as tf
import os
import zipfile


class TextPayload(BaseModel):
    text: str


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)


# def load_saving_model():
#    """
#    Load the pre-trained model for sentiment classification.

#    Returns:
#        from_pretrained.model: The loaded model.
#        from_pretrained.tokenizer: The loaded tokenizer.
#    """
#   tokenizer_path = "./models/tokenizer_roberta"
#   model_path = "./models/model_roberta"

#   tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)
#   model = RobertaForSequenceClassification.from_pretrained(model_path)
#    return model, tokenizer

def download_and_extract(zip_url, extract_to):
    local_zip = tf.keras.utils.get_file(
        fname=os.path.basename(zip_url),
        origin=zip_url,
        extract=False
    )
    with zipfile.ZipFile(local_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_to)


def load_saving_model():
    """
    Load the pre-trained model for sentiment classification.
    """
    model_zip_gcs = \
        "https://storage.googleapis.com/hadock404-models/model_roberta.zip"
    tokenizer_zip_gcs = \
        "https://storage.googleapis.com/hadock404-models/tokenizer_roberta.zip"

    model_local_dir = "/tmp/model_roberta"
    tokenizer_local_dir = "/tmp/tokenizer_roberta"

    if not os.path.exists(model_local_dir):
        download_and_extract(model_zip_gcs, "/tmp/")

    if not os.path.exists(tokenizer_local_dir):
        download_and_extract(tokenizer_zip_gcs, "/tmp/")

    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_local_dir)
    model = RobertaForSequenceClassification.from_pretrained(model_local_dir)

    return model, tokenizer


def preprocess_text(preprocess_sentence):
    """
    text simplification by removing user identification,
    web links, single characters,
    numeric characters,
    non-alphanumeric characters.

    Returns:
        string: The processed sentence.
    """
    preprocess_sentence = re.sub(r'\S*@\S*\s?', '', preprocess_sentence)
    preprocess_sentence = re.sub(r'http\S+', '', preprocess_sentence)
    preprocess_sentence = re.sub(r'[^\w\s]', '', preprocess_sentence)
    preprocess_sentence = re.sub(r'\b\w\b', '', preprocess_sentence)
    preprocess_sentence = re.sub(r'\d', '', preprocess_sentence)
    preprocess_sentence = re.sub(r'\s+', ' ', preprocess_sentence)
    preprocess_sentence = preprocess_sentence.lower()
    return preprocess_sentence


@app.get("/")
def hello():
    """
    Default route to welcome users and direct them to documentation.

    Returns:
        dict: A welcome message.
    """
    return {"message": "Hi, add /docs to the URL to use the API."}


@app.post("/predict_sentiment")
async def display_sentiment(payload: TextPayload):
    """
    Endpoint to upload a text, text processing,
    generate a sentiment classification,
    and return the result.

    Args:
        str (text): The tweet.

    Returns:
        Response: The response containing the predicted sentiment.
    """
    model, tokenizer = load_saving_model()
    text = preprocess_text(payload.text)
    classifier = pipeline("text-classification", model=model,
                          tokenizer=tokenizer)
    result = classifier(text)
    return result
