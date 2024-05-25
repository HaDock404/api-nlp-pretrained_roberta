from fastapi import FastAPI  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from transformers import RobertaForSequenceClassification  # type: ignore
from transformers import RobertaTokenizer  # type: ignore
from transformers import pipeline  # type: ignore
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)


def load_saving_model():
    """
    Load the pre-trained model for sentiment classification.

    Returns:
        from_pretrained.model: The loaded model.
        from_pretrained.tokenizer: The loaded tokenizer.
    """
    tokenizer_path = "./models/tokenizer_roberta"
    model_path = "./models/model_roberta"
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)
    model = RobertaForSequenceClassification.from_pretrained(model_path)
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
async def display_sentiment(text: str):
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
    text = preprocess_text(text)
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
    result = classifier(text)
    return result