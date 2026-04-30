import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

class ToxicClassificationPipeline:

    def __init__(
        self,
        model_path,
        tokenizer_path,
        label_encoder_path,
        max_length=35
    ):
        self.model = load_model(model_path)

        with open(tokenizer_path, "rb") as f:
            self.tokenizer = pickle.load(f)

        with open(label_encoder_path, "rb") as f:
            self.label_encoder = pickle.load(f)

        self.max_length = max_length

    def preprocess(self, text):
        text = text.lower()
        return text

    def tokenize(self, text):
        sequence = self.tokenizer.texts_to_sequences(
            [text]
        )

        padded = pad_sequences(
            sequence,
            maxlen=self.max_length,
            padding="post",
            truncating="post"
        )

        return padded

    def predict(self, text):

        clean_text = self.preprocess(text)

        tokenized_text = self.tokenize(
            clean_text
        )

        prediction = self.model.predict(
            tokenized_text
        )

        predicted_class = np.argmax(
            prediction,
            axis=1
        )[0]

        predicted_label = (
            self.label_encoder
            .inverse_transform(
                [predicted_class]
            )[0]
        )

        confidence = np.max(prediction)

        return {
            "prediction": predicted_label,
            "confidence": float(confidence)
        }