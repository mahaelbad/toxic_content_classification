from pipeline import ToxicClassificationPipeline


pipeline = ToxicClassificationPipeline(
    model_path="toxic_model.keras",
    tokenizer_path="tokenizer.pkl",
    label_encoder_path="label_encoder.pkl"
)


text = input("Enter your text: ")

result = pipeline.predict(text)

print("\nPrediction Result:")
print(result)