from model.nmt_model import translate_text, SUPPORTED_LANGUAGES

print("Supported:", SUPPORTED_LANGUAGES)
print(translate_text("namaste", "English", "Hindi"))
