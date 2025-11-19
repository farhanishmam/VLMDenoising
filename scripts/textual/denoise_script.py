import pandas as pd
import time
import google.generativeai as genai

# ----------------------------
# CONFIG
# ----------------------------
API_KEY = "YOUR_API_KEY_HERE"     # <-- put your Gemini API key
INPUT_CSV = "data/NoisyQuestionPairs.csv"           # your original CSV
OUTPUT_CSV = "./data/Noisy-Denoised_QuestionPairs.csv"

MODEL_NAME = "gemini-2.0-flash"   # the model you want


# ----------------------------
# INIT GEMINI
# ----------------------------
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

# ----------------------------
# LOAD CSV
# ----------------------------
df = pd.read_csv(INPUT_CSV)

# Create column if not exists
if "denoised_question" not in df.columns:
    df["denoised_question"] = ""

# ----------------------------
# PROMPT TEMPLATE
# ----------------------------
PROMPT_TEMPLATE = """
You are an expert at denoising text. Your task is to provide the denoised version of a given noisy question. Follow these instructions:

1. Return only the denoised version of the text or question.
2. Do not provide explanations or additional words.
3. Do not answer the question or alter its intent.
4. Maintain the question format if the input is a question.
5. Avoid presenting the answer in assertive form.

Question: {question}
""".strip()


# ----------------------------
# PROCESS EACH ROW
# ----------------------------
for idx, row in df.iterrows():
    noisy_q = row["modified_question"]

    if pd.isna(noisy_q) or noisy_q.strip() == "":
        df.at[idx, "denoised_question"] = ""
        continue

    prompt = PROMPT_TEMPLATE.format(question=noisy_q)

    # Retry-safe API call
    for _ in range(3):
        try:
            response = model.generate_content(prompt)
            cleaned = response.text.strip()
            df.at[idx, "denoised_question"] = cleaned
            break
        except Exception as e:
            print(f"Error at index {idx}, retrying... ({e})")
            time.sleep(1)

    print(f"Processed row {idx}")

# ----------------------------
# SAVE OUTPUT CSV
# ----------------------------
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved denoised CSV to: {OUTPUT_CSV}")
