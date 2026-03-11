import os
import google.generativeai as genai
from dotenv import load_dotenv

# 1. Load the environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ ERROR: GEMINI_API_KEY not found in .env")
    exit()

print(f"✅ Found API Key: {api_key[:5]}... (hidden)")

# 2. Configure Gemini
genai.configure(api_key=api_key)

# 3. List models
print("\n🔍 Querying Google for available models...")
try:
    count = 0
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"  • {m.name}")  # This is the EXACT string you need
            count += 1
    
    if count == 0:
        print("⚠️ No models found! Your API key might be invalid or has no access.")
    else:
        print(f"\n✅ Success! You have access to {count} models.")
        print("👉 Copy one of the names above (e.g., 'models/gemini-1.5-flash') into your script.")

except Exception as e:
    print(f"\n❌ API Error: {e}")