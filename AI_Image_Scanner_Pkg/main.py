import os
import base64
from dotenv import load_dotenv, find_dotenv
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import UserMessage
from azure.core.credentials import AzureKeyCredential
import analyze
import prompts

# Loads environment variables
load_dotenv(find_dotenv())

# Retrieves API Key
gpt4_api_key = os.environ.get("GPT4_API_KEY")
if not gpt4_api_key:
    raise ValueError("GPT4_API_KEY is not set. Please set it in your .env file.")

image_path = input("Please enter the path to the image file: ")

# Converts the image to Base64
try:
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
except FileNotFoundError:
    print("❌ Error: The file was not found. Please check the path.")
    exit()

# Analyzes the image using `analyze.py` (Multi-Model Analysis)
result_details = analyze.classify_image(base64_image)

# Prints Model Comparisons
print("\n🔍 [Image Analysis Results]")
for model, details in result_details.items():
    print(f"\n🔹 {model}")
    print(f"   - Classification: {details['classification']}")
    print(f"   - Confidence: {details['confidence']:.4f}")

# Format Analysis Result for GPT-4
analysis_result = (
    f"ResNet18 classified the image as {result_details['ResNet18']['classification']} "
    f"(Confidence: {result_details['ResNet18']['confidence']:.4f}). "
    f"EfficientNet-B3 classified the image as {result_details['EfficientNet-B3']['classification']} "
    f"(Confidence: {result_details['EfficientNet-B3']['confidence']:.4f})."
)

# Print Analysis Summary
print(f"\n📝 Summary for AI Model Analysis:\n{analysis_result}")

# Retrieve system message and generate a prompt based on the image analysis
system_message = prompts.system_message
prompt_message = prompts.generate_prompt(analysis_result)  # ✅ Pass analysis_result

# Initialize conversation with system message & AI analysis
initial_conversation = [
    UserMessage(system_message),  # AI Detector System Role
    UserMessage(analysis_result),  # AI Image Classification Result
    UserMessage(prompt_message)  # Request AI's expert interpretation
]

# Create the ChatCompletionsClient
client = ChatCompletionsClient(
    endpoint="https://models.inference.ai.azure.com",
    credential=AzureKeyCredential(gpt4_api_key),
)

# Sends the initial conversation and prints the response
initial_response = client.complete(
    messages=initial_conversation,
    model="gpt-4o",
    temperature=0.0,  # Set to 0 for consistent results
    max_tokens=4096,
    top_p=1
)
initial_ai_reply = initial_response.choices[0].message.content
print("\n🧠 AI (Initial Response Based on Image Analysis):", initial_ai_reply)

# Stores full conversation history
conversation = initial_conversation.copy()
conversation.append(UserMessage(initial_ai_reply))

# Starts Interactive Chat Session
print("\n💬 Interactive Chat Started. Type 'exit' or 'quit' to end the session.\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        print("👋 Exiting chat session.")
        break

    # Appends user input to conversation
    conversation.append(UserMessage(user_input))
    
    # Send updated conversation to GPT-4
    response = client.complete(
        messages=conversation,
        model="gpt-4o",
        temperature=0.0,
        max_tokens=4096,
        top_p=1
    )

    # Extracts & prints AI response
    ai_reply = response.choices[0].message.content
    print("🤖 AI:", ai_reply)

    # Appends AI reply to conversation
    conversation.append(UserMessage(ai_reply))