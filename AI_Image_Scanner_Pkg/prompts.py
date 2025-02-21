system_message = """
You are an expert AI image analysis assistant. Your primary role is to interpret AI-generated image classifications.
You will receive pre-analyzed results and provide conclusions based on those findings.
Your response should focus on explaining what the classification means and offering additional insights.
If the classification suggests the image is AI-generated, discuss possible indicators.
If it suggests the image is real, explain why it appears natural. Keep it to 3-4 sentences max for your reply.
"""

def generate_prompt(analysis_result):
    """
    Generates a structured prompt for GPT-4, ensuring the AI understands the image classification results
    and provides a meaningful conclusion.
    """
    prompt = (
        f"The image provided was analyzed, and the result is: '{analysis_result}'.\n"
        f"Please provide an expert interpretation of this result. Explain why the image was classified this way, "
        f"and discuss any important factors that contributed to the classification. "
        f"Additionally, if the classification is uncertain, explain what further checks might be needed."
    )
    return prompt