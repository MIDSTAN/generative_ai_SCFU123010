import os
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------
# 1. LOAD NVIDIA API KEY
# ---------------------------------------------------------

load_dotenv()

api_key = os.getenv("NVIDIA_API_KEY")

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)


# ---------------------------------------------------------
# 2. PRODUCT IDEA
# ---------------------------------------------------------

product_idea = """
An AI-powered deepfake detection system that analyzes audio,
facial movements, lip-sync, and video together to detect deepfake
videos, identify the exact fake time interval, and explain why
the video was flagged.
"""


# =========================================================
# STEP 1
# Expand product idea into:
# Problem + Solution + Target User
# =========================================================


# -------------------------
# STEP 1 - PROMPT 1
# -------------------------

step_1_prompt_1 = f"""
You are a startup pitch strategist.

Your task is to take a one-line product idea and expand it into
a clear, investor-friendly structured pitch.

PRODUCT IDEA:
{product_idea}

Create the following structure:

1. PROBLEM
- What specific problem exists?
- Who experiences this problem?
- Why is the problem important?
- What is wrong with current solutions?

2. SOLUTION
- What does the proposed product do?
- How does it solve the problem?
- What makes the approach better or different?

3. TARGET USER
- Who will use this product?
- Who is most likely to pay for it?
- Be specific rather than saying "everyone".

Keep the explanation concise, realistic, and easy to understand.

Do not invent unrealistic statistics or claims.

Return the result in a clean, readable format.
Do not return JSON.
"""


# -------------------------
# STEP 1 - PROMPT 2
# -------------------------

step_1_prompt_2 = f"""
You are an expert startup idea analyst.

Expand the following product idea into three sections:

PROBLEM:
Identify the specific customer pain, why it matters, and why
existing solutions are insufficient.

SOLUTION:
Explain how the product directly solves that problem and its
main differentiating value.

TARGET USER:
Identify the specific users/customers who need and may pay for it.

PRODUCT IDEA:
{product_idea}

Use simple, investor-friendly language.
Do not use JSON.
"""


# ---------------------------------------------------------
# Function to call NVIDIA API
# ---------------------------------------------------------

def generate_response(prompt):
    response = client.chat.completions.create(
        model="meta/llama-3.3-70b-instruct",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.7,
        max_tokens=1000
    )

    return response.choices[0].message.content


# ---------------------------------------------------------
# Run STEP 1
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 1 - STRUCTURED PITCH")
print("=" * 60)

# You can use either Step 1 prompt.
structured_pitch = generate_response(step_1_prompt_1)

print(structured_pitch)


# =========================================================
# STEP 2
# Generate investor-style pitch paragraph
# =========================================================
#
# IMPORTANT:
# Step 2 receives the OUTPUT of Step 1,
# NOT the original product idea.
# =========================================================


# -------------------------
# STEP 2 - PROMPT 1
# -------------------------

step_2_prompt_1 = f"""
You are an experienced startup pitch writer.

Using ONLY the structured pitch provided below, write a short
investor-style pitch paragraph.

The paragraph should clearly communicate:

- Who has the problem
- What the problem is
- Why existing solutions are insufficient
- What our solution does
- Why the solution is valuable
- Who the target customer is

Make it sound like a startup pitch to an investor.

Keep it between 80 and 120 words.

Do not introduce new facts that are not present in the structured pitch.
Do not use bullet points.
Do not use headings.
Return only the pitch paragraph.

STRUCTURED PITCH:
{structured_pitch}
"""


# -------------------------
# STEP 2 - PROMPT 2
# -------------------------

step_2_prompt_2 = f"""
You are a professional startup pitch writer.

Convert the structured pitch below into one compelling
investor-style paragraph.

Follow this flow:

Customer → Problem → Existing Gap → Solution → Value → Target Market

Make the paragraph concise, persuasive, and easy to understand.

Avoid technical jargon and exaggerated claims.

Use ONLY information from the structured pitch.

STRUCTURED PITCH:
{structured_pitch}

Return only one paragraph.
"""


# ---------------------------------------------------------
# Run STEP 2
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 2 - INVESTOR PITCH PARAGRAPH")
print("=" * 60)

# You can use either Step 2 prompt.
investor_pitch = generate_response(step_2_prompt_1)

print(investor_pitch)


# =========================================================
# FINAL OUTPUT
# =========================================================

print("\n" + "=" * 60)
print("FINAL RESULT")
print("=" * 60)

print("\n--- STRUCTURED PITCH ---\n")
print(structured_pitch)

print("\n--- INVESTOR PITCH ---\n")
print(investor_pitch)