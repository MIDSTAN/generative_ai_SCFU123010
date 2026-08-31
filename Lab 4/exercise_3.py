import os
import json
from dotenv import load_dotenv
from openai import OpenAI


# =========================================================
# 1. Load NVIDIA API
# =========================================================

load_dotenv()

api_key = os.getenv("NVIDIA_API_KEY")

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)


# =========================================================
# 2. Prompts
# =========================================================

step_1 = """
You are a meeting transcript analysis expert.

Your task is to analyze a raw meeting transcript and extract the important information discussed.

1. Identify the main topics discussed.
2. Summarize the key points under each topic.
3. Identify important decisions or conclusions made.
4. Identify problems, issues, concerns, or blockers mentioned.
5. Ignore greetings, small talk, repetition, and irrelevant conversation.
6. Do not invent or assume information.

Return ONLY valid JSON using this structure:

{
    "topics": [],
    "key_points": [],
    "decisions": [],
    "issues_blockers": []
}
"""


step_2 = """
You are a meeting action-item extraction expert.

Using the meeting discussion, identify every actionable task or follow-up.

For each action item:

1. Describe what needs to be done.
2. Identify the owner if explicitly mentioned.
3. Identify the deadline if explicitly mentioned.
4. Identify dependencies or relevant context.
5. Flag missing owner.
6. Flag missing deadline.
7. Do not invent an owner or deadline.

Return ONLY valid JSON using this structure:

{
    "action_items": [
        {
            "task": "",
            "owner": "",
            "deadline": "",
            "dependency_context": "",
            "missing_information": ""
        }
    ]
}
"""


step_3 = """
You are a meeting task-management formatter.

Using the extracted action items, create a clean structured task table.

For every task include:

- ID
- Task
- Owner
- Deadline
- Dependency/Context
- Status
- Missing Information

Rules:

1. Assign sequential IDs.
2. Use "Not specified" when owner is missing.
3. Use "Not specified" when deadline is missing.
4. Clearly flag missing owner/deadline.
5. Do not invent information.
6. Keep tasks concise and actionable.

Return ONLY valid JSON using this structure:

{
    "task_table": [
        {
            "id": 1,
            "task": "",
            "owner": "",
            "deadline": "",
            "dependency_context": "",
            "status": "",
            "missing_information": ""
        }
    ]
}
"""


# =========================================================
# 3. Raw Meeting Transcript
# =========================================================

meeting_transcript = """
Meeting: Product Development Weekly Sync
Date: August 31, 2026

Rahul: Okay, let's get started. We have about 45 minutes. First, let's talk about the mobile application update.

Priya: The new login screen is almost done. I've completed the UI, but I'm still waiting for the API changes from the backend team.

Amit: Yeah, about that. The authentication API is mostly ready. There's one issue with the refresh token. We're getting an unexpected 401 response sometimes.

Priya: That's exactly the issue I'm seeing on the mobile side.

Rahul: Okay, Amit, can you take ownership of that and fix the refresh token issue?

Amit: Yes, I'll handle it.

Rahul: Good. We need this resolved before the next release.

Amit: I'll try to get it done by Wednesday.

Rahul: Okay, Wednesday works.

Priya: Once that's fixed, I can connect the mobile application to the API. I should need about two days for integration.

Rahul: Fine. Priya, please take care of the integration after Amit finishes the API fix.

Priya: Sure.

Rahul: What about testing?

Sneha: I've started writing test cases for login, logout, and password reset. I haven't started the API failure scenarios yet.

Rahul: Please add those as well. We need to test expired tokens, invalid tokens, and server errors.

Sneha: Okay. I'll add them.

Rahul: When can you finish the complete test suite?

Sneha: Probably Friday.

Rahul: Let's say Friday afternoon then.

Sneha: Works for me.

Amit: One more thing. The backend database migration hasn't been completed yet. We need to add two fields to the users table.

Rahul: Who is handling the migration?

Amit: I think Sameer was going to do it, but I'm not completely sure.

Rahul: Sameer, are you on the call?

Sameer: Yes, I'm here. I can take care of the migration.

Rahul: Great. Sameer, please update the users table and make sure the migration works on the staging database.

Sameer: Sure.

Rahul: Do we have a deadline for that?

Sameer: I can probably finish it tomorrow.

Rahul: Okay, let's have it done by tomorrow evening.

Sameer: Noted.

Rahul: Moving on to the dashboard redesign.

Priya: The new dashboard design from the design team looks good. We need to implement the charts and the activity timeline.

Rahul: How long will that take?

Priya: The charts are straightforward, but the activity timeline requires a new API endpoint.

Amit: I can create the endpoint.

Rahul: Okay, Amit, create the activity timeline API.

Amit: Sure.

Rahul: When can you have it?

Amit: I'll need to check the existing database structure first. Maybe early next week.

Rahul: Fine, let's target Monday.

Amit: Okay.

Priya: Once the endpoint is ready, I'll implement the timeline component.

Rahul: Good.

Sneha: Do we need automated tests for the dashboard too?

Rahul: Yes, but let's first get the feature working. We'll add automated tests after the initial implementation.

Sneha: Okay.

Rahul: Another issue — customer support reported that some users are receiving duplicate email notifications.

Sameer: I've seen that in the logs. It looks like the notification worker may be retrying jobs even after they've completed successfully.

Rahul: Can we investigate that?

Sameer: Yes, I'll check the logs and identify what's causing the duplicate notifications.

Rahul: Good. Please investigate it and report back.

Sameer: Sure.

Rahul: Do we have any other issues?

Priya: The application is also loading slowly on the analytics page when there are more than ten thousand records.

Rahul: Is that a backend or frontend issue?

Priya: We're not sure yet.

Amit: I can look at the API response time first.

Rahul: Okay, Amit, investigate the analytics page performance and see where the bottleneck is.

Amit: Sure.

Rahul: Do we need a deadline for that?

Amit: I can give an initial analysis by Friday.

Rahul: Good, Friday it is.

Rahul: Before we close, let's summarize.

Amit will fix the refresh token issue by Wednesday, create the activity timeline API by Monday, and investigate analytics page performance by Friday.

Priya will integrate the mobile login API after the authentication fix and implement the activity timeline once the new endpoint is available.

Sneha will complete the login and API failure test cases by Friday afternoon.

Sameer will complete the database migration by tomorrow evening and investigate the duplicate email notification problem.

Priya: Sounds good.

Sameer: Yes.

Amit: All good.

Rahul: Great. Let's meet again next Monday and review the progress.

Meeting ended.
"""


# =========================================================
# 4. Function to Call NVIDIA API
# =========================================================

def call_nvidia(prompt):

    completion = client.chat.completions.create(

        model="nvidia/nemotron-3.5-lightning-30b-a3b",

        messages=[
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": meeting_transcript
            }
        ],

        temperature=0.3,
        top_p=0.9,
        max_tokens=12000,

        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": True
            },
            "reasoning_budget": 8000
        },

        stream=True
    )

    response = ""

    for chunk in completion:

        if not chunk.choices:
            continue

        content = chunk.choices[0].delta.content

        if content is not None:
            response += content

    return response


# =========================================================
# 5. Store All Outputs in a List
# =========================================================

outputs = []


# =========================================================
# STEP 1
# =========================================================

response_1 = call_nvidia(step_1)

outputs.append({
    "step": 1,
    "name": "meeting_discussion",
    "output": response_1
})


# =========================================================
# STEP 2
# =========================================================

# Step 2 should receive Step 1 output
step_2_input = f"""
MEETING DISCUSSION EXTRACTED FROM STEP 1:

{response_1}

Now identify the action items.
"""

completion = client.chat.completions.create(

    model="nvidia/nemotron-3.5-lightning-30b-a3b",

    messages=[
        {
            "role": "system",
            "content": step_2
        },
        {
            "role": "user",
            "content": step_2_input
        }
    ],

    temperature=0.3,
    top_p=0.9,
    max_tokens=12000,

    extra_body={
        "chat_template_kwargs": {
            "enable_thinking": True
        },
        "reasoning_budget": 8000
    },

    stream=True
)

response_2 = ""

for chunk in completion:

    if not chunk.choices:
        continue

    content = chunk.choices[0].delta.content

    if content is not None:
        response_2 += content


outputs.append({
    "step": 2,
    "name": "action_items",
    "output": response_2
})


# =========================================================
# STEP 3
# =========================================================

step_3_input = f"""
ACTION ITEMS EXTRACTED FROM STEP 2:

{response_2}

Now convert these action items into the final structured task table.
"""


completion = client.chat.completions.create(

    model="nvidia/nemotron-3.5-lightning-30b-a3b",

    messages=[
        {
            "role": "system",
            "content": step_3
        },
        {
            "role": "user",
            "content": step_3_input
        }
    ],

    temperature=0.3,
    top_p=0.9,
    max_tokens=12000,

    extra_body={
        "chat_template_kwargs": {
            "enable_thinking": True
        },
        "reasoning_budget": 8000
    },

    stream=True
)

response_3 = ""

for chunk in completion:

    if not chunk.choices:
        continue

    content = chunk.choices[0].delta.content

    if content is not None:
        response_3 += content


outputs.append({
    "step": 3,
    "name": "task_table",
    "output": response_3
})


# =========================================================
# 6. Print All Outputs
# =========================================================

for item in outputs:

    print("\n" + "=" * 70)
    print(f"STEP {item['step']} — {item['name']}")
    print("=" * 70)

    print(item["output"])