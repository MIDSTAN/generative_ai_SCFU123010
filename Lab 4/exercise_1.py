import os
import json
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("NVIDIA_API_KEY")

from openai import OpenAI

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = api_key
)
    
job_profile="""
Job Role: Machine Learning Engineer

Job Description:
A Machine Learning Engineer is responsible for designing, developing, training, evaluating, and deploying machine learning models. The role requires strong knowledge of Python, machine learning algorithms, deep learning, data preprocessing, model evaluation, and deployment.

Required Skills:
- Python programming
- NumPy, Pandas, Matplotlib
- Scikit-learn
- Machine Learning algorithms
- Deep Learning
- Neural Networks
- CNN, RNN, LSTM, Transformers
- PyTorch or TensorFlow
- Natural Language Processing
- Computer Vision
- SQL
- Data preprocessing and feature engineering
- Model evaluation and optimization
- Git and GitHub
- REST APIs
- Docker
- Basic cloud computing
- MLOps fundamentals
- Model deployment

Preferred Qualifications:
- Bachelor's degree in Computer Science or related field
- Hands-on machine learning projects
- Experience working with real-world datasets
- Knowledge of Generative AI and Large Language Models
- Understanding of model deployment and production workflows
- Good problem-solving and analytical skills
"""

user_profile="""
Name: Pruthviraj Santosh Sapate

Education:
- B.Tech in Computer Science and Engineering
- Vishwa Prayag University, Solapur, Maharashtra

Technical Skills:
- Python
- C
- C++
- JavaScript
- React.js
- Django
- SQL
- MySQL
- MongoDB
- NumPy
- Pandas
- Scikit-learn
- PyTorch
- TensorFlow
- CNN
- Vision Transformers
- Generative AI
- GANs
- Natural Language Processing
- Computer Vision
- Git and GitHub
- REST APIs

Machine Learning / AI Experience:
- Working on deepfake detection
- Researching cross-modal deepfake detection using image and video data
- Exploring GAN-based approaches for deepfake and defect detection
- Interested in using GAN discriminators as classifiers
- Working with CNN and Vision Transformer architectures
- Experience with prompt engineering and LLM APIs
- Experience using NVIDIA AI APIs and OpenAI-compatible APIs

Projects:
1. Cross-Modal Deepfake Detection
   - Detection of manipulated image and video content
   - CNN and Vision Transformer based architecture
   - Exploring GAN-based discriminator approaches
   - Focus on improving detection of generated and manipulated content

2. GAN-Based Defect Detection
   - Exploring the use of GAN generators to create difficult synthetic samples
   - Using discriminator feedback for classification and detection
   - Investigating adaptive adversarial training

3. Prompt Engineering
   - Building prompts for LLM-based applications
   - Working with structured outputs and API-based LLM systems
   - Experience with NVIDIA AI APIs

4. Full-Stack Applications
   - React.js frontend development
   - Django backend development
   - MySQL and MongoDB database integration
   - REST API development

Current Strengths:
- Python programming
- Machine Learning fundamentals
- Deep Learning
- Computer Vision
- GANs
- CNNs
- Vision Transformers
- Generative AI
- Full-stack development
- Problem solving

Areas to Improve:
- Advanced Machine Learning
- Advanced Deep Learning
- MLOps
- Docker
- Cloud platforms
- Model deployment
- Production ML systems
- Advanced mathematics for ML
- Distributed model training
- System design for ML applications
- Advanced SQL
- Real-world ML engineering practices

Career Goal:
Become a job-ready Machine Learning Engineer with strong practical skills in machine learning, deep learning, computer vision, generative AI, model deployment, and MLOps.
"""
    
prompt = """
role : you are a Job ready roadmap builder expert
context : you have to give roadmap for a particular job role, according to the users linkdin profile
task : study users profile and job profile and give him a complete road map for that job
input data : {user_profile}{job_profile}
output format: Give a flow chart of road map
"""

final_prompt = prompt.format(job_profile=job_profile,user_profile=user_profile)

completion = client.chat.completions.create(
  model="nvidia/nemotron-3.5-lightning-30b-a3b",
  messages=[{"role":"user","content":final_prompt}],
  temperature=1,
  top_p=0.95,
  max_tokens=16384,
  extra_body={"chat_template_kwargs":{"enable_thinking":True},"reasoning_budget":16384},
  stream=True
)
for chunk in completion:
  if not chunk.choices:
    continue
  reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
  if reasoning:
    print(reasoning, end="")
  if chunk.choices[0].delta.content is not None:
    print(chunk.choices[0].delta.content, end="")