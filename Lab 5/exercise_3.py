# step 1 extract the candidates answer to a each interview question from a raw transcript 
# step 2 evaluate each answer against a given set of skills flagging any skill that wasnt clearly demonstrated 
# 3. format the evaluation into structured hiring scorecard with an overall recomendation
import os
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Annotated

load_dotenv()


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY")
)

class interview(BaseModel):
    questions: str = Field(description="questions asked by the interviwer")
    answer : str = Field(description="answers answerd by the candidate")

class scorecard(BaseModel):
    skill: str = Field(description="skills required for the job role")
    demonstrated : bool = Field(description="does the candidate has that skill")
    evidence : str = Field(description="evidence of the skill")
    
class formated_scorecard(BaseModel):
    scorecard_rows: str = Field(description="list of scorecard rows")
    overall_recomendation : str = Field(description="recomendation for the candidate")

transcript_parser = PydanticOutputParser(pydantic_object=interview)
scorecard_parser = PydanticOutputParser(pydantic_object=scorecard)
final_scorecard = PydanticOutputParser(pydantic_object=formated_scorecard)
# summary_parser = StrOutputParser()



input_prompt = PromptTemplate(
    template='''You are a helpful assistant.

Extract the interviewer's questions and the candidate's answers from the given raw interview transcript.

Transcript:
{transcript}

{format_instructions}
''',

    input_variables=["transcript"],

    partial_variables={
        "format_instructions": transcript_parser.get_format_instructions()
    }
)
# Dummy review

dummy_interview = """
Interviewer: Good morning. Could you start by introducing yourself?

Candidate: Good morning. My name is Rahul Sharma. I recently completed my Bachelor's degree in Computer Engineering. I have been focusing mainly on Python, machine learning, and generative AI. I have also worked on a few projects involving LangChain and LLM-based applications.

Interviewer: Great. Can you tell me about one of your recent projects?

Candidate: Sure. I worked on a document question-answering system. The idea was to allow users to upload PDF documents and ask questions about their content. I used Python, LangChain, a vector database, and an LLM. I converted the documents into chunks, generated embeddings, stored them in the vector database, and retrieved relevant chunks whenever the user asked a question.

Interviewer: Why did you use a vector database instead of a traditional SQL database?

Candidate: A vector database is useful when we want semantic similarity search. Instead of searching only for exact keywords, we can retrieve documents based on their meaning. For example, if the user asks "How can I reset my password?" the system can retrieve a document containing "Steps for changing your account credentials" even though the exact words are different.

Interviewer: What is RAG?

Candidate: RAG stands for Retrieval-Augmented Generation. The basic idea is that instead of asking the language model to answer entirely from its internal knowledge, we first retrieve relevant information from an external knowledge base and provide that information to the model as context.

Interviewer: Can you explain the complete RAG pipeline?

Candidate: First, we load the documents. Then we split them into smaller chunks. We generate embeddings for those chunks and store them in a vector database. When the user asks a question, we generate an embedding for the question and search for similar chunks. The retrieved chunks are then added to the prompt and sent to the LLM. Finally, the LLM generates the answer based on the retrieved context.

Interviewer: What problems can occur in a RAG system?

Candidate: There can be several problems. The retriever might return irrelevant documents, the chunks might be too large or too small, the embedding model might not work well for the domain, and the LLM might still hallucinate. There can also be problems with latency and the cost of generating embeddings and LLM responses.

Interviewer: How would you reduce hallucinations?

Candidate: I would improve the retrieval quality first because the model needs good context. I would also design the prompt to tell the model to answer only using the provided context. We could also add a confidence threshold and return a message saying that the information was not found if the retrieved documents are not sufficiently relevant.

Interviewer: Let's move to Python. What is the difference between a list and a tuple?

Candidate: Lists are mutable, meaning their elements can be changed after creation. Tuples are immutable. Lists are generally used when we expect the collection to change, while tuples are useful for fixed collections of values.

Interviewer: What is a Python dictionary?

Candidate: A dictionary stores data in key-value pairs. It provides fast lookup based on keys and is commonly used when we want to associate one piece of information with another.

Interviewer: Can you explain object-oriented programming?

Candidate: Object-oriented programming is a programming paradigm based around objects and classes. The major concepts include encapsulation, inheritance, polymorphism, and abstraction.

Interviewer: What is inheritance?

Candidate: Inheritance allows one class to derive properties and methods from another class. It helps with code reuse and allows us to create specialized versions of existing classes.

Interviewer: Now let's talk about machine learning. What is overfitting?

Candidate: Overfitting occurs when a model learns the training data too closely, including its noise and specific patterns, and therefore performs poorly on unseen data.

Interviewer: How can you reduce overfitting?

Candidate: We can use techniques such as regularization, dropout for neural networks, data augmentation, cross-validation, early stopping, and reducing model complexity. Increasing the amount of training data can also help.

Interviewer: What is the difference between supervised and unsupervised learning?

Candidate: In supervised learning, we train the model using labeled data. For example, predicting whether an email is spam or not spam. In unsupervised learning, the data does not have labels, and the algorithm tries to find patterns or structures in the data, such as clustering customers into different groups.

Interviewer: Have you worked with neural networks?

Candidate: Yes. I have worked with basic feed-forward neural networks and convolutional neural networks. I have also studied architectures such as ResNet and transformers.

Interviewer: What is the main advantage of ResNet?

Candidate: ResNet introduced residual or skip connections. These connections help the network learn residual mappings and make it easier to train very deep networks by reducing problems such as vanishing gradients.

Interviewer: What are transformers?

Candidate: Transformers are neural network architectures that rely heavily on attention mechanisms. They were originally introduced for sequence-to-sequence tasks and are now widely used in NLP, computer vision, speech, and generative AI.

Interviewer: Can you explain self-attention at a high level?

Candidate: Self-attention allows each token in a sequence to consider other tokens when creating its representation. The model calculates relationships between tokens using queries, keys, and values. This allows it to understand contextual relationships between different parts of the input.

Interviewer: Why are transformers so important for generative AI?

Candidate: They can process relationships between tokens effectively and can be trained efficiently on large datasets. Models such as GPT use transformer-based architectures to generate text by predicting the next token based on previous context.

Interviewer: Suppose your LLM application is very slow. How would you debug it?

Candidate: First, I would measure latency at each stage instead of assuming the LLM is the problem. I would check document retrieval time, embedding generation time, network latency, LLM inference time, and response processing. Then I would optimize the slowest component. For example, we could cache embeddings, reduce the number of retrieved chunks, use a faster model, or stream the LLM response.

Interviewer: Good. What are your strengths?

Candidate: I would say problem-solving and willingness to learn. When I don't know something, I usually break the problem down and research the underlying concept rather than simply copying a solution.

Interviewer: What is one area you are currently trying to improve?

Candidate: I am trying to improve my understanding of system design and production-level deployment. I have built several prototypes, but I want to get better at designing systems that can handle large numbers of users.

Interviewer: Where do you see yourself in the next three years?

Candidate: I would like to become a strong AI engineer who can build and deploy production-grade machine learning and generative AI systems. I also want to gain experience working with real-world data and large-scale AI applications.

Interviewer: Do you have any questions for me?

Candidate: Yes. What kind of AI projects would I be working on if I joined the team?

Interviewer: The team currently works on recommendation systems, LLM applications, and some computer vision projects. Depending on your background, you could potentially work across multiple areas.

Candidate: That sounds interesting. I would particularly like to work on the LLM and computer vision projects.

Interviewer: Great. That concludes the interview. Thank you for your time.

Candidate: Thank you for the opportunity. It was great speaking with you."""


# Support ticket summary ChatPromptTemplate

step_2 = ChatPromptTemplate.from_template(
    'you have to evaluate each answer against a given set of skills flagging any skill that wasnt clearly demonstrated \n {result}'
)

step_3 = ChatPromptTemplate.from_template(
    'format the evaluation into structured hiring scorecard with an overall recomendation \n {result}'
)


# Create the chain

chain = input_prompt | llm | transcript_parser | step_2 | llm | scorecard_parser | step_3 | llm | final_scorecard


# Invoke the chain

response = chain.invoke({"transcript": dummy_interview})

print(response)
