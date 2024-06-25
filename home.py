import streamlit as st
# from dotenv import load_dotenv
# import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain import OpenAI, LLMChain
from langchain.prompts import StringPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
# from langchain.tools.retriever import create_retriever_tool
from langchain.agents.load_tools import load_tools
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import AgentAction, AgentFinish
from typing import List, Union
import re


HUGGINGFACEHUB_API_TOKEN = st.secrets("HUGGINGFACEHUB_API_TOKEN")
SERPAPI_API_KEY = st.secrets("SERPAPI_API_KEY")

# Streamlit setup
st.title('AI-Powered Document Processor')
with st.chat_message("assistant"):
    st.write(""" Welcome to AI-Powered Document Processor!*

    *Upload PDFs and Ask Questions*

It allows you to upload PDF documents and ask questions. 
Our smart chatbot first searches the uploaded PDF for answers.
If the information isn't found, it seamlessly fetches answers from the web, 
ensuring comprehensive responses tailored to your queries. Experience efficient knowledge 
retrieval !

""")
# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Check if the necessary objects are in session_state
if 'vectordb' not in st.session_state:
    st.session_state.vectordb = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'agent_executor' not in st.session_state:
    st.session_state.agent_executor = None

# File upload
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file is not None:
    st.success("Successfully uploaded the PDF")
    
    # Save uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Load and process the PDF
    loader = PyPDFLoader("temp.pdf")
    pdf_docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(pdf_docs)
    
    # Embed the documents
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
    vectordb = FAISS.from_documents(final_documents, embeddings)
    retriever = vectordb.as_retriever()
    
    # Store in session state
    st.session_state.vectordb = vectordb
    st.session_state.retriever = retriever

# Define tools
def retrieve_documents(input_text):
    return st.session_state.retriever.get_relevant_documents(input_text)

# Load SerpAPI tool with the API key
serpapi_tool = load_tools(["serpapi"], tool_kwargs={"serp_api_key": SERPAPI_API_KEY})[0]

tools = [
    Tool(
        name="Document Retriever",
        func=retrieve_documents,
        description="Retrieve information from the uploaded document."
    ),
    Tool(
        name="Search",
        func=serpapi_tool.run,
        description="Search the web for information."
    )
]

# Define custom prompt template
template = """Answer the following questions as best you can, but speaking as a compassionate and knowledgeable assistant. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do and search in the document first. 
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! Remember to answer as a compassionate and knowledgeable assistant.

Question: {input}
{agent_scratchpad}"""

class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        kwargs["agent_scratchpad"] = thoughts
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)

prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    input_variables=["input", "intermediate_steps"]
)

class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            if "Action: None" in llm_output:
                return AgentFinish(
                    return_values={"output": llm_output.split("Thought:")[-1].strip()},
                    log=llm_output,
                )
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

output_parser = CustomOutputParser()

repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    temperature=0.2,
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
)

llm_chain = LLMChain(llm=llm, prompt=prompt)
tool_names = [tool.name for tool in tools]

agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=tool_names
)

memory = ConversationBufferWindowMemory(k=2)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=memory
)

# Store agent_executor in session state
st.session_state.agent_executor = agent_executor

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if query := st.chat_input("Ask a question about the document or general queries"):
    # Display user message in chat message container
    st.chat_message("user").markdown(query)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})

    response = st.session_state.agent_executor.run(query)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Display query history
# if st.session_state.messages:
#     st.write("Query History:")
#     for i, message in enumerate(st.session_state.messages):
#         role = "User" if message["role"] == "user" else "Assistant"
#         st.write(f"{i+1}. {role}: {message['content']}")
