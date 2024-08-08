import json 
import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema import BaseOutputParser

st.set_page_config(
    page_title="QuizeGPT",
    page_icon='❓',
)

st.title("Quiz GPT")

llm = ChatOpenAI(
    temperature=0.1,
    model='gpt-3.5-turbo-0125',
    streaming=True,
    callbacks=[
        StreamingStdOutCallbackHandler()
    ],
)

class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.strip()
        if text.startswith("```") and text.endswith("```"):
            text = text[3:-3].strip()
            text = text.replace("'''", "").replace("json", "").strip()
            text = text.replace("True", "true").replace("False", "false")
        if not text:
            st.error("Parsed text is empty. Cannot parse empty text as JSON.")
            return {}
        try:
            return json.loads(text)
        
        except json.JSONDecodeError as e:
            st.error(f"JSONDecodeError: {e.msg} at line {e.lineno} column {e.colno} (char {e.pos})")
            st.error(f"Text that caused the error: {text}")
            return {}

output_parser = JsonOutputParser()

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

question_prompt = ChatPromptTemplate.from_messages(
        [
            (
            "system",
                """
                You are a helpful assistant that is role playing as a teacher.
                    
                Based ONLY on the following context make 10 questions to test the user's knowledge about the text.
                
                Each question should have 4 answers, three of them must be incorrect and one should be correct.
                    
                Use (o) to signal the correct answer.
                    
                Question examples:
                    
                Question: What is the color of the ocean?
                Answers: Red|Yellow|Green|Blue(o)
                    
                Question: What is the capital or Georgia?
                Answers: Baku|Tbilisi(o)|Manila|Beirut
                    
                Question: When was Avatar released?
                Answers: 2007|2001|2009(o)|1998
                    
                Question: Who was Julius Caesar?
                Answers: A Roman Emperor(o)|Painter|Actor|Model
                    
                Your turn!
                    
                Context: {context}
            """,
            )
        ]
    )

formatting_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a powerful formatting algorithm.
     
    You format exam questions into JSON format.
    Answers with (o) are the correct ones.
     
    Example Input:
    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)
         
    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut
         
    Question: When was Avatar released?
    Answers: 2007|2001|2009(o)|1998
         
    Question: Who was Julius Caesar?
    Answers: A Roman Emperor(o)|Painter|Actor|Model
    
     
    Example Output:
     
    ```json
    {{ "questions": [
            {{
                "question": "What is the color of the ocean?",
                "answers": [
                        {{
                            "answer": "Red",
                            "correct": false
                        }},
                        {{
                            "answer": "Yellow",
                            "correct": false
                        }},
                        {{
                            "answer": "Green",
                            "correct": false
                        }},
                        {{
                            "answer": "Blue",
                            "correct": true
                        }}
                ]
            }},
                        {{
                "question": "What is the capital or Georgia?",
                "answers": [
                        {{
                            "answer": "Baku",
                            "correct": false
                        }},
                        {{
                            "answer": "Tbilisi",
                            "correct": true
                        }},
                        {{
                            "answer": "Manila",
                            "correct": false
                        }},
                        {{
                            "answer": "Beirut",
                            "correct": false
                        }}
                ]
            }},
                        {{
                "question": "When was Avatar released?",
                "answers": [
                        {{
                            "answer": "2007",
                            "correct": false
                        }},
                        {{
                            "answer": "2001",
                            "correct": false
                        }},
                        {{
                            "answer": "2009",
                            "correct": true
                        }},
                        {{
                            "answer": "1998",
                            "correct": false
                        }}
                ]
            }},
            {{
                "question": "Who was Julius Caesar?",
                "answers": [
                        {{
                            "answer": "A Roman Emperor",
                            "correct": true
                        }},
                        {{
                            "answer": "Painter",
                            "correct": false
                        }},
                        {{
                            "answer": "Actor",
                            "correct": false
                        }},
                        {{
                            "answer": "Model",
                            "correct": false
                        }}
                ]
            }}
        ]
     }}
    ```
    Your turn!
    Questions: {context}
""",
        )
    ]
)

question_chain = {
    "context" : format_docs
} | question_prompt | llm

formatting_chain = formatting_prompt | llm

@st.cache_data(show_spinner="Loading files...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)

    return docs

@st.cache_data(show_spinner="Making quiz...")
def run_quize_chain(_docs, topic):
    chain = {"context": question_chain} | formatting_chain | output_parser
    response = chain.invoke(docs)

    return response

@st.cache_data(show_spinner="Searching Wikipedia")
def search_wiki(term):
    retriever = WikipediaRetriever(top_k_results=1, lang='en')
    docs = retriever.get_relevant_documents(term)
    return docs


with st.sidebar:
    docs=None
    topic=None
    choice = st.selectbox("Choose what you want to use.", (
        "File", "Wikipedia Article"
    ))
    if choice == "File":
        file = st.file_uploader("Upload a .docx, .txt, .pdf file", type=['pdf', 'txt', 'docx'])

        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("Search Wikipedia...")
        if topic:
            docs = search_wiki(topic)
            

if not docs:
    st.markdown(
        """
    Welcome to QuizGPT.
                
    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
                
    Get started by uploading a file or searching on Wikipedia in the sidebar.
    """
    )

else:
    response = run_quize_chain(docs, topic if topic else file.name)
    st.write(response)
    with st.form("questions_form"):
        for question in response['questions']:
            st.write(question['question'])
            value = st.radio("Select Answer", [answer['answer'] for answer in question['answers']], index=None)

            if {'answer' : value, "correct" : True} in question['answers']:
                st.success("Correct!")
            elif value is not None:
                st.error("Wrong!")


        button = st.form_submit_button()

