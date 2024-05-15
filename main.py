from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import OllamaEmbeddings

new_vector = FAISS.load_local("faiss-index",OllamaEmbeddings(),allow_dangerous_deserialization=True)

retriever = new_vector.as_retriever()

from langchain import hub
prompt = hub.pull("rlm/rag-prompt")
example_messages = prompt.invoke(
    {"context": "filler context", "question": "filler question"}
).to_messages()
example_messages
# print(example_messages[0].content)

from langchain_community.llms.ollama import Ollama

llm = Ollama()
# llm.invoke("test")
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    return "\n\n".join( doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


""" ref
T1651	Cloud Administration Command	Adversaries may abuse cloud management services to execute commands within virtual machines. Resources such as AWS Systems Manager, Azure RunCommand, and Runbooks allow users to remotely run scripts in virtual machines by leveraging installed virtual machine agents.
"""

"""
What are cloud management services, and how can they be abused by adversaries?
How do resources like AWS Systems Manager, Azure RunCommand, and Runbooks allow users to execute commands within virtual machines?
What are the potential risks of adversaries leveraging cloud management services to execute commands?
How can one identify and prevent adversaries from abusing cloud management services for command execution?
Do cloud providers offer security measures to prevent adversaries from abusing these management services?
"""

"What are cloud management services, and how can they be abused by adversaries?"

while True:
    query = input("""
              input query 
              input '0' to quit
              """)
    if query == '0':
        break
    ans = rag_chain.invoke(query)
    print("EN:")
    print(ans)

    import translators as ts
    print("zh-TW:")
    print(ts.translate_text(query_text=ans, translator='google', from_language= 'en', to_language='zh-TW'))