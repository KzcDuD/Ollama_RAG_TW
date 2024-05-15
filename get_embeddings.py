from datasets import load_dataset

dataset = load_dataset("NTTUNLPTEAM/class-textbook",split='train')
format_func = lambda data: f"id: {data['id']}, attck_id: {data['attck_id']}, 'attck_name{data['attck_name']}', 'description{data['description']}', 'kill_chain_phases{data['kill_chain_phases']}', 'domains{data['domains']}', 'tactic_type{data['tactic_type']}'"

from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_text(format_func(dataset))

#save vectorstore to local
vectorstore = FAISS.from_texts(texts=splits, embedding=OllamaEmbeddings())
vectorstore.save_local("faiss-index")