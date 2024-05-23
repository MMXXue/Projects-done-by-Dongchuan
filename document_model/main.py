import os
from langchain.document_loaders import UnstructuredWordDocumentLoader
from langchain.text_splitter import  RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
# from keys import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_API_ENV
import pinecone
from langchain.vectorstores import Pinecone
import requests
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub, PromptTemplate, LLMChain

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"



directory_path = './operator/'
data = []
for filename in os.listdir(directory_path):
    if filename.endswith(".doc") or filename.endswith(".docx"):
        file_path = os.path.join(directory_path, filename)
        loader = UnstructuredWordDocumentLoader(file_path)
        print(loader)
        data.append(loader.load())
print(len(data))
print(data[0])

# 初始化文本分割器
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
texts = text_splitter.split_documents(data[0])
print(f'documents:{len(texts)}')

# Create embeddings and store them in a FAISS vector store
embedder = HuggingFaceEmbeddings(model_name='GanymedeNil/text2vec-large-chinese')
vector_store = FAISS.from_documents(texts, embedder)


# Load the LLM and create a QA chain 设定语言模型
llm=HuggingFaceHub(repo_id="google/flan-t5-xl", model_kwargs={"temperature":0, "max_length":512})
qa_chain = load_qa_chain(llm, chain_type="stuff", verbose=True)

# Ask a question using the QA chain
question = "中国的首都在哪里"
similar_docs = vector_store.similarity_search(question, k = 1)
response = qa_chain.run(input_documents=similar_docs, question=question, verbose = True)
print(len(response))
print(response)




# from langchain.indexes import VectorstoreIndexCreator
# index = VectorstoreIndexCreator().from_loaders([text_splitter.split_documents(data[0])])

# # 分割文档, chunk_size是指切多大， chunk——overlap是指增加上下文的联系
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=20)
# texts = text_splitter.split_documents(data[0])
#
# print(texts)
#
# embedding = HuggingFaceEmbeddings(model_name='shibing624/text2vec-base-chinese')
# vector_store = FAISS.from_documents(texts, embedding)
#
# # Load the LLM and create a QA chain
# llm=HuggingFaceHub(repo_id="google/flan-t5-xl", model_kwargs={"temperature":0, "max_length":512})
# qa_chain = load_qa_chain(llm, chain_type="stuff")
#
# # Ask a question using the QA chain
# question = "如果这是我第一次用这个系统我该怎么办？"
# similar_docs = vector_store.similarity_search(question)
# response = qa_chain.run(input_documents=similar_docs, question=question)
# print(response)



# # 把文字变成数字形式，做一个相似比较
# # embeddings =OpenAIEmbeddings(openai_api_key = "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")
# embeddings = OpenAIEmbeddings()
# pinecone.init(
#     api_key="27bdbf38-9b4b-413d-b485-aae5e83f9724",
#     environment="us-west4-gcp-free"
# )
#
# index_name = "model"
# for i in range(len(texts)):
#     Pinecone.from_texts([t.page_content for t in texts[i]], embeddings, index_name=index_name)
#     print("done")




