import os
import pinecone
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from text2vec import SentenceModel

def load_docs(directory):
  loader = DirectoryLoader(directory)
  documents = loader.load()
  return documents

def split_docs(documents, chunk_size=500, chunk_overlap=300):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

index_name = "model"
pinecone.init(
    api_key="ebe939a7-65fc-4378-9093-33c584438832",
    environment="us-west4-gcp-free"
)



directory = './operator/'

documents = load_docs(directory)
print("文档数量：", len(documents))

docs = split_docs(documents)
print("文档被切割的段数：", len(docs))


os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
# embeddings = HuggingFaceEmbeddings(model_name='shibing624/text2vec-base-chinese')
embeddings = SentenceModel("shibing624/text2vec-base-chinese")
# embeddings = HuggingFaceEmbeddings(model_name='GanymedeNil/text2vec-large-chinese')

index = pinecone.Index(index_name)
pinecone.whoami()
index.delete(deleteAll='true', namespace=index_name)
index = Pinecone.from_documents(docs, embeddings, index_name=index_name)

# query = "第一次使用该系统，我该怎么办？"
# # query = "上线材料需要准备哪些？"
# result = index.similarity_search(query)
# print(result)
# # result = index.similarity_search(query)
# # print(result)
# print(result[0].page_content)


# template = """Question: 我想找到关于{question}相关的内容

# Please provide a concise answer."""

# query01 = "新增广告组和对应品牌是什么？"
# prompt = PromptTemplate(
#   template=template, 
#   input_variables=["question"]
# )
# template.format(question=query01)






# query = "如何新增广告组和对应品牌？"
query = "如何新增广告主？"
result = index.similarity_search(query)
# print(result)



# # google/flan-t5-xxl

# # 创建一个语言模型
# llm = HuggingFaceHub(repo_id="lmsys/fastchat-t5-3b-v1.0", model_kwargs={"temperature":1, "max_length":150})
# # 创建一个处理链
# chain = load_qa_chain(llm, chain_type="stuff", verbose=True)

# chain = LLMChain(llm=llm, prompt=template)



tokenizer = AutoTokenizer.from_pretrained("timpal0l/mdeberta-v3-base-squad2")
model = AutoModelForQuestionAnswering.from_pretrained("timpal0l/mdeberta-v3-base-squad2")
qa = pipeline(task='question-answering',model=model,tokenizer=tokenizer)

answers = ()
for i in result:
    answer = qa(question=query, context=i.page_content)
    answers += (answer,)

max_score_answer = max(answers, key=lambda x: x['score'])
print(max_score_answer)
print(result)

# answer = qa(question=query,context=result[0].page_content)
# print("----------------------")
# print(result[0].page_content)
# print(answer)


# answer = qa(question=query,context=result[0].page_content)
# print("----------------------")
# print(result[0])
# print(answer)





# 使用处理链来生成一个回答
# answer = chain.run(input_documents=result, question=query)
# print("____________________________________--")
# print(answer)






