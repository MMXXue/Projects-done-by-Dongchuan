import os
import pinecone
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import AutoTokenizer, AutoModel
from text2vec import SentenceModel
import torch

# 加载文档
def load_docs(directory):
  loader = DirectoryLoader(directory)
  documents = loader.load()
  return documents

# 分割文档大小以及设置上下文关联
def split_docs(documents, chunk_size=500, chunk_overlap=300):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)



# pinecone key
index_name = "model"
pinecone.init(
    api_key="ebe939a7-65fc-4378-9093-33c584438832",
    environment="us-west4-gcp-free"
)

# 加载文档
directory = './operator/'

documents = load_docs(directory)
print("文档数量：", len(documents))

docs = split_docs(documents)
print("文档被切割的段数：", len(docs))

# 向量化数据
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
# embeddings = HuggingFaceEmbeddings(model_name='shibing624/text2vec-base-chinese')

# 将数据加载入pinecone
index = pinecone.Index(index_name)
pinecone.whoami()
# index.delete(deleteAll='true', namespace=index_name)

# query = "如何新增广告组和对应品牌？"
query = "你好？"

model = SentenceModel('shibing624/text2vec-base-chinese')
index = Pinecone.from_documents(docs, model,index_name=index_name)
embeddings = model.encode(query)
print(type(embeddings))
print(type([0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]))


# result = index.query(
#   vector=embeddings.tolist(),
#   top_k=3,
#   include_values=True
# )
# print(result)



result = index.similarity_search(embeddings.tolist(), k=5)

# # llm语言模型（用于question answering的）
# tokenizer = AutoTokenizer.from_pretrained("timpal0l/mdeberta-v3-base-squad2")
# model = AutoModelForQuestionAnswering.from_pretrained("timpal0l/mdeberta-v3-base-squad2")
# qa = pipeline(task='question-answering',model=model,tokenizer=tokenizer)

# answers = ()
# for i in result:
#     answer = qa(question=query, context=i.page_content)
#     answers += (answer,)

# max_score_answer = max(answers, key=lambda x: x['score'])
# print(max_score_answer)
# # print(result)


