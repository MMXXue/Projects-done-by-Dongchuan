from docx import Document
from langchain import LangChain


def doc_to_json(doc_path):
    doc = Document(doc_path)
    knowledge_base = {}

    for para in doc.paragraphs:
        # 假设每个段落是一个问题和答案的对，用":"分隔
        if ':' in para.text:
            question, answer = para.text.split('\n', 1)
            knowledge_base[question.strip()] = answer.strip()

    return knowledge_base


# 创建一个LangChain实例
lc = LangChain()

# 加载本地的知识库
knowledge_base = doc_to_json('operator/Hiya快速操作手册 - For  Publicis.docx')
lc.load_knowledge_base(knowledge_base)

# 输入问题并获取回答
question = input('请输入你的问题：')
answer = lc.answer(question)

# 打印回答
print(answer)