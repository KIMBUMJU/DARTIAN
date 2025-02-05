import faiss
from langchain_community.docstore import InMemoryDocstore
from langchain_openai import OpenAIEmbeddings
from testfiles.loader import load_pdf_n_split
from langchain_community.vectorstores import FAISS

# 자연어 쿼리를 벡터화 하는 모델
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# pdf 파일 데이터를 로드하고 나눈 객체들
all_splits = load_pdf_n_split(r"C:\FastAPI_projects\DARTIAN\data\[삼성전자]분기보고서(2024.11.14).pdf")


vector_store = FAISS(
    embedding_function=embeddings,
    index=faiss.IndexFlatL2(len(embeddings.embed_query("hello world"))),
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

documents_ids = vector_store.add_documents(documents=all_splits)

# print(len(documents_ids))
# print(documents_ids[:3])

# # FAISS 인덱스에서 저장된 벡터 확인
# all_vectors = vector_store.index.reconstruct_n(0, vector_store.index.ntotal)
#
# # 저장된 벡터 개수와 일부 벡터 출력
# print(f"Number of vectors stored: {vector_store.index.ntotal}")
# print("First 3 vectors:")
# for i, vector in enumerate(all_vectors[:3]):
#     print(f"Vector {i}:", vector)
#
# results = vector_store.similarity_search(
#     "주요 제품 매출이 뭐야?"
# )

# print()
# print(results[0])
# print()
# print(results[1])

# vector_1 = embeddings.embed_query(all_splits[0].page_content)
# vector_2 = embeddings.embed_query(all_splits[1].page_content)
#
# print(f"Generated vectors of length {len(vector_1)}\n")
# print(f"Generated vectors of length {len(vector_2)}\n")
# print(vector_1[:10])