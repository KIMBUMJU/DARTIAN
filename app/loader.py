from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_pdf_n_split(pdf_path:str):
    """
    PDF 문서를 로드하고 텍스트로 변환.
    """
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )

    all_splits = text_splitter.split_documents(documents)

    return all_splits