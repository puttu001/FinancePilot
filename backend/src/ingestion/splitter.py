from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from utils.logging import set_logger
from utils.config import get_config
import re
logger = set_logger(__name__)

class DocumentSplitter:
    def __init__(self):
        chunk_config = get_config("chunking")
        self.chunk_size = chunk_config.get("size",1200)
        self.chunk_overlap = chunk_config.get("overlap",200)

        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#","section"),("##","subsection")]
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". "," "]
        )


    def is_junk_chunk(self,text):
        """Enhanced junk detection with financial table preservation"""
        clean = text.strip()
        financial_keywords = r'\b(revenue|profit|assets|liabilities|equity|debt|ebitda|pat|' \
                            r'gnpa|nnpa|nim|roe|roa|crar|tier|aum|expenses|income|balance)\b'
        if len(clean)<40:
            if re.search(financial_keywords, clean.lower()):
                return False
            return True
        if not re.search(r'[A-Za-z]',clean): return True
        if re.fullmatch(r'[\|\-\s\d]+',clean): return True
        if re.fullmatch(r'\|\s*\w+\s*\|',clean) and len(clean) <30: return True
        if re.search(financial_keywords, clean.lower()) and len(clean) >=30:
            return False
        if re.match(r'^(page \d+|annual report|financial statements?)\s*$', clean.lower()):
            return True
        return False


    def parsed_data_to_chunks(self,parsed_results, file_id, user_id:str = "Anonymous"):
        """
        Convert parsed json results to Langchain documents with metadata
        
        Args:
            parsed_results: Parsed json PDF data from LlamaParse
            file_id: Unique file identifier(filename)
            user_id: ID of the user uploading the document (Isolation purpose)
        """


        final_docs = []
        for item in parsed_results:
            page_content = item.get("content", "")
            page_num = item.get("page", -1)
            page_content = re.sub(r'\n(?:page|pg)\s*\d+\s*\n', '\n', page_content, flags=re.IGNORECASE)
            sections = self.header_splitter.split_text(page_content)

            for section in sections:
                chunks = self.text_splitter.split_documents([section])
                for chunk in chunks:
                    if self.is_junk_chunk(chunk.page_content):
                        continue
                    section_name = chunk.metadata.get("section","General")
                    enriched_text = f"Document: {file_id}\nPage Number: {page_num}.\nSection: {section_name}.\n{chunk.page_content}"
                    chunk.page_content = enriched_text
                    chunk.metadata.update({"page": page_num, "file_id": file_id, "user_id": user_id})
                    final_docs.append(chunk)
        return final_docs


        

