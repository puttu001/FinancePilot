import os
import inspect
from llama_parse import LlamaParse
from utils.logging import set_logger
from dotenv import load_dotenv
load_dotenv()
logger = set_logger(__name__)


async def _maybe_await(result):
    """Handle APIs that may return either direct values or awaitables."""
    if inspect.isawaitable(result):
        return await result
    return result

class DocumentLoader:
    def __init__(self):
        self.api_key = os.getenv("LLAMA_CLOUD_API_KEY")
        if not self.api_key:
            raise ValueError("LLAMA_CLOUD_API_KEY not found in environment variables")
    
    async def get_premium_pages(self,file_path):
        """Scanning premium files including tables, charts etc"""
        parser = LlamaParse(
            api_key=self.api_key,
            result_type="markdown",
            extract_layout=True
        )
        # json_result = await parser.get_json_result(file_path)
        json_result = await _maybe_await(parser.get_json_result(file_path))

        premium_pages = []
        if not json_result:
            logger.warning(f"LlamaParse returned no results for {file_path}")
            return []
        pages = json_result[0].get("pages",[])

        for page in pages:
            items = page.get("items") or page.get("layout") or []
            if any(item.get("type","").lower() in ["table","chart","figure","infographic"]
                   for item in items):
                premium_pages.append(int(page.get("page")))
        return sorted(premium_pages)
    
    async def load_and_parse(self,file_path):
        #finding pages needing premium parser
        premium_pages_int = await self.get_premium_pages(file_path)
        logger.info(f"Identified premium pages: {premium_pages_int}")
        premium_map = {}

        #standard-parsing for whole document
        std_parser = LlamaParse(
            api_key=self.api_key,
            result_type="markdown",
            split_by_page=True
        )
        all_docs = await _maybe_await(std_parser.load_data(file_path))

        #premium_parsing for premium pages
        if premium_pages_int:
            target_pages_str = ",".join(map(str, [p - 1 for p in premium_pages_int]))

            premium_parser = LlamaParse(
                api_key=self.api_key,
                result_type="markdown",
                premium_mode=True,
                target_pages=target_pages_str,
                split_by_page=True
            )
            premium_docs = await _maybe_await(premium_parser.load_data(file_path))
            # Map the high-quality results back to their 1-based page numbers
            premium_map = {p_num: doc.text for p_num,doc in zip(premium_pages_int,premium_docs)}

        parsed_results = []
        for i,doc in enumerate(all_docs):
            page_num = i+1
            content = premium_map.get(page_num,doc.text)

            parsed_results.append({
                "page": page_num,
                "content": content.strip(),
                "metadata": {
                    "source": os.path.basename(file_path),
                    "is_premium": page_num in premium_map
                }
            })
        return parsed_results
    

# if __name__=="__main__":
#     doc_loader = DocumentLoader()
#     result = doc_loader.load_and_parse("document")
#     print(f"Successfully parsed {len(result)} pages.")