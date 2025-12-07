import os
from dotenv import load_dotenv

load_dotenv()
# from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import TavilySearchResults


os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

def get_web_search_tool():
    
    # This tool is used to search content from the web.
    
    return TavilySearchResults(
            max_results=5,
            include_answer=True,
            include_raw_content=True,
            include_images=True,
            # search_depth="advanced",
            # include_domains = []
            # exclude_domains = []
        )