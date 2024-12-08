from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from tavily import TavilyClient
from typing import Literal, Sequence, Optional
import json
import os
import http.client


class TavilySearchInput(BaseModel):
    """Input for the Tavily tool."""

    query: str = Field(description="검색 쿼리")


def format_search_result(result: dict, include_raw_content: bool) -> dict:
    """
    Utility functions for formatting search results.

    Args:
        result (dict): 원본 검색 결과
        include_raw_content (bool): 원본 콘텐츠 포함 여부

    Returns:
        dict: 포맷팅된 검색 결과
    """
    return {
        "url": result["url"],
        "content": json.dumps(
            {
                "title": result["title"],
                "content": result["content"],
                "raw": result["raw_content"] if include_raw_content else "",
            },
            ensure_ascii=False,
        ),
    }


class TavilySearch(BaseTool):
    """
    Tool that queries the Tavily Search API and gets back json
    """

    name: str = "tavily_web_search"
    description: str = (
        "A search engine optimized for comprehensive, accurate, and trusted results. "
        "Useful for when you need to answer questions about current events. "
        "Input should be a search query."
    )
    args_schema: type[BaseModel] = TavilySearchInput
    client: TavilyClient = None
    include_domains: list = []
    exclude_domains: list = []
    max_results: int = 3

    def __init__(
        self,
        api_key: Optional[str] = None,
        include_domains: list = [],
        exclude_domains: list = [],
        max_results: int = 3,
    ):
        """
        TavilySearch 클래스의 인스턴스를 초기화합니다.

        Args:
            api_key (str): Tavily API 키
            include_domains (list): 검색에 포함할 도메인 목록
            exclude_domains (list): 검색에서 제외할 도메인 목록
            max_results (int): 기본 검색 결과 수
        """
        super().__init__()
        if api_key is None:
            api_key = os.environ.get("TAVILY_API_KEY", None)

        if api_key is None:
            raise ValueError("Tavily API key is not set.")

        self.client = TavilyClient(api_key=api_key)
        self.include_domains = include_domains
        self.exclude_domains = exclude_domains
        self.max_results = max_results

    def _run(self, query: str) -> str:
        """BaseTool의 _run 메서드 구현"""
        results = self.search(query)
        return json.dumps(results, ensure_ascii=False)

    def search(
        self,
        query: str,
        search_depth: Literal["basic", "advanced"] = "advanced",
        topic: Literal["general", "news"] = "general",
        days: int = 3,
        max_results: int = None,
        include_domains: Sequence[str] = None,
        exclude_domains: Sequence[str] = None,
        include_answer: bool = False,
        include_raw_content: bool = True,
        include_images: bool = False,
        format_output: bool = True,
        **kwargs,
    ) -> list:
        """
        검색을 수행하고 결과를 반환합니다.

        Args:
            query (str): 검색 쿼리
            search_depth (str): 검색 깊이 ("basic" 또는 "advanced")
            topic (str): 검색 주제 ("general" 또는 "news")
            days (int): 검색할 날짜 범위
            max_results (int): 최대 검색 결과 수
            include_domains (list): 검색에 포함할 도메인 목록
            exclude_domains (list): 검색에서 제외할 도메인 목록
            include_answer (bool): 답변 포함 여부
            include_raw_content (bool): 원본 콘텐츠 포함 여부
            include_images (bool): 이미지 포함 여부
            format_output (bool): 결과를 포맷팅할지 여부
            **kwargs: 추가 키워드 인자

        Returns:
            list: 검색 결과 목록
        """
        if max_results is None:
            max_results = self.max_results

        if include_domains is None:
            include_domains = self.include_domains

        if exclude_domains is None:
            exclude_domains = self.exclude_domains

        response = self.client.search(
            query,
            search_depth=search_depth,
            topic=topic,
            days=days,
            max_results=max_results,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            include_answer=include_answer,
            include_raw_content=include_raw_content,
            include_images=include_images,
            **kwargs,
        )

        if format_output:
            search_results = [
                format_search_result(r, include_raw_content)
                for r in response["results"]
            ]
        else:
            search_results = response["results"]

        return search_results

    def get_search_context(
        self,
        query: str,
        search_depth: Literal["basic", "advanced"] = "basic",
        topic: Literal["general", "news"] = "general",
        days: int = 3,
        max_results: int = 5,
        include_domains: Sequence[str] = None,
        exclude_domains: Sequence[str] = None,
        max_tokens: int = 4000,
        format_output: bool = True,
        **kwargs,
    ) -> str:
        """
        검색 쿼리에 대한 컨텍스트를 가져옵니다. 웹사이트에서 관련 콘텐츠만 가져오는 데 유용하며,
        컨텍스트 추출과 제한을 직접 처리할 필요가 없습니다.

        Args:
            query (str): 검색 쿼리
            search_depth (str): 검색 깊이 ("basic" 또는 "advanced")
            topic (str): 검색 주제 ("general" 또는 "news")
            days (int): 검색할 날짜 범위
            max_results (int): 최대 검색 결과 수
            include_domains (list): 검색에 포함할 도메인 목록
            exclude_domains (list): 검색에서 제외할 도메인 목록
            max_tokens (int): 반환할 최대 토큰 수 (openai 토큰 계산 기준). 기본값은 4000입니다.
            format_output (bool): 결과를 포맷팅할지 여부
            **kwargs: 추가 키워드 인자

        Returns:
            str: 컨텍스트 제한까지의 검색 컨텍스트를 포함하는 JSON 문자열
        """
        response = self.client.search(
            query,
            search_depth=search_depth,
            topic=topic,
            max_results=max_results,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            include_answer=False,
            include_raw_content=False,
            include_images=False,
            **kwargs,
        )

        sources = response.get("results", [])
        if format_output:
            context = [
                format_search_result(source, include_raw_content=False)
                for source in sources
            ]
        else:
            context = [
                {
                    "url": source["url"],
                    "content": json.dumps(
                        {"title": source["title"], "content": source["content"]},
                        ensure_ascii=False,
                    ),
                }
                for source in sources
            ]

        # max_tokens 처리 로직은 여기에 구현해야 합니다.
        # 현재는 간단히 모든 컨텍스트를 반환합니다.
        return json.dumps(context, ensure_ascii=False)
    



class SerperApiSearch(BaseModel):
    """Input for the Serper tool."""

    query: str = Field(description="검색 쿼리")


class SerperSearch(BaseTool):
    """
    Tool that queries the Serper API and gets back JSON results.
    """

    name: str = "serper_web_search"
    description: str = (
        "A search engine optimized for comprehensive, accurate, and trusted results. "
        "Useful for when you need to answer questions about current events or get detailed web search results. "
        "Input should be a search query."
    )
    args_schema: type[BaseModel] = SerperApiSearch
    api_key: Optional[str] = None
    max_results: int = 5

    def __init__(self, api_key: Optional[str] = None, max_results: int = 5):
        """
        SerperSearch 클래스 초기화.

        Args:
            api_key (str): Serper API 키
            max_results (int): 기본 검색 결과 수
        """
        super().__init__()
        if api_key is None:
            api_key = os.environ.get("SERPER_API_KEY", None)

        if api_key is None:
            raise ValueError("Serper API key is not set.")

        self.api_key = api_key
        self.max_results = max_results

    def _run(self, query: str) -> str:
        """BaseTool의 _run 메서드 구현."""
        results = self.search(query)
        return json.dumps(results, ensure_ascii=False)

    def search(
        self,
        query: str,
        location: str = "Seoul, Seoul, South Korea",
        gl: str = "kr",
        hl: str = "ko",
        tbs: str = "qdr:d",
        max_results: int = None,
        **kwargs,
    ) -> list:
        """
        Serper API를 사용하여 검색을 수행.

        Args:
            query (str): 검색 쿼리
            location (str): 검색 위치
            gl (str): 국가 코드
            hl (str): 언어 코드
            tbs (str): 시간 기반 검색 필터
            max_results (int): 검색 결과 수 제한

        Returns:
            list: 검색 결과 목록
        """
        if max_results is None:
            max_results = self.max_results

        conn = http.client.HTTPSConnection("google.serper.dev")
        payload = json.dumps(
            {
                "q": query,
                "location": location,
                "gl": gl,
                "hl": hl,
                "tbs": tbs,
            }
        )
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }

        conn.request("POST", "/search", payload, headers)
        res = conn.getresponse()
        data = res.read()
        conn.close()

        try:
            response = json.loads(data.decode("utf-8"))
            return response
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON response from Serper API.")

    def get_formatted_results(
        self,
        query: str,
        max_results: int = 5,
    ) -> list:
        """
        포맷된 검색 결과를 반환.

        Args:
            query (str): 검색 쿼리
            max_results (int): 반환할 최대 검색 결과 수

        Returns:
            list: 포맷된 검색 결과 목록
        """
        raw_results = self.search(query, max_results=max_results)
        formatted_results = []

        if "organic" in raw_results:
            for result in raw_results["organic"][:max_results]:
                formatted_results.append(
                    {
                        "title": result.get("title", ""),
                        "link": result.get("link", ""),
                        "snippet": result.get("snippet", ""),
                        "date": result.get("date", ""),
                    }
                )
        return formatted_results

    def get_search_context(
        self,
        query: str,
        max_results: int = 5,
        **kwargs,
    ) -> str:
        """
        검색 결과를 컨텍스트 형태로 반환.

        Args:
            query (str): 검색 쿼리
            max_results (int): 최대 검색 결과 수

        Returns:
            str: 컨텍스트 JSON 문자열
        """
        raw_results = self.search(query, max_results=max_results, **kwargs)
        if "organic" not in raw_results:
            return json.dumps({"results": []}, ensure_ascii=False)

        context = [
            {
                "url": result.get("link", ""),
                "content": result.get("snippet", ""),
                "title": result.get("title", ""),
            }
            for result in raw_results["organic"][:max_results]
        ]
        return json.dumps(context, ensure_ascii=False)
