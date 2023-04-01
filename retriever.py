from typing import Any, Dict, List
from pydantic import BaseModel, Field, root_validator
from langchain.docstore.document import Document
from langchain.schema import BaseRetriever
from langchain.docstore.document import Document


class VectorStoresRetriever(BaseRetriever, BaseModel):
    vectorstores: List[Dict[str, Any]]
    search_type: str = "similarity"
    search_kwargs: dict = Field(default_factory=dict)

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @root_validator()
    def validate_search_type(cls, values: Dict) -> Dict:
        """Validate search type."""
        if "search_type" in values:
            search_type = values["search_type"]
            if search_type not in ("similarity", "mmr"):
                raise ValueError(f"search_type of {search_type} not allowed.")
        return values

    def get_relevant_documents(self, query: str, inputs: Dict[str, Any]) -> List[Document]:
        """Retrieve relevant documents based on the query from the provided VectorStores."""
        all_docs = []
        for vectorstore in self.vectorstores:
            if vectorstore['title'] in inputs['titles']:
                if self.search_type == "similarity":
                    docs = vectorstore['vectorstore'].similarity_search(query, **self.search_kwargs)
                elif self.search_type == "mmr":
                    docs = vectorstore['vectorstore'].max_marginal_relevance_search(
                        query, **self.search_kwargs
                    )
                else:
                    raise ValueError(f"search_type of {self.search_type} not allowed.")
                all_docs += docs
        return all_docs

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """Async method to retrieve relevant documents. Not implemented."""
        raise NotImplementedError("VectorStoreRetriever does not support async")
