from pathlib import Path
from langchain.chains.llm import LLMChain
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
from pydantic import BaseModel, Extra
from langchain.chains.base import Chain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.docstore.document import Document
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.docstore.document import Document
from retriever import VectorStoresRetriever
from langchain.chains import ConversationalRetrievalChain


def _format_chat_history(chat_history: List[Tuple[str, str]]) -> str:
    formatted = ""
    for human, ai in chat_history:
        formatted += f"\nHuman: {human}\nAssistant: {ai}"
    return formatted

class AdvanceConversationalRetrievalChain(Chain, BaseModel):
    question_generator: LLMChain
    vectorstore_selector_chain: LLMChain
    combine_docs_chain: BaseCombineDocumentsChain

    retriever: VectorStoresRetriever

    max_tokens_limit: Optional[int] = None
    output_key: str = "answer"
    return_source_documents: bool = False
    get_chat_history: Optional[Callable[[Tuple[str, str]], str]] = None

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid
        arbitrary_types_allowed = True
        allow_population_by_field_name = True

    @property
    def input_keys(self) -> List[str]:
        """Input keys."""
        return ["question", "chat_history"]

    @property
    def output_keys(self) -> List[str]:
        """Return the output keys.

        :meta private:
        """
        _output_keys = [self.output_key]
        if self.return_source_documents:
            _output_keys = _output_keys + ["source_documents"]
        return _output_keys

    def _trim_docs_to_token_limit(self, docs: List[Document]) -> List[Document]:
        if not self.max_tokens_limit or not isinstance(self.combine_docs_chain, StuffDocumentsChain):
            return docs

        doc_count = len(docs)
        token_counts = [self.combine_docs_chain.llm_chain.llm.get_num_tokens(doc.page_content) for doc in docs]
        total_tokens = sum(token_counts[:doc_count])

        while total_tokens > self.max_tokens_limit:
            doc_count -= 1
            total_tokens -= token_counts[doc_count]

        return docs[:doc_count]

    def _get_docs(self, question: str, inputs: Dict[str, Any]) -> List[Document]:
        docs = self.retriever.get_relevant_documents(question, inputs)
        return self._trim_docs_to_token_limit(docs)

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        question = inputs["question"]
        get_chat_history = self.get_chat_history or _format_chat_history
        chat_history_str = get_chat_history(inputs["chat_history"])

        new_question = question if not chat_history_str else self.question_generator.run(
            question=question, chat_history=chat_history_str
        )

        docs = self._get_docs(new_question, inputs)
        new_inputs = inputs.copy()
        new_inputs["question"] = new_question
        new_inputs["chat_history"] = chat_history_str
        answer, _ = self.combine_docs_chain.combine_docs(docs, **new_inputs)
        
        if self.return_source_documents:
            return {self.output_key: answer, "source_documents": docs}
        else:
            return {self.output_key: answer}

    async def _acall(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        new_inputs = inputs.copy()

        question = inputs["question"]
        get_chat_history = self.get_chat_history or _format_chat_history
        chat_history_str = get_chat_history(inputs["chat_history"])
        new_inputs["chat_history"] = chat_history_str

        new_question = question if not chat_history_str else await self.question_generator.arun(
            question=question, chat_history=chat_history_str
        )
        new_inputs["question"] = new_question
        print('Question:', new_question)

        titles = await self.vectorstore_selector_chain.arun(
            question=new_question
        )
        new_inputs['titles'] = titles
        print('Titles:', titles)

        docs = self._get_docs(new_question, new_inputs)
        print('Doc:', docs[0])

        answer, _ = await self.combine_docs_chain.acombine_docs(docs, **new_inputs)

        if self.return_source_documents:
            return {self.output_key: answer, "source_documents": docs}
        else:
            return {self.output_key: answer}

    def save(self, file_path: Union[Path, str]) -> None:
        if self.get_chat_history:
            raise ValueError("Chain not savable when `get_chat_history` is not None.")
        super().save(file_path)
