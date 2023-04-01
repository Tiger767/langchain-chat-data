from typing import Any, Dict, List, Tuple
from langchain.callbacks.base import AsyncCallbackManager
from langchain.callbacks.tracers import LangChainTracer
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from conversational_retrievel_chain import AdvanceConversationalRetrievalChain
from retriever import VectorStoresRetriever


def create_prompt_templates(vectorstores: List[Dict[str, Any]]) -> Tuple[ChatPromptTemplate, ChatPromptTemplate, ChatPromptTemplate]:
    """
    Create and return the prompt templates for standalone question, vectorstore selector, and answer.

    Parameters:
        vectorstores: A list of vectorstore dictionaries with metadata.

    Returns:
        A tuple containing the standalone question prompt template, vectorstore selector prompt template, and answer prompt template.
    """
    all_templates = [
        # Contextualize Prompt Templates
        [
            # System Prompt
            ("SystemMessagePromptTemplate", "You are a prompt reformer that takes relvant context from previous messages and turns a prompt into fully contextualized and stand alone prompt."),
            
            # Example 1
            ("HumanMessagePromptTemplate", "Given the following conversation and prompt, rephrase the prompt to be a standalone contextualized prompt. Do not try to respond to the prompt or answer any questions. If the prompt is already standalone or does not relate to the previous conversation, just restate the text following Prompt.\n\nPrevious Messages:\nHuman: I really like to eat at the Fast Food Store near my house.\nAssistant: It's great that you enjoy eating at the Fast Food Store near your house! Fast food restaurants can be a convenient and tasty option for many people. However, it's important to remember that eating fast food regularly can have negative health effects. Fast food is often high in calories, fat, and sodium, and low in important nutrients like fiber and vitamins.\n\nPrompt: It really is the best place to eat."),
            ("AIMessagePromptTemplate", "The Fast Food Store near my house is the best place to eat."),
            
            # Example 2
            ("HumanMessagePromptTemplate", "Given the following conversation and prompt, rephrase the prompt to be a standalone contextualized prompt. Do not try to respond to the prompt or answer any questions. If the prompt is already standalone or does not relate to the previous conversation, just restate the text following Prompt.\n\nPrevious Messages:\nHuman: What do you think is the best way to get to the dollar store when I leave in 15 mintues.\nAssistant: Assuming that you have already determined the location of the nearest dollar store, the quickest way to get there in 15 minutes depends on your transportation options and distance.\n\nIf the dollar store is within walking distance, you could walk or jog there if you are comfortable doing so and have appropriate footwear. Alternatively, you could take a bike if you have one available and it would be faster than walking.\n\nIf you are unable to walk or bike, you could consider using a ride-sharing service such as Uber or Lyft to get there quickly. If you have a car available, driving may also be an option, but keep in mind that traffic and parking may affect your travel time.\n\nTo ensure you arrive on time, make sure to plan your route ahead of time, factor in any traffic or wait times, and have your payment method ready if using a ride-sharing service.\n\nPrompt: I think I will get there with the first option you provided."),
            ("AIMessagePromptTemplate", "I think I will get to the dollar store when I leave in 15 mintues with the first option, walking."),
            
            # Example 3
            ("HumanMessagePromptTemplate", "Given the following conversation and prompt, rephrase the prompt to be a standalone contextualized prompt. Do not try to respond to the prompt or answer any questions. If the prompt is already standalone or does not relate to the previous conversation, just restate the text following Prompt.\n\nPrevious Messages:\nHuman: I like to program in python, do you think it is a good language?\nAssistant: As an AI language model, I don't have opinions or personal preferences. However, Python is a popular and widely used programming language that has a large and supportive community.\n\nPrompt: Is rock climbing a sport?"),
            ("AIMessagePromptTemplate", "Is rock climbing a sport?"),

            # Human Prompt
            ("HumanMessagePromptTemplate", "Given the following conversation and prompt, rephrase the prompt to be a standalone contextualized prompt. Do not try to respond to the prompt or answer any questions. If the prompt is already standalone or does not relate to the previous conversation, just restate the text following Prompt.\n\nPrevious Messages:\n{chat_history}\n\nPrompt: {question}"),
        ],
        # Vectorstore Selector Prompt Template
        [
            # Human Prompt
            ("HumanMessagePromptTemplate", "Given the following descriptions of each topic, which topics are relevant to the prompt. Topics are broad and contain more information than is listened in the descriptions. Only list the topic names that are relevant to the prompt, avoid any other words. If there are no options related, say None. Minimize explanation.\n\n" + "".join(f"{vectorstore['title']}\n{vectorstore['description']}\n\n" for vectorstore in vectorstores) + "Prompt: {question}"),
        ],
        # Answer Prompt Templates
        [
            # System and Human Prompt
            ("SystemMessagePromptTemplate", "Use the following pieces of context to respond to the question at the end. If you don't know an answer, just say that you don't know, don't try to make up an answer."),
            ("HumanMessagePromptTemplate", "Context:\n{context}\n\nPrompt: {question}")
        ]
    ]
    
    prompt_map = {
        'SystemMessagePromptTemplate': SystemMessagePromptTemplate,
        'HumanMessagePromptTemplate': HumanMessagePromptTemplate,
        'AIMessagePromptTemplate': AIMessagePromptTemplate
    }
    return tuple([ChatPromptTemplate.from_messages(
        [prompt_map[prompt_class].from_template(template) for prompt_class, template in templates]
    ) for templates in all_templates])


def create_llm_chains(
    manager: AsyncCallbackManager, question_manager: AsyncCallbackManager, stream_manager: AsyncCallbackManager, standalone_question_prompt: ChatPromptTemplate, vectorstore_selector_prompt: ChatPromptTemplate, answer_prompt: ChatPromptTemplate
) -> Tuple[LLMChain, LLMChain, LLMChain]:
    """
    Create and return the LLMChains for standalone question, vectorstore selector, and answer.

    Parameters:
        manager: The AsyncCallbackManager to manage callbacks.
        question_manager: The AsyncCallbackManager to manage callbacks for the standalone question generator.
        stream_manager: The AsyncCallbackManager to manage callbacks for the answer generator.
        standalone_question_prompt: The standalone question prompt template.
        vectorstore_selector_prompt: The vectorstore selector prompt template.
        answer_prompt: The answer prompt template.

    Returns:
        A tuple containing the LLMChains for standalone question, vectorstore selector, and answer doc chain.
    """
    standalone_prompt_generator_llm = ChatOpenAI(
        temperature=0,
        verbose=True,
        callback_manager=question_manager
    )
    vectorstate_selector = ChatOpenAI(temperature=0, verbose=True)
    chat_llm = ChatOpenAI(
        streaming=True,
        callback_manager=stream_manager,
        verbose=True,
        temperature=0,
    )

    question_generator = LLMChain(
        llm=standalone_prompt_generator_llm, prompt=standalone_question_prompt, callback_manager=manager
    )
    vectorstore_selector = LLMChain(
        llm=vectorstate_selector, prompt=vectorstore_selector_prompt, callback_manager=manager
    )
    doc_chain = load_qa_chain(
        chat_llm, chain_type="stuff", prompt=answer_prompt, callback_manager=manager
    )

    return question_generator, vectorstore_selector, doc_chain

def get_chain(
    vectorstores: List[Dict[str, Any]], question_handler, stream_handler, tracing: bool = False
):
    manager = AsyncCallbackManager([])
    question_manager = AsyncCallbackManager([question_handler])
    stream_manager = AsyncCallbackManager([stream_handler])
    if tracing:
        tracer = LangChainTracer()
        tracer.load_default_session()
        manager.add_handler(tracer)
        question_manager.add_handler(tracer)
        stream_manager.add_handler(tracer)

    # Create prompts
    standalone_question_prompt, vectorstore_selector_prompt, answer_prompt = create_prompt_templates(vectorstores)

    # Create LLMChains
    question_generator, vectorstore_selector, doc_chain = create_llm_chains(
        manager, question_manager, stream_manager, standalone_question_prompt, vectorstore_selector_prompt, answer_prompt
    )

    # Create Retriver for multiple vectorstores
    retriever = VectorStoresRetriever(vectorstores=vectorstores, search_type="similarity", search_kwargs={"k": 4})

    # Create Advance QA Chain
    qa = AdvanceConversationalRetrievalChain(
        retriever=retriever,
        question_generator=question_generator,
        vectorstore_selector_chain=vectorstore_selector,
        combine_docs_chain=doc_chain,
        callback_manager=manager,
    )

    return qa
