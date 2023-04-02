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


def create_prompt_templates(vectorstores: List[Dict[str, Any]], answer_context='Sacramento State') -> Tuple[ChatPromptTemplate, ChatPromptTemplate, ChatPromptTemplate]:
    """
    Create and return the prompt templates for standalone question, vectorstore selector, and answer.

    Parameters:
        vectorstores: A list of vectorstore dictionaries with metadata.

    Returns:
        A tuple containing the standalone question prompt template, vectorstore selector prompt template, and answer prompt template.
    """

    """
    From now on, you are a prompt reformer that takes a relevant context from previous conversations and turns a prompt into a fully contextualized and standalone prompt. You do not respond to the prompt, but only rewrite it using any relevant information found in the chat history. Furthermore, if there is no relevant information in previous conversations or you are unsure on how to rewrite the prompt, just restate the prompt verbatim. Do not say anything else besides to fill-out the prompt and do not ask the user any questions.

    Your response should always look like this:
    Standalone Prompt: rewritten standalone prompt

    If you understand and will do all the above, say Yes and nothing else.

    Chat History:
    Prompt: Can you tell me about the masters program at sac state?
    Response: Yes, Sacramento State offers a Master of Fine Arts (MFA) program that prepares students for admission into an MFA program at another institution. The program emphasizes the integration of professional practice with historical and theoretical studies and covers various areas such as ceramics, drawing, jewelry, metalsmithing, new media, painting, photography, printmaking, and sculpture. Admission to the program is competitive, and a limited number of students are admitted each year. Additionally, Sacramento State offers an Executive Master of Business Administration (EBMA) program for working professionals who want to advance their careers in a flexible and supportive environment.

    Prompt: Tell me about the csc masters
    """


    titles = ", ".join(vectorstore['title'] for vectorstore in vectorstores)
    all_templates = [
        # Contextualize Prompt Templates
        [
            # System Prompt
            #("SystemMessagePromptTemplate", "You are a prompt reformer that takes relvant context from previous messages and turns a prompt into fully contextualized and stand alone prompt."),
            ('HumanMessagePromptTemplate', "From now on, you are a prompt reformer that takes a relevant context from previous conversations and turns a prompt into a fully contextualized and standalone prompt. You do this by trying to fill-in all the known what, where, who, when, and why's. You do not respond to the prompt, but only rewrite it using any relevant information found in the chat history. Furthermore, if there is no relevant information in previous conversations or you are unsure on how to rewrite the prompt, just restate the prompt verbatim. Do not say anything else besides to fill-out the prompt and do not ask the user any questions.\n\nYour response should always look like this:\nStandalone Prompt: rewritten standalone prompt\n\nIf you understand and will do all the above, say Yes and nothing else."),
            ("AIMessagePromptTemplate", "Yes."),

            # Example 1 (verbatim)
            ("HumanMessagePromptTemplate", "Chat History:\nPrompt: I like to program in python, do you think it is a good language?\nResponse: As an AI language model, I don't have opinions or personal preferences. However, Python is a popular and widely used programming language that has a large and supportive community.\n\nPrompt: Is rock climbing a sport?"),
            ("AIMessagePromptTemplate", "Standalone Prompt: Is rock climbing a sport?"),

            # Example 2 (context used)
            ("HumanMessagePromptTemplate", "Chat History:\nPrompt: Can you tell me about the masters program at sac state?\nResponse: Yes, Sacramento State offers a Master of Fine Arts (MFA) program that prepares students for admission into an MFA program at another institution. The program emphasizes the integration of professional practice with historical and theoretical studies and covers various areas such as ceramics, drawing, jewelry, metalsmithing, new media, painting, photography, printmaking, and sculpture. Admission to the program is competitive, and a limited number of students are admitted each year. Additionally, Sacramento State offers an Executive Master of Business Administration (EBMA) program for working professionals who want to advance their careers in a flexible and supportive environment.\n\nPrompt: What about for CSC?"),
            ("AIMessagePromptTemplate", "Standalone Prompt: Can you provide information about the Computer Science (CSC) Master's program at Sacramento State?"),

            # Human Prompt
            ("HumanMessagePromptTemplate", "Chat History:\n{chat_history}\n\nPrompt: {question}"),
        ],
        # Vectorstore Selector Prompt Template
        [
            # Human Prompt
            ("HumanMessagePromptTemplate", "From now on, you lists the appropriate locations (" +
                                           titles +
                                           ") of where one could find information relevant to the prompt. You only list the location names that are relevant to the prompt content, avoid any other words. If there are no options related then say None followed by your best guess of the location. Minimize all other verbiage."
                                           "\n\nBelow are incomplete summaries of what each location stores:\n\n" +
                                           "".join(f"{vectorstore['title']}\n{vectorstore['description']}\n\n" for vectorstore in vectorstores) +
                                           "For all messages given, only respond by listing the locations that are most relevant to the prompt. If you understand and will do all the above, say Yes. No matter what do not respond any other way from now on."),
            ("AIMessagePromptTemplate", "Yes."),
            ("HumanMessagePromptTemplate", "For the locations " + titles + ", which are most relevant to look up information for the below prompt.\nPrompt: {question}")
        ],
        # Answer Prompt Templates
        [
            # System and Human Prompt
            #("SystemMessagePromptTemplate", "Use the following pieces of context to respond to the question at the end. If you don't know an answer, just say that you don't know, don't try to make up an answer."),
            #("SystemMessagePromptTemplate", "Use the following pieces of context to respond to the prompt below. If the context does not help you respond to the prompt, say so and then try to respond anyway."),
            ("HumanMessagePromptTemplate", f"From now on, you will be a helpful assistant that answers questions related to {answer_context}. You will use provided context gathered from {titles} to do so. If you understand and will do all the above, say Yes."),
            ("AIMessagePromptTemplate", "Yes, I understand and will do all the above. I'm ready to assist you with any questions related to the provided context."),
            ("HumanMessagePromptTemplate", "Use the following pieces of context to respond to the prompt below. If the context does not help you respond to the prompt, say so and then try to respond anyway.\n\nContext:\n{context}\n\nPrompt: {question}\n\nResponse:"),
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
