import logging
import pickle
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from langchain.vectorstores import VectorStore

from callback import QuestionGenCallbackHandler, StreamingLLMCallbackHandler
from query_data import get_chain
from schemas import ChatResponse

app = FastAPI()
templates = Jinja2Templates(directory="templates")
vectorstore: Optional[VectorStore] = None


@app.on_event("startup")
async def startup_event():
    logging.info("loading vectorstore")
    global vectorstore
    vectorstore = [
        {'vectorstore': None, 'title': 'Handbook', 'description': 'The handbook has all the rules and guidelines related to Sacramento State. It is only updated once a year. Rules for conduct on campus. Academic calendar. Policies for student organizations. Campus safety guidelines.'},
        {'vectorstore': None, 'title': 'Website', 'description': 'The website has any general info relating to Sacramento State that likely is not in the catalog or handbook. Campus news and events. Information for prospective students. Athletics schedules and scores.'},
        {'vectorstore': None, 'title': 'Catalog', 'description': 'The Catalog has all the class info, major info, and other degree info related to Sacramento State. List of majors and minors. Course descriptions and requirements. General education requirements. Graduation requirements.'},
    ]
    with open("vectorstore_csus_handbook.pkl", "rb") as f:
        vectorstore[0]['vectorstore'] = pickle.load(f)
    with open("vectorstore_csus_website.pkl", "rb") as f:
        vectorstore[1]['vectorstore'] = pickle.load(f)
    with open("vectorstore_csus_catalog.pkl", "rb") as f:
        vectorstore[2]['vectorstore'] = pickle.load(f)

@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    question_handler = QuestionGenCallbackHandler(websocket)
    stream_handler = StreamingLLMCallbackHandler(websocket)
    chat_history = []
    qa_chain = get_chain(vectorstore, question_handler, stream_handler)
    # Use the below line instead of the above line to enable tracing
    # Ensure `langchain-server` is running
    # qa_chain = get_chain(vectorstore, question_handler, stream_handler, tracing=True)

    while True:
        try:
            # Receive and send back the client message
            question = await websocket.receive_text()
            resp = ChatResponse(sender="you", message=question, type="stream")
            await websocket.send_json(resp.dict())

            # Construct a response
            start_resp = ChatResponse(sender="bot", message="", type="start")
            await websocket.send_json(start_resp.dict())

            result = await qa_chain.acall(
                {"question": question, "chat_history": chat_history}
            )
            chat_history.append((question, result["answer"]))

            end_resp = ChatResponse(sender="bot", message="", type="end")
            await websocket.send_json(end_resp.dict())
        except WebSocketDisconnect:
            logging.info("websocket disconnect")
            break
        except Exception as e:
            logging.error(e)
            resp = ChatResponse(
                sender="bot",
                message="Sorry, something went wrong. Try again.",
                type="error",
            )
            await websocket.send_json(resp.dict())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000)
