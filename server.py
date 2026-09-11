"""
server.py — GUI backend. Thin FastAPI wrapper around web_session.py, which
in turn wraps the existing pipeline modules unchanged. No new AI-agent
logic lives here — only HTTP plumbing.

Run with:
    python server.py
or:
    uvicorn server:app --reload
"""
import os
import json
import uuid

from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

import web_session
from web_session import ScannedPdfError, SessionBusyError

app = FastAPI(title="AI CV Feedback Co-Pilot")

os.makedirs(web_session.UPLOAD_DIR, exist_ok=True)


class MessageBody(BaseModel):
    message: str


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/api/upload/init")
async def upload_init(cv: UploadFile, jd_text: str = Form(...)):
    """Just lands the file + JD text on disk and returns a pending_id — fast,
    no parsing. The actual (slow) parsing streams separately over GET at
    /api/upload/{pending_id}/stream, since browser EventSource can't carry a
    file upload itself."""
    if not jd_text.strip():
        raise HTTPException(400, "Job description text is empty.")

    cv_path = os.path.join(web_session.UPLOAD_DIR, f"cv_{uuid.uuid4().hex}_{cv.filename}")
    with open(cv_path, "wb") as f:
        f.write(await cv.read())

    pending_id = web_session.save_pending_upload(cv_path, jd_text, cv.filename)
    return JSONResponse({"pending_id": pending_id})


@app.get("/api/upload/{pending_id}/stream")
def upload_stream(pending_id: str):
    def event_source():
        for event_name, data in web_session.run_upload_stream(pending_id):
            yield _sse(event_name, data)

    return StreamingResponse(event_source(), media_type="text/event-stream")


@app.get("/api/session/{session_id}/analyze/stream")
def analyze_stream(session_id: str):
    try:
        session = web_session.get_session(session_id)
    except KeyError:
        raise HTTPException(404, "Unknown session_id — it may have expired (server restarted).")

    # Checked BEFORE constructing StreamingResponse: once streaming starts,
    # the 200 status is already committed, so this can't become a clean 409
    # after the fact — see web_session.try_acquire_session_lock's docstring.
    if not web_session.try_acquire_session_lock(session_id):
        raise HTTPException(409, "This session is already processing a request.")

    def event_source():
        try:
            for event_name, data in web_session.run_analysis_stream(session):
                yield _sse(event_name, data)
        finally:
            web_session.release_session_lock(session_id)

    return StreamingResponse(event_source(), media_type="text/event-stream")


@app.post("/api/session/{session_id}/message")
def send_message(session_id: str, body: MessageBody):
    try:
        session = web_session.get_session(session_id)
    except KeyError:
        raise HTTPException(404, "Unknown session_id — it may have expired (server restarted).")

    try:
        event = web_session.submit_answer(session, body.message)
    except SessionBusyError:
        raise HTTPException(409, "This session is already processing a message.")
    return JSONResponse(event)


@app.post("/api/session/{session_id}/revised-cv")
async def upload_revised_cv(session_id: str, cv: UploadFile):
    try:
        session = web_session.get_session(session_id)
    except KeyError:
        raise HTTPException(404, "Unknown session_id — it may have expired (server restarted).")

    cv_path = os.path.join(web_session.UPLOAD_DIR, f"revised_{uuid.uuid4().hex}_{cv.filename}")
    with open(cv_path, "wb") as f:
        f.write(await cv.read())

    try:
        web_session.add_revised_cv(session, cv_path)
    except ScannedPdfError as e:
        raise HTTPException(400, str(e))

    return JSONResponse({"ok": True})


@app.get("/api/session/{session_id}/revised-cv/stream")
def revised_cv_stream(session_id: str):
    try:
        session = web_session.get_session(session_id)
    except KeyError:
        raise HTTPException(404, "Unknown session_id — it may have expired (server restarted).")
    if session.revised_cv_data is None:
        raise HTTPException(400, "No revised CV uploaded yet for this session.")

    if not web_session.try_acquire_session_lock(session_id):
        raise HTTPException(409, "This session is already processing a request.")

    def event_source():
        try:
            for event_name, data in web_session.run_revised_analysis_stream(session):
                yield _sse(event_name, data)
        finally:
            web_session.release_session_lock(session_id)

    return StreamingResponse(event_source(), media_type="text/event-stream")


# Mounted last so it doesn't shadow the /api routes above.
app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
