import asyncio
import os
import re

import dotenv
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from src.reviewbot.main import post_merge_request_note, work_agent

dotenv.load_dotenv()

app = FastAPI()

GITLAB_SECRET = os.environ.get("GITLAB_WEBHOOK_SECRET")
GITLAB_TOKEN = os.environ.get("GITLAB_BOT_TOKEN")
GITLAB_API_V4 = os.environ.get("GITLAB_API_V4_URL")
BOT_USERNAME = os.environ.get("GITLAB_BOT_USERNAME")  # without @


def post_mr_note(project_id: str, mr_iid: str, body: str):
    url = f"{GITLAB_API_V4.rstrip('/')}/projects/{project_id}/merge_requests/{mr_iid}/notes"
    r = requests.post(
        url,
        headers={"PRIVATE-TOKEN": GITLAB_TOKEN},
        data={"body": body},
        timeout=30,
    )
    r.raise_for_status()


def get_pipeline_status(project_id: str, pipeline_id: int) -> str:
    url = f"{GITLAB_API_V4.rstrip('/')}/projects/{project_id}/pipelines/{pipeline_id}"
    r = requests.get(
        url,
        headers={"PRIVATE-TOKEN": GITLAB_TOKEN},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["status"]


def mr_has_conflicts(mr: dict) -> bool:
    # GitLab MR payload includes this
    return mr.get("detailed_merge_status") == "conflict"


def pipeline_passed(project_id: str, pipeline_id: int) -> bool:
    if not pipeline_id:
        return False

    url = f"{GITLAB_API_V4.rstrip('/')}/projects/{project_id}/pipelines/{pipeline_id}"
    r = requests.get(
        url,
        headers={"PRIVATE-TOKEN": GITLAB_TOKEN},
        timeout=30,
    )
    r.raise_for_status()
    return r.json().get("status") == "success"


@app.post("/webhook")
async def gitlab_webhook(req: Request):
    token = req.headers.get("X-Gitlab-Token")
    if GITLAB_SECRET and token != GITLAB_SECRET:
        raise HTTPException(status_code=403, detail="Invalid token")

    payload = await req.json()
    kind = payload.get("object_kind")

    # -------------------------------------------------
    # CASE 1: /review command in MR comment (always run)
    # -------------------------------------------------
    if kind == "note":
        note = payload["object_attributes"]
        if note.get("action") != "create":
            return JSONResponse({"ignored": "not a new note"})

        # if author id is 83 (the bot) then ignore
        if note.get("author_id") == 83:
            return JSONResponse({"ignored": "bot note"})

        text = note.get("note", "")
        pattern = rf"(?:/review\b.*@{re.escape(BOT_USERNAME)}|@{re.escape(BOT_USERNAME)}.*?/review\b)"
        if not re.search(pattern, text):
            return JSONResponse({"ignored": "no /review command"})

        mr = payload.get("merge_request")
        if not mr:
            return JSONResponse({"ignored": "not an MR note"})

        project_id = payload["project"]["id"]
        mr_iid = mr["iid"]

        await asyncio.to_thread(
            work_agent,
            GITLAB_API_V4,
            project_id,
            mr_iid,
            GITLAB_TOKEN,
        )

        return JSONResponse({"status": "manual review triggered"})

    # -------------------------------------------------
    # CASE 2: MR created / updated → conditional review
    # -------------------------------------------------
    if kind == "pipeline":
        attrs = payload["object_attributes"]
        mr = payload.get("merge_request")
        detailed_status = attrs.get("detailed_status")

        project_id = payload["project"]["id"]
        mr_iid = mr["iid"]

        if detailed_status not in ["passed", "failed"]:
            return JSONResponse({"ignored": "pipeline is not in a final state"})

        if detailed_status != "passed":
            post_merge_request_note(
                GITLAB_API_V4,
                GITLAB_TOKEN,
                project_id,
                mr_iid,
                "Pipeline was not successful. If you want ReviewBot to review your changes, please re-run the pipeline, and make sure it passes. Or you can manually call ReviewBot by typing: \n\n @project_29_bot_5a466f228cb9d019289c41195219f291 /review",
            )
            return JSONResponse({"ignored": "pipeline failed"})

        # conditions
        if mr_has_conflicts(mr):
            post_merge_request_note(
                GITLAB_API_V4,
                GITLAB_TOKEN,
                project_id,
                mr_iid,
                "Merge conflicts present. Please resolve them and commit changes to re-run the pipeline. Or you can manually call ReviewBot by typing: \n\n @project_29_bot_5a466f228cb9d019289c41195219f291 /review",
            )
            return JSONResponse({"ignored": "merge conflicts present"})

        await asyncio.to_thread(
            work_agent,
            GITLAB_API_V4,
            project_id,
            mr_iid,
            GITLAB_TOKEN,
        )

        return JSONResponse({"status": "auto review triggered"})

    return JSONResponse({"ignored": f"unsupported event {kind}"})
