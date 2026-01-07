import os

import dotenv
import requests
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from src.reviewbot.agent.workflow import work_agent
from src.reviewbot.agent.workflow.gitlab_notes import post_mr_note
from src.reviewbot.infra.config.env import load_env

dotenv.load_dotenv()

app = FastAPI()


def get_required_env(key: str) -> str:
    """Get required environment variable or raise error with helpful message."""
    value = os.environ.get(key)
    if not value:
        raise ValueError(f"Missing required environment variable: {key}")
    return value


GITLAB_SECRET = get_required_env("GITLAB_WEBHOOK_SECRET")
GITLAB_TOKEN = get_required_env("GITLAB_BOT_TOKEN")
GITLAB_API_V4 = get_required_env("GITLAB_API_V4_URL") + "/api/v4"
BOT_USERNAME = os.environ.get("GITLAB_BOT_USERNAME")
BOT_ID = get_required_env("GITLAB_BOT_AUTHOR_ID")


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
async def gitlab_webhook(req: Request, background_tasks: BackgroundTasks):
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
        # pattern = rf"(?:/review\b.*@{re.escape(BOT_USERNAME)}|@{re.escape(BOT_USERNAME)}.*?/review\b)"
        # if not re.search(pattern, text):
        #     return JSONResponse({"ignored": "no /review command"})

        if text.strip() != "/reviewbot review":
            return JSONResponse({"ignored": "not a review command"})

        mr = payload.get("merge_request")
        if not mr:
            return JSONResponse({"ignored": "not an MR note"})

        project_id = payload["project"]["id"]
        mr_iid = mr["iid"]

        config = load_env()
        background_tasks.add_task(
            work_agent,
            config,
            project_id,
            mr_iid,
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

        if detailed_status not in ["passed", "failed"]:
            return JSONResponse({"ignored": "pipeline is not in a final state"})

        if not mr:
            return JSONResponse({"ignored": "not an MR pipeline"})

        mr_iid = mr["iid"]

        if detailed_status != "passed":
            post_mr_note(
                GITLAB_API_V4,
                GITLAB_TOKEN,
                project_id,
                mr_iid,
                "Pipeline was not successful. If you want ReviewBot to review your changes, please re-run the pipeline, and make sure it passes. Or you can manually call ReviewBot by typing: \n\n /reviewbot review",
            )
            return JSONResponse({"ignored": "pipeline failed"})

        # conditions
        if mr_has_conflicts(mr):
            post_mr_note(
                GITLAB_API_V4,
                GITLAB_TOKEN,
                project_id,
                mr_iid,
                "Merge conflicts present. Please resolve them and commit changes to re-run the pipeline. Or you can manually call ReviewBot by typing: \n\n /reviewbot review",
            )
            return JSONResponse({"ignored": "merge conflicts present"})

        config = load_env()
        background_tasks.add_task(
            work_agent,
            config,
            project_id,
            mr_iid,
        )

        return JSONResponse({"status": "auto review triggered"})

    return JSONResponse({"ignored": f"unsupported event {kind}"})
