"""Integration test for GitLab discussion creation with predefined issues.

This test actually calls the GitLab API to create discussions.
Requires the following environment variables to be set:
- GITLAB_API_V4_URL: GitLab API base URL
- GITLAB_BOT_TOKEN: GitLab API token
"""

import os
from pathlib import Path

import pytest

from reviewbot.agent.workflow import GitLabConfig, handle_file_issues
from reviewbot.context import Context, store_manager_ctx
from reviewbot.core.issues.issue import Issue, IssueSeverity
from reviewbot.infra.embeddings.store_manager import CodebaseStoreManager
from reviewbot.infra.git.clone import clone_repo_persistent, get_repo_name
from reviewbot.infra.git.repo_tree import tree
from reviewbot.infra.gitlab.clone import build_clone_url
from reviewbot.infra.gitlab.diff import fetch_mr_diffs, get_mr_branch
from reviewbot.infra.issues.in_memory_issue_store import InMemoryIssueStore


@pytest.fixture
def gitlab_config():
    """Create a GitLab configuration from environment variables."""
    api_v4 = os.getenv("GITLAB_API_V4_URL")
    token = os.getenv("GITLAB_BOT_TOKEN")
    project_id = "29"
    mr_iid = "5"

    if not all([api_v4, token, project_id, mr_iid]):
        pytest.skip(
            "GitLab credentials not set. "
            "Set GITLAB_API_V4_URL, GITLAB_BOT_TOKEN, GITLAB_TEST_PROJECT_ID, and GITLAB_TEST_MR_IID"
        )

    # Type narrowing: we know these are not None after the check above
    assert api_v4 is not None
    assert token is not None
    assert project_id is not None
    assert mr_iid is not None
    api_v4 = api_v4 + "/api/v4"
    return GitLabConfig(
        api_v4=api_v4,
        token=token,
        project_id=project_id,
        mr_iid=mr_iid,
    )


@pytest.fixture
def sample_issues():
    """Create a list of predefined issues for testing."""
    return [
        Issue(
            title="[TEST] Potential null pointer exception",
            description="The variable 'user' might be None when accessing user.name. This is a test issue.",
            file_path="api/feature_testing/mdsl/mdsl.go",
            start_line=30,
            end_line=30,
            severity=IssueSeverity.HIGH,
            status="open",
        ),
    ]


def test_handle_file_issues_creates_discussion(
    gitlab_config: GitLabConfig,
    sample_issues: list[Issue],
):
    """Integration test that actually creates a GitLab discussion with predefined issues."""
    # Set up GitLab API connection
    api_v4 = gitlab_config.api_v4
    token = gitlab_config.token
    project_id = gitlab_config.project_id
    mr_iid = gitlab_config.mr_iid

    # Clone the repository
    clone_url = build_clone_url(api_v4, project_id, token)
    diffs, diff_refs = fetch_mr_diffs(api_v4, project_id, mr_iid, token)
    branch = get_mr_branch(api_v4, project_id, mr_iid, token)
    repo_path = clone_repo_persistent(clone_url, branch=branch)
    repo_path = Path(repo_path).resolve()
    repo_tree = tree(repo_path)

    # Initialize store manager
    manager = CodebaseStoreManager()
    manager.set_repo_root(repo_path)
    manager.set_repo_name(get_repo_name(repo_path))
    manager.set_tree(repo_tree)
    manager.set_diffs(diffs)
    manager.get_store()

    # Set up context for tools (needed for read_file.invoke())
    issue_store = InMemoryIssueStore()
    token_ctx = store_manager_ctx.set(
        Context(store_manager=manager, issue_store=issue_store)
    )

    try:
        # Call the function - this will make real API calls to GitLab
        handle_file_issues(
            "api/feature_testing/mdsl/mdsl.go",
            sample_issues,
            gitlab_config,
            diffs,
            diff_refs,
        )

        # If we get here without an exception, the discussion was created successfully
        # You can verify by checking the merge request in GitLab
    finally:
        # Clean up context
        store_manager_ctx.reset(token_ctx)


def test_code_block_in_markdown_between_line_numbers(
    gitlab_config: GitLabConfig,
):
    """Test that code blocks in markdown correctly extract code from diff between line numbers."""
    from unittest.mock import patch

    # Set up GitLab API connection
    api_v4 = gitlab_config.api_v4
    token = gitlab_config.token
    project_id = gitlab_config.project_id
    mr_iid = gitlab_config.mr_iid

    # Clone the repository
    clone_url = build_clone_url(api_v4, project_id, token)
    diffs, diff_refs = fetch_mr_diffs(api_v4, project_id, mr_iid, token)
    branch = get_mr_branch(api_v4, project_id, mr_iid, token)
    repo_path = clone_repo_persistent(clone_url, branch=branch)
    repo_path = Path(repo_path).resolve()
    repo_tree = tree(repo_path)

    # Initialize store manager
    manager = CodebaseStoreManager()
    manager.set_repo_root(repo_path)
    manager.set_repo_name(get_repo_name(repo_path))
    manager.set_tree(repo_tree)
    manager.set_diffs(diffs)
    manager.get_store()

    # Set up context for tools
    issue_store = InMemoryIssueStore()
    token_ctx = store_manager_ctx.set(
        Context(store_manager=manager, issue_store=issue_store)
    )

    # Find a file that actually has changes in the diff
    if not diffs:
        pytest.skip("No diffs found in merge request")

    # Use the first file with a diff
    # get the first .go file
    go_files = [
        file for file in diffs if file.new_path and file.new_path.endswith(".go")
    ]
    if not go_files:
        pytest.skip("No .go files found in diff")
    file_diff = go_files[1]
    test_file_path = file_diff.new_path or file_diff.old_path
    if not test_file_path:
        pytest.skip("No valid file path in diff")

    # Find a reasonable line range in the diff
    # Look for the first hunk to get line numbers
    import re

    hunk_header_re = re.compile(r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@")
    patch_lines = file_diff.patch.splitlines()

    line_start = None
    line_end = None

    for line in patch_lines:
        match = hunk_header_re.match(line)
        if match:
            new_start = int(match.group(3))
            new_count = int(match.group(4)) if match.group(4) else 1
            line_start = new_start
            line_end = min(
                new_start + new_count - 1, new_start + 10
            )  # Limit to reasonable range
            break

    if line_start is None or line_end is None:
        pytest.skip("Could not find valid line range in diff")

    issue_with_line_range = Issue(
        title="[TEST] Code block extraction test",
        description="Testing that code blocks correctly extract code from diff between line numbers.",
        file_path=test_file_path,
        start_line=line_start,
        end_line=line_end,
        severity=IssueSeverity.MEDIUM,
        status="open",
    )

    # Verify the diff contains content for our line range
    # We'll check that the diff has content in the expected range
    patch_lines = file_diff.patch.splitlines()
    has_content_in_range = False
    for line in patch_lines:
        if line.startswith(("+", "-", " ")) and len(line) > 1:
            has_content_in_range = True
            break

    if not has_content_in_range:
        pytest.skip("Diff does not contain sufficient content for testing")

    try:
        # Mock the GitLab API calls to capture the reply body
        with (
            patch(
                "reviewbot.agent.workflow.create_discussion"
            ) as mock_create_discussion,
            patch(
                "reviewbot.agent.workflow.reply_to_discussion"
            ) as mock_reply_to_discussion,
        ):
            mock_create_discussion.return_value = "discussion-123"

            # Call the function
            handle_file_issues(
                test_file_path, [issue_with_line_range], gitlab_config, diffs, diff_refs
            )

            # Get the reply body that was sent
            call_args = mock_reply_to_discussion.call_args
            reply_body = call_args.kwargs["body"]

            # Verify the code block uses diff syntax
            assert "```diff" in reply_body, (
                "Code block should use diff syntax highlighting"
            )

            # Extract the code block content
            code_block_start = reply_body.find("```diff")
            code_block_end = reply_body.find("```", code_block_start + 7)
            assert code_block_end != -1, "Code block should be properly closed"

            # Get the content between the code block markers (skip the ```diff part)
            code_block_content = reply_body[
                code_block_start + 7 : code_block_end
            ].strip()

            # Verify the code block contains diff markers
            assert (
                "+" in code_block_content
                or "-" in code_block_content
                or " " in code_block_content
            ), "Code block should contain diff markers (+, -, or space)"

            # Verify the code block contains code from the diff
            # It should have some actual code content (not just empty)
            assert len(code_block_content.strip()) > 0, (
                "Code block should contain code extracted from diff"
            )

            # Verify it contains lines that could be from the diff
            # (either added lines with +, removed with -, or context with space)
            assert any(
                line.startswith(("+", "-", " "))
                for line in code_block_content.splitlines()
                if line.strip()
            ), "Code block should contain diff-formatted lines"

            # Verify the line info is correct in the markdown
            assert (
                f"Line {line_start}-{line_end}" in reply_body
                or f"Line {line_end}-{line_start}" in reply_body
            ), "Line range should be displayed correctly"

            # Write to a markdown file for inspection
            with open("reply_body.md", "w") as f:
                f.write(reply_body)

    finally:
        # Clean up context
        store_manager_ctx.reset(token_ctx)
