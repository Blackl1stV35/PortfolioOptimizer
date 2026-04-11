"""
utils/github_commit.py  --  GitHub Auto-Commit Utility (Priority 4)
====================================================================
Automatically commits and pushes changed files (primarily portfolio.yaml)
after dividend processing, trade logging, or What-If Apply actions.

Configuration (choose one):
  1. .streamlit/secrets.toml  →  [github] pat = "ghp_..."  repo_url = "https://..."
  2. .env file                →  GITHUB_PAT=... GITHUB_REPO_URL=...
  3. portfolio.yaml           →  github: pat: "..." repo_url: "..."

The PAT is never written to logs or UI. All failures are caught gracefully
— the UI remains usable even if GitHub operations fail.
"""

import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


def _load_github_config(cfg: dict) -> dict:
    """
    Load PAT and repo URL from (in priority order):
    1. .streamlit/secrets.toml (Streamlit Cloud native)
    2. Environment variables
    3. portfolio.yaml [github] section
    """
    pat      = None
    repo_url = None

    # 1. Streamlit secrets
    try:
        import streamlit as st
        pat      = st.secrets.get("github", {}).get("pat")
        repo_url = st.secrets.get("github", {}).get("repo_url")
    except Exception:
        pass

    # 2. Environment variables
    if not pat:
        pat      = os.environ.get("GITHUB_PAT")
        repo_url = repo_url or os.environ.get("GITHUB_REPO_URL")

    # 3. portfolio.yaml [github] section
    if not pat:
        gh = cfg.get("github", {})
        pat      = gh.get("pat")
        repo_url = repo_url or gh.get("repo_url")

    return {"pat": pat, "repo_url": repo_url}


def _inject_pat(repo_url: str, pat: str) -> str:
    """Inject PAT into HTTPS URL for authentication."""
    if not repo_url or not pat:
        return repo_url or ""
    if "://" in repo_url:
        proto, rest = repo_url.split("://", 1)
        # Remove any existing credentials
        if "@" in rest:
            rest = rest.split("@", 1)[1]
        return f"{proto}://x-token:{pat}@{rest}"
    return repo_url


def commit_and_push(
    repo_root: str,
    files: list[str],
    message: str,
    cfg: dict,
    author_name: str = "PortfolioOptimizer Bot",
    author_email: str = "bot@portfoliooptimizer.local",
) -> dict:
    """
    Stage specific files, commit, and push.

    Parameters
    ----------
    repo_root : absolute path to git repository root
    files     : list of file paths to stage (relative to repo_root)
    message   : commit message
    cfg       : portfolio config dict (for PAT lookup)

    Returns
    -------
    dict with keys: success (bool), message (str), error (str|None)
    """
    gh = _load_github_config(cfg)
    pat      = gh.get("pat")
    repo_url = gh.get("repo_url")

    if not pat:
        return {
            "success": False,
            "message": "GitHub PAT not configured",
            "error":   "Add github.pat to .streamlit/secrets.toml or portfolio.yaml",
        }

    root = Path(repo_root)
    if not (root / ".git").exists():
        return {
            "success": False,
            "message": "Not a git repository",
            "error":   f"No .git directory at {repo_root}",
        }

    def _run(args, **kwargs):
        return subprocess.run(
            args, cwd=str(root), capture_output=True, text=True,
            env={**os.environ,
                 "GIT_AUTHOR_NAME":     author_name,
                 "GIT_AUTHOR_EMAIL":    author_email,
                 "GIT_COMMITTER_NAME":  author_name,
                 "GIT_COMMITTER_EMAIL": author_email},
            **kwargs,
        )

    try:
        # Set remote with PAT
        if repo_url:
            auth_url = _inject_pat(repo_url, pat)
            _run(["git", "remote", "set-url", "origin", auth_url])

        # Stage only the requested files
        for f in files:
            result = _run(["git", "add", f])
            if result.returncode != 0:
                log.warning("git add %s: %s", f, result.stderr)

        # Check if there's anything to commit
        status = _run(["git", "status", "--porcelain"])
        if not status.stdout.strip():
            return {"success": True, "message": "Nothing to commit — files unchanged", "error": None}

        # Commit
        commit_msg = f"[PortfolioOptimizer] {message} — {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        result = _run(["git", "commit", "-m", commit_msg])
        if result.returncode != 0:
            return {"success": False, "message": "Commit failed", "error": result.stderr[:500]}

        # Push
        result = _run(["git", "push"])
        if result.returncode != 0:
            # Try setting upstream
            branch = _run(["git", "branch", "--show-current"]).stdout.strip() or "main"
            result = _run(["git", "push", "--set-upstream", "origin", branch])
            if result.returncode != 0:
                return {"success": False, "message": "Push failed", "error": result.stderr[:500]}

        log.info("GitHub: committed and pushed — %s", commit_msg)
        return {"success": True, "message": f"Committed: {commit_msg}", "error": None}

    except FileNotFoundError:
        return {"success": False, "message": "git not found", "error": "Install git on the server"}
    except Exception as exc:
        return {"success": False, "message": "Unexpected error", "error": str(exc)[:500]}


def auto_commit_portfolio(
    repo_root: str,
    cfg: dict,
    action: str = "update",
    extra_files: Optional[list] = None,
) -> dict:
    """
    Convenience wrapper — commits config/portfolio.yaml + optional extra files.

    action examples: "dividend confirmed", "new trade T005", "What-If applied"
    """
    files = ["config/portfolio.yaml"]
    if extra_files:
        files.extend(extra_files)
    return commit_and_push(
        repo_root=repo_root,
        files=files,
        message=f"portfolio.yaml: {action}",
        cfg=cfg,
    )
