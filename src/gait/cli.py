from __future__ import annotations

import os
import sys
import argparse
import json
import urllib.request
import urllib.error

from pathlib import Path

from .repo import GaitRepo
from .schema import Turn
from .objects import short_oid
from .log import walk_commits
from .tokens import count_turn_tokens
from .remote import RemoteSpec, remote_add, remote_get, push as remote_push, fetch as remote_fetch, pull as remote_pull, clone_into

# ----------------------------
### Ollama API Helpers 
# ---------------------------

def _ollama_url(host: str, path: str) -> str:
    host = host.rstrip("/")
    if not host.startswith("http://") and not host.startswith("https://"):
        host = "http://" + host
    return host + path


def _ollama_json(host: str, path: str, payload: dict | None = None, timeout: float = 60.0) -> dict:
    url = _ollama_url(host, path)
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw) if raw.strip() else {}
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {e.code} {e.reason}: {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cannot reach Ollama at {url}: {e}") from e


def ollama_list_models(host: str) -> list[str]:
    r = _ollama_json(host, "/api/tags")
    models = []
    for m in (r.get("models") or []):
        name = m.get("name")
        if name:
            models.append(name)
    return models


def ollama_chat(
    host: str,
    model: str,
    messages: list[dict],
    *,
    temperature: float | None = None,
    num_predict: int = 200,
) -> str:
    payload: dict = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"num_predict": num_predict},
    }
    if temperature is not None:
        payload["options"]["temperature"] = temperature

    r = _ollama_json(host, "/api/chat", payload=payload, timeout=600.0)
    return ((r.get("message") or {}).get("content")) or ""

def _resolve_commitish(repo: GaitRepo, commitish: str | None) -> str:
    """
    Resolve a commit-ish into a concrete commit id string.
    v0 supports:
      - None / "HEAD" / "@": current HEAD commit
      - any prefix/full hash: passed through (repo.get_commit() can validate later)
    """
    if commitish is None or commitish in ("HEAD", "@"):
        cid = repo.head_commit_id()
        if not cid:
            raise ValueError("HEAD is empty (no commits).")
        return cid
    return commitish.strip()

# ----------------------------
# Remote 
# ---------------------------

def _require_gaithub_token() -> str:
    tok = os.environ.get("GAITHUB_TOKEN", "").strip()
    if not tok:
        raise RuntimeError("Missing GAITHUB_TOKEN env var (PAT-style token for gaithubd).")
    return tok


def cmd_remote_add(args: argparse.Namespace) -> int:
    repo = GaitRepo.discover()
    remote_add(repo, args.name, args.url)
    print(f"remote {args.name} -> {args.url}")
    return 0


def cmd_push(args: argparse.Namespace) -> int:
    repo = GaitRepo.discover()
    token = _require_gaithub_token()
    base_url = remote_get(repo, args.remote)

    spec = RemoteSpec(base_url=base_url, owner=args.owner, repo=args.repo, name=args.remote)
    remote_push(repo, spec, token=token, branch=args.branch)
    print(f"pushed {args.branch or repo.current_branch()} to {args.remote} ({args.owner}/{args.repo})")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    repo = GaitRepo.discover()
    token = _require_gaithub_token()
    base_url = remote_get(repo, args.remote)

    spec = RemoteSpec(base_url=base_url, owner=args.owner, repo=args.repo, name=args.remote)
    heads, mems = remote_fetch(repo, spec, token=token)
    print(f"fetched: heads={len(heads)} memory={len(mems)}")
    return 0


def cmd_pull(args: argparse.Namespace) -> int:
    repo = GaitRepo.discover()
    token = _require_gaithub_token()
    base_url = remote_get(repo, args.remote)

    spec = RemoteSpec(base_url=base_url, owner=args.owner, repo=args.repo, name=args.remote)
    merge_id = remote_pull(
        repo, spec,
        token=token,
        branch=args.branch or repo.current_branch(),
        with_memory=args.with_memory,
    )
    print(f"pulled {args.remote}/{args.branch or repo.current_branch()} -> {repo.current_branch()}")
    print(f"HEAD:   {merge_id}")
    if args.with_memory:
        print(f"memory: {repo.read_memory_ref(repo.current_branch())}")
    return 0


def cmd_clone(args: argparse.Namespace) -> int:
    token = _require_gaithub_token()
    dest = Path(args.path).resolve()

    spec = RemoteSpec(base_url=args.url, owner=args.owner, repo=args.repo, name=args.remote)
    clone_into(dest, spec, token=token, branch=args.branch)

    print(f"cloned {args.owner}/{args.repo} into {dest}")
    return 0

# ----------------------------
# Commands
# ----------------------------

def cmd_init(args: argparse.Namespace) -> int:
    root = Path(args.path).resolve()
    repo = GaitRepo(root=root)
    repo.init()
    print(f"Initialized GAIT repo in {repo.gait_dir}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    repo = GaitRepo.discover()
    branch = repo.current_branch()
    head = repo.head_commit_id()
    print(f"root:   {repo.root}")
    print(f"branch: {branch}")
    print(f"HEAD:   {head or '(empty)'}")
    return 0


def cmd_branch(args: argparse.Namespace) -> int:
    repo = GaitRepo.discover()
    repo.create_branch(
        args.name,
        from_commit=args.from_commit,
        inherit_memory=(not args.no_inherit_memory),
    )
    print(f"Created branch {args.name}")
    return 0


def cmd_checkout(args: argparse.Namespace) -> int:
    repo = GaitRepo.discover()
    repo.checkout(args.name)
    print(f"Switched to branch {args.name}")
    return 0


def cmd_revert(args: argparse.Namespace) -> int:
    repo = GaitRepo.discover()
    branch = repo.current_branch()
    head = repo.head_commit_id()
    if not head:
        raise ValueError("Nothing to revert (branch has no commits).")

    # target commit
    if args.commit is None:
        c = repo.get_commit(head)
        parents = c.get("parents") or []
        if not parents:
            repo.write_ref(branch, "")
            print(f"reverted: {branch} is now empty")
            return 0
        target = parents[0]
    else:
        target = _resolve_commitish(repo, args.commit)

    resolved = repo.reset_branch(target)
    print(f"reverted: {branch} -> {resolved}")
    print(f"HEAD:   {repo.head_commit_id()}")

    if args.also_memory:
        old_mem = repo.read_memory_ref(branch)
        new_mem = repo.reset_memory_to_commit(branch, repo.head_commit_id())
        print(f"memory: {old_mem} -> {new_mem}")

    return 0


def cmd_record_turn(args: argparse.Namespace) -> int:
    repo = GaitRepo.discover()

    context = json.loads(args.context) if args.context else {}
    tools = json.loads(args.tools) if args.tools else {}
    model = json.loads(args.model) if args.model else {}

    turn = Turn.v0(
        user_text=args.user,
        assistant_text=args.assistant,
        context=context,
        tools=tools,
        model=model,
        visibility=args.visibility,
    )
    turn_id, commit_id = repo.record_turn(turn, message=args.message or "")
    print(f"turn:   {turn_id}")
    print(f"commit: {commit_id}")
    print(f"branch: {repo.current_branch()} -> {commit_id}")
    return 0


def cmd_log(args: argparse.Namespace) -> int:
    repo = GaitRepo.discover()
    for c in walk_commits(repo, limit=args.limit):
        cid = c["_id"]
        msg = c.get("message") or ""
        kind = c.get("kind") or ""
        created = c.get("created_at") or ""
        turn_ids = c.get("turn_ids") or []

        parents = c.get("parents") or []
        p = ",".join(short_oid(x) for x in parents) if parents else "-"
        merge_flag = " (merge)" if len(parents) > 1 else ""

        print(f"{short_oid(cid)}{merge_flag}  {created}  {kind}  p=[{p}]  turns={len(turn_ids)}  {msg}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    repo = GaitRepo.discover()
    commit_id = _resolve_commitish(repo, args.commit)

    commit = repo.get_commit(commit_id)
    print(f"commit: {commit_id}")
    print(f"branch: {commit.get('branch')}")
    print(f"kind:   {commit.get('kind')}")
    print("-" * 60)

    turn_ids = commit.get("turn_ids") or []
    if not turn_ids:
        print("(no turns attached to this commit)")
        return 0

    for i, tid in enumerate(turn_ids, 1):
        turn = repo.get_turn(tid)
        user = (turn.get("user") or {}).get("text", "")
        assistant = (turn.get("assistant") or {}).get("text", "")

        print(f"[Turn {i}]")
        print("User:")
        print(user)
        print("\nAssistant:")
        print(assistant)
        print("-" * 60)

    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    repo = GaitRepo.discover()
    merge_id = repo.merge(
        args.source,
        message=args.message or "",
        with_memory=args.with_memory,
    )
    print(f"merged: {args.source} -> {repo.current_branch()}")
    print(f"HEAD:   {merge_id}")
    if args.with_memory:
        print(f"memory: {repo.read_memory_ref(repo.current_branch())}")
    return 0


def cmd_pin(args: argparse.Namespace) -> int:
    repo = GaitRepo.discover()

    def find_last_turn_commit() -> str:
        head = repo.head_commit_id()
        if not head:
            raise ValueError("No HEAD commit to pin.")
        cid = head
        seen = set()
        while cid and cid not in seen:
            seen.add(cid)
            c = repo.get_commit(cid)
            if (c.get("turn_ids") or []):
                return cid
            parents = c.get("parents") or []
            cid = parents[0] if parents else ""
        raise ValueError("No commit with turns found in history to pin.")

    if args.last:
        commit_id = find_last_turn_commit()
    else:
        if not args.commit:
            raise ValueError("Provide a commit id/prefix or use --last.")
        commit_id = args.commit

    mem_id = repo.pin_commit(commit_id, note=args.note or "")
    print(f"pinned commit {commit_id} into memory")
    print(f"memory: {mem_id}")
    return 0


def cmd_memory(args: argparse.Namespace) -> int:
    repo = GaitRepo.discover()
    manifest = repo.get_memory()
    print(f"branch: {repo.current_branch()}")
    print(f"pinned: {len(manifest.items)}")
    print("-" * 60)
    for i, it in enumerate(manifest.items, start=1):
        print(f"{i}. turn={short_oid(it.turn_id)} commit={short_oid(it.commit_id)} note={it.note}")
    return 0


def cmd_unpin(args: argparse.Namespace) -> int:
    repo = GaitRepo.discover()
    manifest = repo.get_memory()
    if not manifest.items:
        print("nothing to unpin (memory is empty)")
        return 0

    mem_id = repo.unpin_index(args.index)
    print(f"unpinned #{args.index}")
    print(f"memory: {mem_id}")
    return 0


def cmd_budget(args: argparse.Namespace) -> int:
    repo = GaitRepo.discover()
    b = repo.budget_for_memory()
    print(f"branch: {b['branch']}")
    print(f"pinned_items: {b['pinned_items']}")
    print(f"tokens_input_total: {b['tokens_input_total']}")
    print(f"tokens_output_total: {b['tokens_output_total']}")
    print(f"unknown_token_fields: {b['unknown_token_fields']}")
    return 0


def cmd_context(args: argparse.Namespace) -> int:
    repo = GaitRepo.discover()
    bundle = repo.build_context_bundle(full=args.full)

    if args.json:
        print(json.dumps(bundle, ensure_ascii=False, indent=2))
        return 0

    print(f"branch: {bundle['branch']}")
    print(f"memory: {bundle['memory_id']}")
    print(f"pinned: {bundle['pinned_items']}")
    print("-" * 60)

    if not bundle["items"]:
        print("(no pinned memory)")
        return 0

    for it in bundle["items"]:
        print(f"[PIN {it['index']}] note={it.get('note','')}")
        print("User:")
        print(it.get("user_text", ""))
        print("\nAssistant:")
        print(it.get("assistant_text", ""))
        print("-" * 60)

    return 0


# ----------------------------
# Ollama chat 
# ----------------------------

def cmd_chat(args: argparse.Namespace) -> int:
    repo = GaitRepo.discover()

    host = args.host
    model = args.model

    # If no model provided, pick the first available or fail nicely
    if not model:
        models = ollama_list_models(host)
        if not models:
            raise RuntimeError("No Ollama models found. Run `ollama pull llama3.1` (or similar) and try again.")
        model = models[0]
        print(f"[gait] no --model provided; using: {model}")

    # Build a system message from GAIT memory (optional)
    messages: list[dict] = []
    if not args.no_memory:
        bundle = repo.build_context_bundle(full=False)
        if bundle.get("items"):
            lines = ["You are a helpful assistant. Use the following pinned context from GAIT memory if relevant:"]
            for it in bundle["items"]:
                u = it.get("user_text", "").strip()
                a = it.get("assistant_text", "").strip()
                note = it.get("note", "").strip()
                header = f"- PIN {it['index']}" + (f" ({note})" if note else "")
                lines.append(header)
                if u:
                    lines.append(f"  User: {u}")
                if a:
                    lines.append(f"  Assistant: {a}")
            system_text = "\n".join(lines)
            messages.append({"role": "system", "content": system_text})

    if args.system:
        messages.append({"role": "system", "content": args.system})

    # ----------------------------
    # Resume from HEAD history (replay last N turns)
    # ----------------------------
    do_resume = (not args.no_resume) and (args.resume_turns > 0)
    if do_resume:
        start = _resolve_commitish(repo, args.resume_from)
        turns = repo.iter_turns_from_head(start_commit=start, limit_turns=args.resume_turns)

        for t in turns:
            u = (t.get("user") or {}).get("text", "")
            a = (t.get("assistant") or {}).get("text", "")
            if u:
                messages.append({"role": "user", "content": u})
            if a:
                messages.append({"role": "assistant", "content": a})

        if turns:
            print(f"[gait] resumed {len(turns)} prior turn(s) from history")
            last_u = ((turns[-1].get("user") or {}).get("text", "") or "").strip()
            if last_u:
                print(f"[gait] last user: {last_u[:80]}{'...' if len(last_u) > 80 else ''}")

    print(f"[gait] repo={repo.root} branch={repo.current_branch()} model={model} host={host}")
    print("[gait] enter your questions. Commands: /models /model NAME /pin /revert [/revert COMMIT] /memory /exit")
    print()

    while True:
        try:
            user_text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[gait] bye.")
            return 0

        if not user_text:
            continue

        # --- small text-menu commands ---
        if user_text in ("/exit", "/quit"):
            print("[gait] bye.")
            return 0

        if user_text == "/models":
            ms = ollama_list_models(host)
            if not ms:
                print("[gait] no models found.")
            else:
                for m in ms:
                    print(m)
            continue

        if user_text.startswith("/model "):
            new_model = user_text.split(" ", 1)[1].strip()
            if not new_model:
                print("[gait] usage: /model <name>")
                continue
            model = new_model
            print(f"[gait] model set to: {model}")
            continue

        if user_text == "/memory":
            bundle = repo.build_context_bundle(full=False)
            print(json.dumps(bundle, ensure_ascii=False, indent=2))
            continue

        if user_text.startswith("/revert"):
            parts = user_text.split()
            commit = parts[1] if len(parts) > 1 else None

            # reuse your revert behavior: default parent-of-HEAD
            branch = repo.current_branch()
            head = repo.head_commit_id()
            if not head:
                print("[gait] nothing to revert (empty branch)")
                continue

            if commit is None:
                c = repo.get_commit(head)
                parents = c.get("parents") or []
                if not parents:
                    repo.write_ref(branch, "")
                    print(f"[gait] reverted: {branch} is now empty")
                else:
                    resolved = repo.reset_branch(parents[0])
                    print(f"[gait] reverted: {branch} -> {resolved}")
            else:
                resolved = repo.reset_branch(commit)
                print(f"[gait] reverted: {branch} -> {resolved}")

            if args.also_memory:
                old_mem = repo.read_memory_ref(branch)
                new_mem = repo.reset_memory_to_commit(branch, repo.head_commit_id())
                print(f"[gait] memory: {old_mem} -> {new_mem}")

            continue

        if user_text == "/pin":
            # pin the last commit with turns (skip merges) — same logic as cmd_pin
            head = repo.head_commit_id()
            if not head:
                print("[gait] no HEAD commit to pin.")
                continue

            cid = head
            seen = set()
            found = None
            while cid and cid not in seen:
                seen.add(cid)
                c = repo.get_commit(cid)
                if (c.get("turn_ids") or []):
                    found = cid
                    break
                parents = c.get("parents") or []
                cid = parents[0] if parents else ""

            if not found:
                print("[gait] no commit with turns found to pin.")
                continue

            mem_id = repo.pin_commit(found, note=args.pin_note or "")
            print(f"[gait] pinned {found} into memory")
            print(f"[gait] memory: {mem_id}")
            continue

        # --- normal chat turn ---
        messages.append({"role": "user", "content": user_text})

        try:
            assistant_text = ollama_chat(
                host,
                model,
                messages,
                temperature=args.temperature,
                num_predict=args.num_predict,
            )

        except Exception as e:
            print(f"[gait] ollama error: {e}")
            # remove last user message so next try isn't polluted
            messages.pop()
            continue

        print(f"ai> {assistant_text}\n")
        messages.append({"role": "assistant", "content": assistant_text})

        # record to GAIT
        tokens = count_turn_tokens(
            user_text=user_text,
            assistant_text=assistant_text,
        )

        turn = Turn.v0(
            user_text=user_text,
            assistant_text=assistant_text,
            context={"ollama_host": host},
            tools={},
            model={
                "provider": "ollama",
                "model": model,
            },
            tokens=tokens,
            visibility="private",
        )

        _, commit_id = repo.record_turn(turn, message=args.message or "chat")
        if args.echo_commit:
            print(f"[gait] committed: {short_oid(commit_id)}")

# ----------------------------
# Parser
# ----------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gait")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init", help="Initialize a GAIT repo in PATH (default: .)")
    s.add_argument("path", nargs="?", default=".")
    s.set_defaults(func=cmd_init)

    s = sub.add_parser("status", help="Show current repo status")
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("branch", help="Create a branch")
    s.add_argument("name")
    s.add_argument("--from-commit", default=None)
    s.add_argument("--no-inherit-memory", action="store_true", help="Do not inherit HEAD+ memory from current branch")
    s.set_defaults(func=cmd_branch)

    s = sub.add_parser("checkout", help="Switch branches")
    s.add_argument("name")
    s.set_defaults(func=cmd_checkout)

    s = sub.add_parser("record-turn", help="Record a user+assistant turn and auto-commit")
    s.add_argument("--user", required=True)
    s.add_argument("--assistant", required=True)
    s.add_argument("--message", default="")
    s.add_argument("--visibility", default="private", choices=["private", "shareable"])
    s.add_argument("--context", default="", help="JSON string")
    s.add_argument("--tools", default="", help="JSON string")
    s.add_argument("--model", default="", help="JSON string")
    s.set_defaults(func=cmd_record_turn)

    s = sub.add_parser("log", help="Show commit log")
    s.add_argument("--limit", type=int, default=20)
    s.set_defaults(func=cmd_log)

    s = sub.add_parser("show", help="Show prompts and responses for a commit (default: HEAD)")
    s.add_argument("commit", nargs="?", default=None)
    s.set_defaults(func=cmd_show)

    s = sub.add_parser("pin", help="Pin a commit's turns into branch HEAD+ memory")
    s.add_argument("commit", nargs="?", default=None, help="Commit id/prefix (required unless --last)")
    s.add_argument("--last", action="store_true", help="Pin last commit with turns (skips merges)")
    s.add_argument("--note", default="", help="Optional note for why this was pinned")
    s.set_defaults(func=cmd_pin)

    s = sub.add_parser("memory", help="List pinned HEAD+ memory items for this branch")
    s.set_defaults(func=cmd_memory)

    s = sub.add_parser("unpin", help="Remove a pinned memory item by index (use `gait memory` to see indices)")
    s.add_argument("index", type=int)
    s.set_defaults(func=cmd_unpin)

    s = sub.add_parser("budget", help="Show token budget summary for pinned HEAD+ memory")
    s.set_defaults(func=cmd_budget)

    s = sub.add_parser("merge", help="Merge SOURCE branch into the current branch (creates merge commit)")
    s.add_argument("source")
    s.add_argument("--message", default="")
    s.add_argument("--with-memory", action="store_true", help="Also merge HEAD+ memory (pinned items)")
    s.set_defaults(func=cmd_merge)

    s = sub.add_parser("context", help="Print the branch HEAD+ context pack (from pinned memory)")
    s.add_argument("--json", action="store_true", help="Output JSON (agent/MCP-friendly)")
    s.add_argument("--full", action="store_true", help="Include raw context/tools/model/tokens per turn")
    s.set_defaults(func=cmd_context)

    s = sub.add_parser("revert", help="Rewind current branch HEAD to a prior commit (default: parent of HEAD)")
    s.add_argument("commit", nargs="?", default=None, help="Commit id/prefix (default: first parent of HEAD)")
    s.add_argument("--also-memory", action="store_true", help="Also rewind HEAD+ memory via memory reflog")
    s.set_defaults(func=cmd_revert)

    s = sub.add_parser("chat", help="Interactive local chat with Ollama (records every turn into GAIT)")
    s.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "127.0.0.1:11434"),
                   help="Ollama host (default: 127.0.0.1:11434 or OLLAMA_HOST)")
    s.add_argument("--model", default="", help="Model name (example: llama3.1:latest)")
    s.add_argument("--temperature", type=float, default=None, help="Optional temperature")
    s.add_argument("--system", default="", help="Extra system prompt (optional)")
    s.add_argument("--no-memory", action="store_true", help="Do not inject GAIT pinned memory into system prompt")
    s.add_argument("--also-memory", action="store_true", help="When using /revert, also rewind HEAD+ memory via reflog")
    s.add_argument("--pin-note", default="", help="Optional note used by /pin")
    s.add_argument("--message", default="chat", help="Commit message label for recorded turns")
    s.add_argument("--echo-commit", action="store_true", help="Print short commit id after each recorded turn")
    s.add_argument("--num-predict", type=int, default=200, help="Max tokens to generate per reply")
    s.add_argument("--resume-turns", type=int, default=20,
                   help="Replay last N turns from HEAD history into the chat context (default: 20)")
    s.add_argument("--no-resume", action="store_true",
                   help="Do not replay history (start fresh, only memory/system prompts)")
    s.add_argument("--resume-from", default="HEAD",
                   help="Commitish to resume from (default: HEAD)")
    s.set_defaults(func=cmd_chat)

    # ----------------------------
    # Remote plumbing (v0)
    # ----------------------------

    s = sub.add_parser("remote", help="Manage remotes")
    sub2 = s.add_subparsers(dest="remote_cmd", required=True)

    r = sub2.add_parser("add", help="Add a remote")
    r.add_argument("name")
    r.add_argument("url", help="Base gaithubd URL, e.g. http://127.0.0.1:8787")
    r.set_defaults(func=cmd_remote_add)

    s = sub.add_parser("push", help="Push objects + refs to remote")
    s.add_argument("remote", nargs="?", default="origin")
    s.add_argument("--owner", required=True)
    s.add_argument("--repo", required=True)
    s.add_argument("--branch", default=None)
    s.set_defaults(func=cmd_push)

    s = sub.add_parser("fetch", help="Fetch refs + objects from remote")
    s.add_argument("remote", nargs="?", default="origin")
    s.add_argument("--owner", required=True)
    s.add_argument("--repo", required=True)
    s.set_defaults(func=cmd_fetch)

    s = sub.add_parser("pull", help="Fetch + merge remote tracking branch into current branch")
    s.add_argument("remote", nargs="?", default="origin")
    s.add_argument("--owner", required=True)
    s.add_argument("--repo", required=True)
    s.add_argument("--branch", default=None)
    s.add_argument("--with-memory", action="store_true")
    s.set_defaults(func=cmd_pull)

    s = sub.add_parser("clone", help="Clone a repo from gaithubd")
    s.add_argument("url", help="Base gaithubd URL, e.g. http://127.0.0.1:8787")
    s.add_argument("--owner", required=True)
    s.add_argument("--repo", required=True)
    s.add_argument("--path", required=True)
    s.add_argument("--remote", default="origin")
    s.add_argument("--branch", default="main")
    s.set_defaults(func=cmd_clone)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)
