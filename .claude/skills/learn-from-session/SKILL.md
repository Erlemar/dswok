---
name: learn-from-session
description: Extract patterns and learnings from the current session. Use when asked to learn from this session, capture session learnings, or review what was accomplished. Outputs structured findings for use with /llm-rules-updater.
---


# Learn from Session


## Step 1: Review Session Context


Examine what was accomplished in this session, focusing on:


- Commands run repeatedly or permission prompts that appeared multiple times
- Friction points: wrong approaches, wasted tool calls, repeated corrections
- Knowledge gaps: things looked up that could be pre-loaded as rules
- Patterns that emerged across multiple files or tasks
- Failed approaches: things that didn't work and why (anti-patterns)
- Commands that required manual approval (check if they could be added to `permissions.allow` in `~/.claude/settings.json`)


## Step 2: Identify Improvement Opportunities


Beyond rules updates, check for opportunities across all Claude Code configuration:


| Category | What to look for |
|----------|-----------------|
| **CLAUDE.md** | Universal conventions, workflow preferences, common pitfalls, anti-patterns ("tried X, doesn't work because Y, do Z instead") |
| **Memory topic files** | Domain-specific knowledge → route to the right topic file (check `memory/` dir for existing files; keep MEMORY.md for universal rules and pointers only) |
| **Slash Commands** | Repetitive multi-step tasks (3+ commands), common workflows |
| **Skills** | Domain-specific guidance, code review rules, conventions |
| **Agents** | Specialized workflows, multi-step analysis tasks |
| **Hooks** | Pre-commit checks, build verification, auto-formatting |
| **Permissions** | Commands that required manual approval — check against existing allowlist in `~/.claude/settings.json` under `permissions.allow` |


## Step 3: Output Structured Findings


Present results in this format. Each finding must be concrete — include the specific command, file, or pattern observed.


```markdown
## Session Summary


**What we did:**
- [Brief bullet points of main accomplishments]


**Patterns observed:**
- [Recurring actions, questions, or friction points]


---


## Extracted Learnings


- Ran `arc lint -a` 4 times after editing — add a PostToolUse hook for auto-formatting
- `buck test` failed twice because target was `test_foo` not `foo_test` — add naming convention to CLAUDE.md
- Permission prompt for `jf submit` appeared 3 times — add to allowlist


---


## Improvement Opportunities


### Rules Updates
- [specific rule text to add or change, with target file path]


### Anti-Patterns Discovered
- **[What was tried]** → [Why it failed] → [Do this instead]
 - Target file: [which memory/CLAUDE.md file to add this to]


### Memory Updates
- [specific entry to add/update] → Target: [topic file name]


### Slash Commands
- **`/[name]`** — [use case], Priority: [High/Medium/Low]


### Skills
- **`[name]`** — [domain], Priority: [High/Medium/Low]


### Hooks
- **[name]** — [type], Priority: [High/Medium/Low]


### Permissions
- **`Bash(command:pattern*)`** — [what it does], approved N times (in allowlist: yes/no)
```


## Step 4: Offer to Apply


After presenting findings, ask:
> "Would you like me to apply any of these changes now? I can:
> - Update memory files (MEMORY.md or topic files)
> - Add permissions to settings.json
> - Create/update CLAUDE.md entries"


Apply only the changes the user approves. Do NOT auto-apply without confirmation.
For complex CLAUDE.md rule edits, use `/llm-rules-updater` instead.


## Related Skills


- **`/llm-rules-updater`**: Create or update LLM rules (rules file discovery, format-aware editing, diff submission)
- **`/learn-from-history`**: Extract patterns from past session transcripts
- **`/analyze-diffs`**: Extract patterns from Phabricator diffs
- **`/project-rule-evaluator`**: Audit CLAUDE.md files for quality and structure


## Do Not


- Suggest improvements for one-time tasks
- Be vague ("improve documentation") — always be concrete
- Ignore user's actual workflow patterns
- Suggest skills or commands that duplicate existing ones
- Suggest permissions already in the allowlist
- Route all memory updates to MEMORY.md — use topic files



