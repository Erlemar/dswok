---
name: unslop-code
description: Detect and roast AI code slop - redundant, unreadable, or unnecessarily complex code patterns. Focuses on stupid comments, sloppy tests, over-abstraction, and repetitive code that makes codebases painful to maintain.
---

# Unslop Code

Scan code for AI-generated slop and roast it accordingly. No mercy for tutorial comments, vacuous tests, or "enterprise" abstractions for 3-line scripts.

## Workflow

**Step 1: Get the code**
- Uncommitted changes (`git diff` / `git status`)
- Specific files/directories user specifies
- Pasted code

If unclear, ask: "Uncommitted changes or specific files?"

**Step 2: Scan for slop**
Flag instances with: Pattern name, Location, Severity, Why it's slop

**Step 3: Present the roast**
Summary stats + detailed findings with code snippets

**Step 4: Let user pick fixes**
- Fix all
- Fix by severity
- Interactive selection
- Report only (no changes)

## Slop Patterns

### 1. COMMENT SLOP (MAXIMUM PRIORITY)

**What:** 99% of AI-generated comments are garbage. Delete them all.

**The Rule:** If the comment just restates what the code does, DELETE IT.

**Crime 1: The Narrator**
```python
# Retrieve all users from the repository
all_users = user_repository.get_all_users()

# Create a list to store active users
active_users = []

# Iterate over every user
for user in all_users:
    # If user is active...
    if user.is_active:
        # ...add them to the active users list
        active_users.append(user)

# Return the list of active users
return active_users
```
**The roast:** You're explaining that `get_all_users()` gets all users. That loops iterate. That if checks conditions. Every single comment here is pure noise. DELETE ALL OF THEM.

**Crime 2: The TODO Graveyard**
```python
def process_data(data):
    # TODO: Add error handling
    # TODO: Implement caching
    # TODO: Add logging
    # TODO: Optimize performance
    # NOTE: This might need refactoring
    # FIXME: Handle edge cases
    return transform(data)
```
**The roast:** 6 TODO comments, zero actual code improvements. Either DO the thing or DELETE the comment. TODOs are not documentation, they're procrastination made visible.

**Crime 3: The Import Explainer**
```python
# Import the os module for operating system operations
import os

# Import sys module for system-specific parameters
import sys

# Import json module for JSON parsing and serialization
import json
```
**The roast:** We know what `import os` does. It imports os. DELETE IMMEDIATELY.

**What comments to KEEP (very few):**
- Why a non-obvious approach was chosen
- Business logic that's not clear from code alone
- Warnings about gotchas or edge cases
- Links to specs/tickets for context

**Examples of GOOD comments:**
```python
# Using exponential backoff here because the API rate-limits aggressively
retry_with_backoff()

# Batch size of 500 chosen based on testing - higher causes OOM
for batch in chunks(data, 500):
    process(batch)

# DO NOT change this without updating the mobile client (ticket: T123456)
API_VERSION = "v2"
```

**Severity:** MAXIMUM - this is THE #1 sign of AI slop.

### 2. VACUOUS TESTS

**What:** Tests that verify nothing meaningful.

**Crime 1: The Tautology**
```python
result = process()
assert result == OK or result != OK  # Always true
```
This passes if the function returns ANYTHING. You're testing the law of excluded middle.

**Crime 2: The Crash Dummy**
```python
def test_handles_null():
    my_api.process_data(None)
    my_api.process_data({})
    # No assertions - just hoping it doesn't crash
```
WHERE'S THE ASSERTION? You're calling functions and hoping. This isn't testing, it's Russian roulette.

**The roast:** Your tests don't test. Coverage says 95% but actual verification is 0%.

### 3. ABSTRACTION INFLATION

**What:** Creating "enterprise frameworks" for simple scripts.

**Patterns:** Interfaces with one implementation, Service/Repository layers for basic CRUD, Factory for objects with one variant, Builder for 2-3 field objects, DI framework for a 50-line script.

**Example:**
```python
# For a simple S3 upload script
class StorageService(ABC):
    @abstractmethod
    def upload(self, file: str) -> bool: pass

class AwsS3StorageService(StorageService):
    def upload(self, file: str) -> bool: ...

class LocalFsStorageService(StorageService):  # Never used
    def upload(self, file: str) -> bool: ...

class StorageServiceFactory:
    def create_storage_service(self) -> StorageService: ...
```

**The roast:** You've built a microservice architecture for what should be `boto3.upload_file()`.

### 4. CONTEXT-BLIND REINVENTION

**What:** Rewriting existing utils instead of using them.

**Example:**
```python
# Codebase already has:
def send_email(user: User, template_id: str): ...

# AI-generated duplicate:
def notify_user_via_email(user: User, template: str):
    import smtplib
    server = smtplib.SMTP('smtp.gmail.com', 587)
    # ... 30 lines of manual SMTP
```

**The roast:** There's literally a `send_email()` function. You reinvented email.

### 5. CHATBOT BLEED

**What:** Conversational language in code — "I hope this helps!", "Certainly! Here's the implementation", "Let me know if you need anything else".

**The roast:** This is production code, not a chatbot conversation. DELETE immediately.

### 6. CORPORATE JARGON IN CODE

**What:** Marketing speak in technical code.

```python
def leverage_caching_mechanism_to_enhance_performance():
    """Utilizes sophisticated paradigm to facilitate optimization."""

# Better:
def cache_results():
    """Cache results for faster lookups."""
```

**The roast:** "Leverage" means "use". "Facilitate" means "enable". Stop writing like a management consultant.

### 7. DUPLICATION DRIFT

**What:** Same types/functions defined multiple times across files, often with slightly different names.

### 8. INCONSISTENT PARADIGM MASH

**What:** Mixing patterns randomly in the same file — async/await next to callbacks, ORM next to raw SQL, different error handling styles.

### 9. SPEC BLEED

**What:** Prompt/ticket vocabulary leaking into code names — `implement_business_requirement_3_2()`, `UltimateTierFeatureFlagV2AsRequested`.

### 10. SLEEP-BASED TEST WAITS

**What:** Fixed sleeps instead of proper waiting mechanisms. Use `waitForCondition()`, not prayers and timeouts.

## Detection Guidelines

**MAXIMUM SLOP (Definitely AI):**
- Conversational patterns in comments ("I hope this helps!")
- 10+ narrated/paraphrasing comments
- 5+ patterns across multiple categories

**Strong Signal:**
- 5+ unnecessary comments
- 3-4 slop patterns
- Context-blind reinvention

**Moderate:**
- 2-3 unnecessary comments
- 1-2 patterns
- Could be junior dev, could be AI

## Output Format

```
═══════════════════════════════════════
 AI SLOP DETECTION REPORT
═══════════════════════════════════════

Source: [where the code came from]
Total slop found: [X] patterns
Signal: [MAXIMUM / STRONG / MODERATE / WEAK]

Slop Breakdown:
  Comment Slop:             [count] ← PRIORITY
  Vacuous Tests:            [count]
  Abstraction Inflation:    [count]
  Duplication:              [count]
  Other Slop:               [count]

Verdict: [one-line summary]
```

### Detailed Findings

For each finding:
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[#] Pattern: [name]
Severity: [MAXIMUM / Strong / Moderate]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Location: [file:line]

Slop Found:
[code snippet]

The Roast:
[Brutal but educational explanation]

Fix:
[Better version OR just DELETE]
```

## User Options

After showing the report:

```
Found [X] slop patterns. What now?

1. Delete all comment slop (recommended)
2. Fix all MAXIMUM severity patterns
3. Fix all patterns
4. Let me pick which to fix
5. Just show me the report (no edits)

Pick (1-5): _
```

## What This Skill Does NOT Check

This skill focuses ONLY on slop (redundant/unreadable/complex code). It does NOT check security vulnerabilities, performance issues, algorithmic correctness, or general bad practices.

## The Mission

Code slop wastes time. Narrated comments are noise. Vacuous tests give false confidence. Over-abstraction makes simple things complex. This skill roasts it mercilessly and helps you delete it.

**Be brutal. Be specific. Delete the slop. Especially the comments.**
