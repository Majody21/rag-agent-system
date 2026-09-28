# Demo walkthrough script (about 2.5 minutes)

Record the live demo in a browser, with a terminal open for the MCP segment.
The password and parental-leave questions were tested on the live app; the
remote-work upload question was tested locally.

## Before you record

- Open https://abdultaboo-rag-agent.streamlit.app 5 minutes early so it is
  awake, then refresh right before recording (a fresh session resets the
  10-question limit).
- Browser zoom 125%, bookmarks bar hidden, other tabs closed.
- Copy `docs/demo_upload/remote_work.md` to your desktop, ready to drag in.
- Terminal in the repo folder, large font, venv active
  (`.venv\Scripts\activate` on Windows, `source .venv/bin/activate` on
  macOS or Linux). Run the MCP command below once beforehand to warm it up.
- Portfolio page open in a second tab: https://abdultaboo.netlify.app/rag-agent/

## Script

**0:00 to 0:20. What it is** (screen: the demo, sidebar visible)

> "This is a retrieval-augmented AI agent over a company's internal
> documents. It runs on Claude, its retrieval tools are served over the Model
> Context Protocol, and every answer cites the exact file and page it came
> from. On the left are four sample documents for a fictional company: an HR
> guide, an expense policy, an IT runbook as a PDF, and an employee roster."

**0:20 to 0:55. Ask a question**

Click the example button **How do I reset my password if I lost my MFA device?**

> "While it works: the agent decides to search, the search runs in a separate
> MCP server, and Claude writes the answer only from what came back."

When the answer appears:

> "It's a step-by-step procedure, and each step has a source tag, like
> it_password_reset.pdf, page 2."

**0:55 to 1:20. The citation and the refusal**

Point at the chips under the answer.

> "These chips come from what retrieval actually returned, not from the
> model's text, so they always point at a real passage. And if the answer
> isn't in the documents, it says so."

Type `What is the parental leave policy?`

> "It searched, found nothing relevant, and told me instead of guessing."

**1:20 to 1:50. The MCP tool call**

Point at the caption **MCP tools called: search_documents**.

> "The agent doesn't import any retrieval code. It's an MCP client. Here's the
> same flow from the command line."

Switch to the terminal and run:

```bash
python scripts/demo_mcp_e2e.py "How do I reset my password if I lost my MFA device?"
```

> "It lists the tools the agent discovered from the MCP server. The agent only
> gets the read tools; ingestion and deletion are held back. Then each MCP call
> with its arguments, the citations that came back, and the answer. The same
> server plugs into Claude Desktop with a few lines of config."

**1:50 to 2:20. Upload**

Back in the browser, drag `remote_work.md` into the sidebar and click
**Ingest uploaded files**.

> "Uploads go through the MCP ingest tool: load, chunk, embed, store. It's
> searchable right away."

Type `How many days a week can I work remotely?`

> "And the answer cites the file I just uploaded."

**2:20 to 2:40. Close** (portfolio tab, scroll past the diagram to the cost table)

> "The project page has the architecture, the design decisions, and a cost
> model built from measured token usage and current AWS and Anthropic
> pricing. The demo itself has rate limits and a hard spend cap. The code and
> tests are on GitHub, linked from here."

## If something goes wrong

- **App is loading or asleep:** wait about 30 seconds for it to wake.
- **"Limit reached":** refresh the page to start a new session.
- **Slow answer (5 to 20 seconds is normal):** keep talking through the
  "while it works" line.

A full take uses about 4 questions (roughly $0.20). The upload stays visible
to other visitors for 30 minutes, which is fine because it is fictional.
