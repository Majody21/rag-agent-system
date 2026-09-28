# Demo walkthrough script (2 to 3 minutes)

A backup recording for interviews, in case the hosted demo is asleep or
unreachable. Record the Streamlit app locally (`streamlit run
app/streamlit_app.py`) with a terminal open beside it.

**Before recording:** run `python scripts/ingest.py --src data/sample_docs`,
have `docs/demo_upload/remote_work.md` ready to drag in, and open a
terminal in the repo.

---

**0:00 to 0:20. What it is**

> "This is a retrieval-augmented agent over a company's internal documents.
> The agent runs on Claude, retrieval runs through an MCP server, and every
> answer cites the exact file and page it came from. The sidebar shows the
> four fictional sample documents already indexed."

*Screen:* the app, sidebar visible with the four documents.

**0:20 to 0:50. Ask a question**

> "I'll ask how to reset a password when the MFA device is lost."

*Type:* `How do I reset my password if I lost my MFA device?`

> "The answer is a step-by-step procedure, and each claim has an inline tag
> like `it_password_reset.pdf, page 2`."

**0:50 to 1:15. The citation**

> "Below the answer are chips built from what retrieval actually returned,
> not from what the model wrote. Page 2 here is the page in the PDF the
> passage came from. If retrieval finds nothing, the agent says it couldn't
> find the answer instead of guessing."

*Optional:* ask `What is the parental leave policy?` and show the refusal.

**1:15 to 1:45. The MCP tool call**

> "Under the answer: 'MCP tools called: search_documents.' The agent doesn't
> import any retrieval code. It's an MCP client, and search runs in a
> separate MCP server process."

*Switch to the terminal and run:*

```bash
python scripts/demo_mcp_e2e.py "How do I reset my password if I lost my MFA device?"
```

> "This prints the tools the agent discovered from the server, each MCP call
> with its arguments, and the citations that came back. The same server
> plugs into Claude Desktop with a few lines of config."

**1:45 to 2:20. Upload**

> "Now I'll add a new document."

*Drag* `remote_work.md` *into the sidebar and click* **Ingest uploaded files**.

> "The upload goes through the MCP ingest tool: load, chunk, embed, store.
> It's searchable right away."

*Type:* `How many days a week can I work remotely?`

> "And the answer cites the file I just uploaded."

**2:20 to 2:45. Close**

> "The repo has the architecture diagram, a cost model built from measured
> token usage and current AWS and Anthropic prices, and tests that run the
> full question to MCP tool to cited answer path. Links are on my portfolio
> page."
