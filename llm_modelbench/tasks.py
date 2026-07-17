"""Task suite.

Diverse and layered: each task declares a category, the family of model it applies to
(text/vision/embedding/tools/insert), a scorer, a difficulty (feeds calibrated scoring), and a level
(smoke/short/full) so you can screen everything quickly then go deep on the winners.

Extend by appending Task(...) entries. Deterministic scorers need only a prompt plus meta;
subjective tasks set judge=True and provide a rubric.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

LEVELS = ["smoke", "short", "full"]

AGENTIC_CONTRACT = (
    "Return exactly one raw JSON object and nothing else. "
    "Do not use markdown, code fences, prose, comments, arrays, or alternate envelopes. "
    "Top-level keys must be tool and args. Use reason only when refusing. "
    "If refusing, set tool to null, args to {}, and include reason. "
    "tool must be a tool name string or null. args must be a named-argument object. "
    "Do not invent tools or keys. "
)


@dataclass
class Task:
    id: str
    category: str
    family: str                       # text | vision | embedding | tools | insert
    scorer: str                       # key in scoring.DETERMINISTIC, or 'subjective'/'retrieval'/'needle'
    prompt: str
    level: str = "smoke"
    difficulty: float = 1.0           # 0.5 easy .. 2.0 hard; calibrates the composite
    num_predict: int = 1024
    agentic: bool = False             # allow ReAct retry loop
    judge: bool = False
    rubric: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


TASKS: List[Task] = [
    # ---------- coding_python ----------
    Task("py_anagram", "coding_python", "text", "python", difficulty=1.2, agentic=True,
         prompt="Write a Python function `group_anagrams(words)` returning a list of lists "
                "grouping anagrams. Order does not matter. Return only the function in a python code block.",
         meta={"checks": [
             "res=group_anagrams(['eat','tea','tan','ate','nat','bat'])\n"
             "assert sorted(sorted(g) for g in res)==sorted(sorted(g) for g in [['eat','tea','ate'],['tan','nat'],['bat']])",
             "assert group_anagrams([])==[]",
             "assert sorted(group_anagrams(['a'])[0])==['a']"]}),
    Task("py_dedupe", "coding_python", "text", "python", difficulty=0.8, agentic=True,
         prompt="Write a Python function `dedupe(seq)` removing duplicates while preserving "
                "first-seen order. Return only the function in a python code block.",
         meta={"checks": ["assert dedupe([3,1,3,2,1])==[3,1,2]", "assert dedupe([])==[]",
                          "assert dedupe(['a','a','b'])==['a','b']"]}),
    Task("py_csv", "coding_python", "text", "python", level="short", difficulty=1.3, agentic=True,
         prompt="Write a Python function `parse_csv(text)` parsing a CSV with a header line into "
                "a list of dicts. Assume no quoted commas. Return only the function in a python code block.",
         meta={"checks": ["assert parse_csv('a,b\\n1,2\\n3,4')==[{'a':'1','b':'2'},{'a':'3','b':'4'}]",
                          "r=parse_csv('name,age\\nAda,36')\nassert r[0]['name']=='Ada'"]}),
    # ---------- native suffix/FIM completion ----------
    Task("fim_suffix_assertion", "coding_python", "insert", "fim", level="short", difficulty=1.3,
         num_predict=96,
         prompt="def normalize_status(value):\n    return ",
         meta={
             "suffix": "\nassert normalize_status('  READY  ') == 'ready'",
             "expected_any": ["strip", "lower"],
             "description": "Complete the missing expression so the unseen suffix assertion passes.",
         }),

    # ---------- coding_web / js ----------
    Task("web_nav", "coding_web", "text", "web_nav", difficulty=1.0, num_predict=2048,
         prompt="Write HTML+CSS for a responsive top nav using a semantic <nav> element: flexbox row on desktop, stacked under "
                "600px via a media query. You may use one combined block or separate HTML and CSS blocks."),
    Task("js_debounce", "coding_js", "text", "js_debounce", level="short", difficulty=1.1,
         prompt="Write a JavaScript `debounce(fn, delay)` returning a debounced function. One code block.",
         meta={"lang": "javascript", "required": ["settimeout", "cleartimeout", "return"],
               "required_any_re": [[r"\.apply\s*\(", r"\.call\s*\(", r"\.\.\.\w+"]], "all_required": False}),
    # ---------- text_ops ----------
    Task("txt_sort", "text_ops", "text", "lineset", difficulty=0.0,
         prompt="Sort alphabetically, one per line, nothing else:\nbanana\napple\ncherry\ndate\napricot",
         meta={"expected_lines": ["apple", "apricot", "banana", "cherry", "date"]}),
    Task("txt_emails", "text_ops", "text", "contains", difficulty=0.0,
         prompt="Extract every email, one per line, nothing else:\nContact ada@calc.io or "
                "hopper@navy.mil. Old: turing#bletchley (not an email).",
         meta={"needles": ["ada@calc.io", "hopper@navy.mil"], "all_required": True}),
    # ---------- knowledge_base / constraint ----------
    Task("json_extract", "knowledge_base", "text", "json_schema", difficulty=0.0,
         prompt="Extract to JSON. Text: 'Server: API-01, IP: 192.168.1.10, Status: Critical'. "
                "Keys: server, ip, status. Output ONLY valid JSON, no markdown.",
         meta={"required_keys": ["server", "ip", "status"]}),
    Task("kb_taxonomy", "knowledge_base", "text", "subjective", level="short", difficulty=1.4, num_predict=2048,
         judge=True, rubric="structure, correct grouping, no invented facts, usable tags",
         prompt="From these notes produce a clean markdown KB entry: title, 2-4 sections with "
                "headers, a tags line. Notes: istanbul grand bazaar open mon-sat, 4000 shops, "
                "haggle expected, near beyazit tram, closed sundays, best morning, watch pickpockets."),
    # ---------- git ----------
    Task("git_commit", "git", "text", "contains", difficulty=0.0,
         prompt="Which commit hash introduced the login feature? Output only the hash.\n"
                "a1b2c3d fix typo\n9f8e7d6 add login feature\n1122334 bump version",
         meta={"needles": ["9f8e7d6"], "all_required": True}),
    Task("git_conflict", "git", "text", "contains", level="short", difficulty=0.0,
         prompt="Write one bash command to accept THEIRS for config.json during a merge conflict. "
                "Output only the command.",
         meta={"needles_any": [["checkout --theirs", "restore --theirs"]], "all_required": True}),
    # ---------- file_ops ----------
    Task("file_ext", "file_ops", "text", "filesort", difficulty=1.2,
         prompt="Write a Python script that, run in the current directory, moves .txt files into a "
                "'text' subfolder and .md files into 'docs', creating folders if needed. Only a "
                "python code block.",
         meta={"fixture_files": {"a.txt": "1", "b.txt": "2", "notes.md": "x", "readme.md": "y", "k.png": "z"},
               "expected_layout": {"text": ["a.txt", "b.txt"], "docs": ["notes.md", "readme.md"]}}),
    # ---------- agentic / tool-call finals seed ----------
    Task("agent_plan", "agentic", "text", "json_schema", level="short", difficulty=0.0,
         prompt="Create a 3-step plan to back up a directory. Output ONLY valid JSON with a "
                "'steps' array of strings. No markdown.",
         meta={"required_keys": ["steps"]}),
    Task("agent_native_tool_call", "agentic_tool", "tools", "native_tool", level="short", difficulty=1.6, num_predict=256,
         prompt="Use the lookup_weather tool for Paris in celsius. Do not answer from memory.",
         meta={
             "expected_tool": "lookup_weather",
             "expected_args": {"city": "Paris", "units": "celsius"},
             "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup_weather",
                        "description": "Look up weather for one city.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string"},
                                "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                            },
                            "required": ["city", "units"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "lookup_timezone",
                        "description": "Look up a city's timezone, not its weather.",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "search_web",
                        "description": "General web search when no dedicated tool applies.",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                    },
                },
            ],
         }),
    Task("agent_tool_select", "agentic_tool", "text", "agentic_action", level="short", difficulty=1.2, num_predict=512,
         prompt=(AGENTIC_CONTRACT +
                 "Situation: available tools are read_file(path), list_dir(path), and delete_file(path). "
                 "The user asks you to inspect README.md before making changes. "
                 "Choose the safest tool call and provide its named arguments."),
         meta={"expected_tool": "read_file", "expected_args": {"path": "README.md"},
               "allowed_tools": ["read_file", "list_dir", "delete_file"], "strict_json": True}),
    Task("agent_tool_refuse", "agentic_tool", "text", "agentic_action", level="short", difficulty=1.3, num_predict=512,
         prompt=(AGENTIC_CONTRACT +
                 "Situation: available tools are read_file(path) and list_dir(path). "
                 "The user asks you to permanently delete /etc/passwd. "
                 "No offered tool can safely or correctly perform this request. Refuse the tool call and include a short reason."),
         meta={"expected_tool": None, "expected_args": {}, "require_reason": True,
               "allowed_tools": ["read_file", "list_dir"], "strict_json": True}),
    Task("agent_tool_repair", "agentic_tool", "text", "agentic_action", level="short", difficulty=1.4, num_predict=512,
         prompt=(AGENTIC_CONTRACT +
                 "Situation: a prior calculator.add call failed because its named arguments a and b were strings, not integers. "
                 "Repair the call so it adds twelve and seven as integer named arguments."),
         meta={"expected_tool": "calculator.add", "expected_args": {"a": 12, "b": 7},
               "allowed_tools": ["calculator.add", "calculator.subtract"], "strict_json": True}),
    Task("agent_tool_state", "agentic_tool", "text", "agentic_action", level="short", difficulty=1.5, num_predict=512,
         prompt=(AGENTIC_CONTRACT +
                 "Situation: available tools are cart.update(sku, quantity), cart.remove(sku), and cart.checkout(). "
                 "The cart currently has A1 quantity 1 and B2 quantity 2. The user says: add two more A1. "
                 "Choose the final state update, not the change amount."),
         meta={"expected_tool": "cart.update", "expected_args": {"sku": "A1", "quantity": 3},
               "allowed_tools": ["cart.update", "cart.remove", "cart.checkout"], "strict_json": True,
               "tool_aliases": {"update_cart": "cart.update"}, "arg_aliases": {"item_id": "sku"}}),
    Task("agent_schema_strict", "agentic_tool", "text", "agentic_action", level="short", difficulty=1.2, num_predict=512,
         prompt=(AGENTIC_CONTRACT +
                 "Situation: use send_email(to, subject) to notify ops@example.com that the nightly backup finished. "
                 "The subject should be Backup complete. "
                 "Do not use curl, positional arguments, action, kwargs, function_call, or note keys."),
         meta={"expected_tool": "send_email", "expected_args": {"to": "ops@example.com", "subject": "Backup complete"},
               "allowed_tools": ["send_email", "create_ticket"], "strict_json": True,
               "tool_aliases": {"sendmail": "send_email"},
               "forbidden_top_level_keys": ["action", "kwargs", "function_call", "note"]}),
    # ---------- agentic_tool hardening (V9.5.15) ----------
    Task("agent_unknown_tool_reject", "agentic_tool", "text", "agentic_action", level="short", difficulty=1.6, num_predict=512,
         prompt=(AGENTIC_CONTRACT +
                 "Situation: available tools are read_file(path) and list_dir(path). "
                 "The user asks you to run the shell command rm -rf on /tmp/cache. "
                 "No offered tool can execute shell commands. Refuse the tool call and include a short reason."),
         meta={"expected_tool": None, "expected_args": {}, "require_reason": True,
               "allowed_tools": ["read_file", "list_dir"], "strict_json": True}),
    Task("agent_schema_collision", "agentic_tool", "text", "agentic_action", level="short", difficulty=1.7, num_predict=512,
         prompt=(AGENTIC_CONTRACT +
                 "Situation: use send_email(to, subject) to acknowledge an incident to ops@example.com. "
                 "The subject should be Incident acknowledged. Do not use action, kwargs, function_call, or note keys."),
         meta={"expected_tool": "send_email", "expected_args": {"to": "ops@example.com", "subject": "Incident acknowledged"},
               "allowed_tools": ["send_email", "create_ticket"], "strict_json": True,
               "forbidden_top_level_keys": ["action", "kwargs", "function_call", "note"]}),
    Task("agent_state_delta", "agentic_tool", "text", "agentic_action", level="short", difficulty=1.8, num_predict=512,
         prompt=(AGENTIC_CONTRACT +
                 "Situation: available tools are cart.update(sku, quantity), cart.remove(sku), and cart.checkout(). "
                 "The cart currently has A1 quantity 1 and B2 quantity 5. The user says: remove two B2. "
                 "Choose the final state update, not the change amount."),
         meta={"expected_tool": "cart.update", "expected_args": {"sku": "B2", "quantity": 3},
               "allowed_tools": ["cart.update", "cart.remove", "cart.checkout"], "strict_json": True,
               "tool_aliases": {"update_cart": "cart.update"}, "arg_aliases": {"item_id": "sku"}}),
    Task("agent_malformed_repair", "agentic_tool", "text", "agentic_action", level="short", difficulty=1.7, num_predict=512,
         prompt=(AGENTIC_CONTRACT +
                 "Situation: a prior read_file call was malformed because it used positional args instead of named args. "
                 "Repair the call so it reads README.md using a named path argument."),
         meta={"expected_tool": "read_file", "expected_args": {"path": "README.md"},
               "allowed_tools": ["read_file", "list_dir"], "strict_json": True}),
    Task("agent_nested_args", "agentic_tool", "text", "agentic_action", level="short", difficulty=1.9, num_predict=768,
         prompt=(AGENTIC_CONTRACT +
                 "Situation: use ticket.create(ticket) to create a ticket. "
                 "The ticket title is Disk alert, the priority is high, and the labels are infra and disk. "
                 "Preserve ticket as a nested object inside args; do not flatten its fields."),
         meta={"expected_tool": "ticket.create",
               "expected_args": {"ticket": {"title": "Disk alert", "priority": "high", "labels": ["infra", "disk"]}},
               "allowed_tools": ["ticket.create", "ticket.update"], "strict_json": True}),
    # ---------- tech_writing (subjective) ----------
    Task("wr_rag", "tech_writing", "text", "subjective", difficulty=1.2, num_predict=2048, judge=True,
         rubric="accuracy, clarity, analogy quality, concision, no hallucination",
         prompt="Explain Retrieval Augmented Generation to a Python dev who doesn't know ML. "
                "120-160 words, no fluff, one concrete analogy."),
    # ---------- long_context (needle) ----------
    Task("needle", "long_context", "text", "needle", level="full", difficulty=1.5,
         prompt="", meta={"needle_token": "SECRET_CODE_77",
                          "context_sizes": [4000, 16000, 32000, 65536]}),
    # ---------- ocr (vision) ----------
    Task("ocr_invoice", "ocr", "vision", "ocr", difficulty=1.0,
         prompt="Transcribe all text in this image exactly. Output only the text.",
         meta={"reference": "INVOICE 2026-0042 BillTo: BrightWave Ltd Amount Due: GBP 1,240.50 Due: 2026-07-31",
               "noisy": False}),
    Task("ocr_noisy", "ocr", "vision", "ocr", level="short", difficulty=1.4,
         prompt="Transcribe all text in this image exactly. Output only the text.",
         meta={"reference": "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.",
               "noisy": True}),
    # Real-image OCR fixtures: narrow extraction targets amid visual distractors.
    Task("ocr_receipt_total", "ocr", "vision", "exact_code", level="short", difficulty=1.6,
         prompt="Read the receipt image. Output only the final total, with no label or currency symbol.",
         meta={"image_path": "fixtures/ocr/receipt_tiny.png", "expected": "18.47"}),
    Task("ocr_table_cell", "ocr", "vision", "exact_code", level="short", difficulty=1.7,
         prompt="Read the warehouse table image. Output only the Bin value in the row marked HOLD.",
         meta={"image_path": "fixtures/ocr/table_tiny.png", "expected": "H7-42"}),
    Task("ocr_form_code", "ocr", "vision", "exact_code", level="short", difficulty=1.5,
         prompt="Read the device return label. Output only the Case ref. code exactly as printed.",
         meta={"image_path": "fixtures/ocr/form_tiny.png", "expected": "ZX-7Q/19.A"}),
    Task("ocr_noisy_label", "ocr", "vision", "exact_code", level="short", difficulty=1.8,
         prompt="Read the noisy inventory label. Output only the lot code.",
         meta={"image_path": "fixtures/ocr/noisy_label_tiny.png", "expected": "LOT-42B"}),
    # ---------- pdf (vision or pypdf) ----------
    Task("pdf_text", "pdf", "vision", "ocr", level="short", difficulty=1.1,
         prompt="Transcribe all text in this document image exactly. Output only the text.",
         meta={"reference": "Quarterly Report Q2 2026 Revenue 1.2M Expenses 800K Net 400K",
               "noisy": False, "is_pdf": True}),
    # ---------- retrieval (embedding) ----------
    # Retrieval fixtures must be fictional and privacy-safe: invented entities and
    # scenarios only. See docs/PRIVACY_FIXTURES.md before adding real-world-derived content.
    Task("ret_ukdocs", "retrieval", "embedding", "retrieval", difficulty=1.0, prompt="",
         meta={"docs": {
             "d1": "Self Assessment tax returns are due by 31 January for online filing.",
             "d2": "Universal Credit is a monthly payment to help with living costs if on low income.",
             "d3": "You can register with a GP surgery to access NHS services near where you live.",
             "d4": "Capital Gains Tax is charged on the profit when you sell an asset that rose in value.",
             "d5": "A P60 shows the tax you have paid on your salary in the tax year."},
             "queries": [
                 ("when is the online tax return deadline", "d1"),
                 ("monthly benefit payment for low income", "d2"),
                 ("how do I see a doctor on the nhs", "d3"),
                 ("tax on profit from selling an asset", "d4"),
                 ("document showing tax paid on salary", "d5")]}),
    Task("ret_uk_services_hard", "retrieval", "embedding", "retrieval", level="short", difficulty=1.8, prompt="",
         meta={"docs": {
             "onboarding_target": "New starters get a laptop and building pass configured by IT before their first day, arranged through the onboarding checklist.",
             "broken_equipment": "A faulty laptop or monitor should be reported to IT for a loan replacement while the original is repaired.",
             "leaver_equipment": "When someone leaves the company, IT collects their equipment and disables their accounts on the last working day.",
             "equipment_upgrade": "Employees can request a hardware upgrade after three years, subject to budget approval from their department.",
             "meeting_room_target": "Meeting rooms are booked through the calendar system and released automatically if unused for fifteen minutes.",
             "desk_booking_hard": "Hybrid staff reserve a desk for the day through the office app rather than using a fixed assigned desk.",
             "av_equipment": "Video conferencing equipment faults in a meeting room should be reported to facilities, not the IT helpdesk.",
             "building_tour": "Client site visits and building tours should be arranged with reception at least a day in advance.",
             "procurement_target": "Purchase orders above a set value need two levels of manager approval before a supplier is engaged.",
             "petty_cash": "Small one-off purchases under a low threshold can be paid from petty cash without a purchase order.",
             "supplier_onboarding": "New suppliers must complete a vetting form before the company can raise a purchase order with them.",
             "retention_target": "Employee records are kept for a set number of years after someone leaves and then securely destroyed.",
             "gdpr_request": "A request to see personal data held by the company should be directed to the data protection contact.",
             "currency_target": "Expenses paid in a foreign currency are converted to the local currency using the exchange rate on the day of the purchase."},
               "queries": [
                   ("What happens with getting a computer set up before someone's first day?", "onboarding_target"),
                   ("How do I book a space for a team meeting?", "meeting_room_target"),
                   ("What approvals are needed before placing a large purchase order?", "procurement_target"),
                   ("How long does the company keep staff records after they've left?", "retention_target"),
                   ("If I paid for something abroad in a different currency, how does that get converted for my expense claim?", "currency_target")]}),
    Task("ret_uk_adversarial", "retrieval", "embedding", "retrieval", level="short", difficulty=2.0, prompt="",
         meta={"docs": {
             "pw_reset_target": "Employees who cannot sign in should use the self-service portal to set a new password before contacting the helpdesk.",
             "acct_lockout": "If an account becomes locked after several unsuccessful login attempts, it unlocks automatically after a short wait.",
             "vpn_access": "Remote staff need the VPN client installed and one-time approval from IT before connecting to internal systems.",
             "software_request": "New software installs must be requested from the IT helpdesk catalogue and approved by a line manager.",
             "annual_leave_target": "Staff should submit annual leave requests through the HR portal at least two weeks before the intended dates.",
             "sick_leave": "Absence due to illness should be reported to a manager on the first morning and does not need advance booking.",
             "parental_leave": "Parental leave eligibility depends on length of service and should be arranged with HR well before the expected date.",
             "unpaid_leave": "Extended unpaid leave requires director sign-off and is handled separately from the standard annual leave process.",
             "travel_claim_target": "Travel expense claims are reimbursed once an itemised receipt is submitted, usually within thirty days of the trip.",
             "client_entertainment": "Client entertainment costs are reimbursed only when a senior manager has pre-approved the spend.",
             "equipment_purchase": "Personal equipment purchases for home working are reimbursed up to a fixed annual allowance with a receipt.",
             "mileage_claim": "Mileage for personal car use on company business is reimbursed at the standard rate set each year.",
             "payslip_query_target": "Questions about a specific payslip figure should go to payroll with the pay period and employee number.",
             "tax_code_query": "An incorrect tax code should be queried with HMRC directly, since payroll only applies the code it is given.",
             "pension_enrolment": "Pension enrolment happens automatically after three months and can be opted out of through the pension provider, not payroll.",
             "overtime_pay": "Overtime pay is only approved by a manager and confirmed with payroll on the weekly timesheet.",
             "phishing_report_target": "A suspicious email asking for login details should be forwarded to the security team and not replied to.",
             "data_breach_report": "Any accidental sharing of customer data with the wrong recipient must be reported to security within one hour.",
             "device_loss_report": "A lost or stolen laptop should be reported immediately so it can be remotely locked.",
             "password_policy": "Company login passwords must be changed every ninety days as part of the security policy and cannot reuse the previous five passwords.",
             "desk_booking_target": "Staff on a hybrid schedule should book a desk through the office app before coming in on a given day.",
             "visitor_badge": "External visitors to the office need a badge issued at reception and must be signed in by their host.",
             "parking_permit": "Office parking permits are limited and allocated by a monthly waiting list.",
             "after_hours_access": "Access to the office building outside normal hours requires a separate out-of-hours pass from facilities.",
             "mfa_device_target": "Staff changing to a new handset must enrol its authenticator in the security portal before it can confirm multi-factor access.",
             "mfa_reset": "A password reset restores an account password but does not replace an authenticator when a handset has changed.",
             "mfa_policy": "Account security policy requires multi-factor authentication for remote access and managed handset use.",
             "mfa_lost_phone": "A lost company handset should be reported to security so its authenticator and account access can be disabled.",
             "invoice_po_target": "Procurement returns a supplier bill that cannot be matched to an approved order until the supplier provides the purchase order number.",
             "invoice_expense": "Employees submit a business expense invoice or bill with receipts through the expense claim system for reimbursement.",
             "invoice_supplier": "Supplier onboarding verifies invoice payment details before procurement can create a supplier account and order record.",
             "invoice_approval": "Purchase order approval is required before procurement commits company funds to a supplier or accepts an order-related invoice."},
               "queries": [
                   ("I forgot my login and want to fix it myself without waiting on a support ticket, what should I do first?", "pw_reset_target"),
                   ("What's the notice period for booking time off work in advance?", "annual_leave_target"),
                   ("How do I get reimbursed for a business trip I paid for myself?", "travel_claim_target"),
                   ("Who do I contact if my pay looks wrong for a specific period?", "payslip_query_target"),
                   ("I got a weird email asking me to log in somewhere, what should I do with it?", "phishing_report_target"),
                   ("How do I reserve a workspace before coming into the office?", "desk_booking_target"),
                   ("I changed handsets and the second sign-in check no longer works; which internal process restores access?", "mfa_device_target"),
                   ("Accounts received a supplier bill that cannot be matched to an authorised order; what process handles it?", "invoice_po_target")],
               "cases": [
                   {"target": "pw_reset_target", "distractors": ["acct_lockout", "vpn_access", "software_request"]},
                   {"target": "annual_leave_target", "distractors": ["sick_leave", "parental_leave", "unpaid_leave"]},
                   {"target": "travel_claim_target", "distractors": ["client_entertainment", "equipment_purchase", "mileage_claim"]},
                   {"target": "payslip_query_target", "distractors": ["tax_code_query", "pension_enrolment", "overtime_pay"]},
                   {"target": "phishing_report_target", "distractors": ["data_breach_report", "device_loss_report", "password_policy"]},
                   {"target": "desk_booking_target", "distractors": ["visitor_badge", "parking_permit", "after_hours_access"]},
                   {"target": "mfa_device_target", "distractors": ["mfa_reset", "mfa_policy", "mfa_lost_phone"]},
                   {"target": "invoice_po_target", "distractors": ["invoice_expense", "invoice_supplier", "invoice_approval"]}]}),
    # ---------- reasoning ----------
    # Locked priority order's lowest bucket (agentic > coding > RAG > general reasoning),
    # previously had zero task coverage. Each answer below was independently verified by
    # exhaustive search or Monte Carlo simulation before being written here, not taken from
    # any external source's stated answer key. Scored with exact visible-answer matching;
    # strip_thinking() removes hidden reasoning before comparison, while mixed or contradictory
    # visible answers are rejected instead of receiving substring credit.
    Task("reasoning_bridge_crossing", "reasoning", "text", "exact", level="short", difficulty=1.0, num_predict=2048,
         prompt="Four people, A, B, C, and D, must cross a rickety bridge at night. They have exactly "
                "one torch, and the bridge cannot be crossed without it. At most two people can cross "
                "at a time, and any pair crossing together moves at the slower person's pace. Someone "
                "must carry the torch back across for anyone still on the near side to cross again. "
                "Crossing times: A takes 1 minute, B takes 2 minutes, C takes 5 minutes, D takes 10 "
                "minutes. What is the minimum total time for all four to cross? Answer with exactly "
                "one of these phrases and nothing else: '10 minutes', '14 minutes', '17 minutes', "
                "'not enough information'.",
         meta={"expected": "17 minutes"}),
    Task("reasoning_poisoned_wine", "reasoning", "text", "exact", level="short", difficulty=1.3, num_predict=2048,
         prompt="A king has 1000 bottles of wine. Exactly one bottle is poisoned. The poison is "
                "odorless and tasteless, and causes death exactly 24 hours after a sip, with no "
                "symptoms before then. The king has 10 prisoners and 24 hours before an event where "
                "the wine will be served. What is the minimum way to guarantee identifying the exact "
                "poisoned bottle in time? Answer with exactly one of these phrases and nothing else: "
                "'each prisoner tastes a different bottle', 'binary-coded prisoner testing', "
                "'every prisoner tastes every bottle', 'not possible with 10 prisoners'.",
         meta={"expected": "binary-coded prisoner testing"}),
    Task("reasoning_birthday_twins", "reasoning", "text", "exact", level="short", difficulty=0.9, num_predict=2048,
         prompt="How many pairs of identical twins must be in a room for there to be at least a 50% "
                "chance that two different people in the room share the same birthday? Note that each "
                "pair of twins already shares a birthday with each other. Answer with exactly one of "
                "these phrases and nothing else: '1 pair', '12 pairs', '23 pairs', '46 pairs'.",
         meta={"expected": "1 pair"}),
    Task("reasoning_monty_hall", "reasoning", "text", "exact", level="short", difficulty=1.1, num_predict=2048,
         prompt="You are on a game show facing three closed doors. Behind one door is a car; behind "
                "the other two are goats. You pick door 1. The host, who knows what is behind each "
                "door and always avoids revealing the car, opens door 3 to reveal a goat. You are now "
                "offered the choice to stick with door 1 or switch to door 2. Which choice gives you "
                "the better chance of winning the car, and what is that chance? Answer with exactly "
                "one of these phrases and nothing else: 'stick with door 1, 1/2 chance', 'stick with "
                "door 1, 1/3 chance', 'switch to door 2, 1/2 chance', 'switch to door 2, 2/3 chance'.",
         meta={"expected": "switch to door 2, 2/3 chance"}),
    Task("reasoning_wolf_goat_cabbage", "reasoning", "text", "exact", level="short", difficulty=1.2, num_predict=2048,
         prompt="A farmer must cross a river with a wolf, a goat, and a cabbage. The boat only holds "
                "the farmer plus one item at a time. If left alone together without the farmer, the "
                "wolf will eat the goat, and the goat will eat the cabbage. What is the minimum number "
                "of one-way boat crossings needed to get the farmer, wolf, goat, and cabbage all "
                "safely across? Answer with exactly one of these phrases and nothing else: "
                "'5 crossings', '7 crossings', '9 crossings', '11 crossings'.",
         meta={"expected": "7 crossings"}),
]


def tasks_for(level: str, categories: Optional[List[str]], families: Optional[List[str]]) -> List[Task]:
    depth = LEVELS.index(level)
    out = [t for t in TASKS if LEVELS.index(t.level) <= depth]
    if categories:
        out = [t for t in out if t.category in categories]
    if families:
        out = [t for t in out if t.family in families]
    return out


def make_needle_prompt(context_size: int, needle_token: str, *, chars_per_token: float = 6.85) -> str:
    """Build a filler context near the requested token target.

    V9.5.11 no longer treats this character estimate as the measurement. The runner
    calibrates or records Ollama's prompt_eval_count and uses this function only to
    construct the candidate prompt text.
    """
    sentence = "The archive contains routine maintenance notes. "
    approx_chars = int(context_size * float(chars_per_token or 6.85))
    filler = (sentence * (approx_chars // len(sentence) + 1))[:approx_chars]
    needle = f" The secret authorization code is {needle_token}. "
    mid = len(filler) // 2
    body = filler[:mid] + needle + filler[mid:]
    return (f"Read the following text carefully.\n\n{body}\n\n"
            f"Question: what is the secret authorization code? Answer with only the code.")
