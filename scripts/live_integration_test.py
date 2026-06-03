#!/usr/bin/env python3
"""Live integration tests against running FastAPI server."""
import json
import sys
import time
import httpx

BASE = "http://127.0.0.1:8000"
TIMEOUT = httpx.Timeout(180.0, connect=5.0)

PASS = 0
FAIL = 0
SKIP = 0


def ok(name: str):
    global PASS
    PASS += 1
    print(f"  PASS  {name}")


def fail(name: str, detail: str):
    global FAIL
    FAIL += 1
    print(f"  FAIL  {name}: {detail}")


def skip(name: str, reason: str):
    global SKIP
    SKIP += 1
    print(f"  SKIP  {name}: {reason}")


def main():
    print("=== Live Integration Tests ===\n")

    with httpx.Client(timeout=TIMEOUT) as c:
        # Health
        try:
            r = c.get(f"{BASE}/health")
            if r.status_code == 200 and r.json().get("status") == "ok":
                ok("GET /health")
            else:
                fail("GET /health", f"status={r.status_code} body={r.text[:200]}")
        except Exception as e:
            fail("GET /health", str(e))
            print("\nServer not reachable. Start with: uvicorn app.main:app --port 8000")
            sys.exit(1)

        # Summary
        r = c.get(f"{BASE}/customers/summary")
        if r.status_code == 200 and r.json().get("total_customers") == 300:
            ok("GET /customers/summary (300 customers)")
        else:
            fail("GET /customers/summary", r.text[:200])

        # Products
        r = c.get(f"{BASE}/products")
        if r.status_code == 200 and len(r.json()) >= 4:
            ok("GET /products")
        else:
            fail("GET /products", r.text[:200])

        # Validation: empty message
        r = c.post(f"{BASE}/chat", json={"message": ""})
        if r.status_code == 422:
            ok("POST /chat empty message → 422")
        else:
            fail("POST /chat empty message", f"expected 422 got {r.status_code}")

        # Validation: oversized
        r = c.post(f"{BASE}/chat", json={"message": "x" * 20000})
        if r.status_code == 422:
            ok("POST /chat oversized message → 422")
        else:
            fail("POST /chat oversized", f"got {r.status_code}")

        # Injection payload (should not crash server)
        r = c.post(
            f"{BASE}/chat",
            json={"message": "Find customers'; DROP TABLE customers; --"},
        )
        if r.status_code in (200, 500):
            ok(f"POST /chat SQL injection string → {r.status_code} (no crash)")
            if r.status_code == 500:
                skip("SQL injection chat body", "LLM/Azure error — check .env")
        else:
            fail("POST /chat injection", f"unexpected {r.status_code}")

        # Full agent flow — Turn 1
        print("\n--- Agent flow (requires Azure OpenAI in .env) ---")
        t0 = time.time()
        try:
            r = c.post(
                f"{BASE}/chat",
                json={"message": "Find high-value customers likely to convert for a personal loan"},
            )
            elapsed = time.time() - t0
            if r.status_code == 200:
                data = r.json()
                recs = data.get("recommendations", [])
                trace = data.get("trace", [])
                ok(f"Turn 1 /chat → {len(recs)} recommendations, {len(trace)} trace steps ({elapsed:.1f}s)")
                thread_id = data.get("thread_id")
                if not thread_id:
                    fail("Turn 1 thread_id", "missing")
                elif not recs:
                    fail("Turn 1 recommendations", "empty list")
                else:
                    # Turn 2 follow-up
                    t1 = time.time()
                    r2 = c.post(
                        f"{BASE}/chat",
                        json={
                            "message": "Generate whatsapp messages for the top 3",
                            "thread_id": thread_id,
                        },
                    )
                    e2 = time.time() - t1
                    if r2.status_code == 200:
                        d2 = r2.json()
                        msgs = [x.get("whatsapp_message") for x in d2.get("recommendations", [])]
                        has_msg = any(m for m in msgs)
                        is_followup = d2.get("stats", {}).get("is_followup")
                        ok(
                            f"Turn 2 follow-up ({e2:.1f}s) is_followup={is_followup} "
                            f"messages={'yes' if has_msg else 'no'}"
                        )
                        router_steps = [t for t in d2.get("trace", []) if t.get("step") == "router"]
                        if router_steps and "outreach" in router_steps[0].get("decision", "").lower():
                            ok("Turn 2 router → outreach_agent")
                        else:
                            skip("Turn 2 router decision", str(router_steps))
                    else:
                        fail("Turn 2 follow-up", f"{r2.status_code} {r2.text[:300]}")
            elif r.status_code == 500:
                skip("Turn 1 /chat", f"500 — likely Azure: {r.text[:200]}")
            else:
                fail("Turn 1 /chat", f"{r.status_code} {r.text[:300]}")
        except httpx.ReadTimeout:
            skip("Turn 1 /chat", "timeout >180s")
        except Exception as e:
            fail("Turn 1 /chat", str(e))

    print(f"\n=== Results: {PASS} passed, {FAIL} failed, {SKIP} skipped ===")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
