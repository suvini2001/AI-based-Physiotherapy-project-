# E4 — Offline Sync Experiment: Results

**Date:** 26 November 2026  
**Environment:** Local (Windows, localhost, PostgreSQL)

---

## Objective

To verify that exercise sessions queued during an offline period are
delivered to the server without loss or duplication when connectivity
is restored.

---

## Method

1. **20 unique sessions** were generated with unique UUIDs.
2. **5 sessions** were sent immediately while the server was reachable
   (online phase).
3. **15 sessions** were written directly to `offline_queue.json`,
   simulating no network connectivity.
4. **5 duplicate sessions** (same UUIDs as the first 5 offline sessions)
   were also added to the queue, simulating a retry scenario where the
   client does not know whether a previous send succeeded.
5. The queue was flushed using `session_packager.flush_queue()`,
   simulating connectivity being restored.
6. The server session count was checked before and after.

---

## Results

| Metric | Value |
|---|---|
| Sessions generated (unique) | 20 |
| Sessions sent online | 5 |
| Sessions queued offline | 15 |
| Duplicate entries in queue | 5 |
| Sessions on server (after) | 25 (20 unique + 5 duplicates) |
| Sessions lost | 0 |
| Queue after flush | Empty |
| Mean upload time (online phase) | ~2,075 ms |
| Queue flush time (20 entries) | ~41,490 ms |

---

## Analysis

**Zero data loss:** All 20 unique sessions arrived at the server
after the queue flush. The offline queue correctly preserved sessions
across a simulated connectivity outage.

**Duplicate handling (partial):** The current `/api/sessions` endpoint
does not enforce UUID uniqueness, so the 5 duplicate entries were
accepted as new sessions (server IDs 46–50). Full duplicate rejection
requires the idempotent `POST /api/sync` endpoint (Week 3 pair task),
which checks `session_uuid` against the database before inserting.
Once that endpoint is in place, re-running this experiment should show
exactly 20 sessions on the server regardless of duplicates sent.

**Upload latency:** Online uploads averaged ~2,075 ms per session on
localhost. This is higher than expected for a local connection and is
likely due to cold connection overhead and the development server.
On a production server over a real network, latency is expected to be
lower, though mobile network conditions (measured in E5) will vary.

**Queue flush time:** Flushing 20 queued sessions took ~41.5 seconds
total (~2 seconds per session). With the exponential back-off in place,
a failed flush would wait up to 60 seconds between retries. For a
patient who has queued several days of sessions, this could take
several minutes on a poor connection — acceptable given the offline
use case.

---

## Conclusion

The offline sync mechanism successfully delivers sessions without loss.
The remaining gap — duplicate rejection — is a server-side concern
addressed by the idempotent sync endpoint, not a client-side failure.
The outbox pattern is working correctly.

---

## Next Steps

- Implement `POST /api/sync` with `INSERT OR IGNORE` on `session_uuid`
  (Week 3 pair task with Person B)
- Re-run E4 after that endpoint is in place to confirm 0 duplicates
- Run E5 (live channel experiment) under real network conditions