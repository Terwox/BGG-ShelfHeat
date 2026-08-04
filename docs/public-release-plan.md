# ShelfHeat Public Release Plan

## Recommendation

Release the first public ShelfHeat website as a public Hugging Face Gradio Space on free CPU Basic hardware. The public flow should ask for a BoardGameGeek (BGG) username plus a shelf photo, fetch the user's BGG collection server-side through the approved BGG XML API token, run ShelfHeat, and return the interactive heatmap outputs.

Do not expose BGG collection CSV upload in the user interface. It is fine for the server to normalize the fetched BGG response into a temporary CSV or JSON shape behind the scenes, as long as users only type their BGG username.

## Current Access Status

- An existing BGG XML API approval email from `api@boardgamegeek.com` was found in Gmail, dated 2026-02-24.
- The BGG API bearer token has been supplied by the project owner and stored locally in `.secrets` as `BGG_API_TOKEN`.
- For deployment, copy that value into the Hugging Face Space secret named `BGG_API_TOKEN`; do not commit it, render it in the browser, or print it in logs.

## Why This Path

- Current ShelfHeat already has the full local pipeline in `shelfheat/cli.py`, BGG username/API collection loading in `shelfheat/match.py`, CSV parsing that can be reused only as internal normalization, and editable generated heatmap HTML in `shelfheat/heatmap.py`.
- Hugging Face Spaces supports public app URLs, Gradio apps, Docker apps, and static HTML apps.
- Hugging Face CPU Basic currently provides a free 2 vCPU / 16 GB memory / 50 GB non-persistent disk environment.
- The free CPU path will be slow, but it is aligned with the existing README estimate that CPU processing can take minutes per photo.
- BGG's current XML API guidance requires registered/authorized API use, server-side requests where possible, low request volume, and Powered by BGG attribution. This plan now satisfies that by using the approved app token as a server secret.
- Vercel-style serverless hosting is a poor first target because the pipeline has large dependencies, image uploads, long inference, and memory-heavy model loading.

## MVP Website Scope

The first public site should let a user:

1. Visit the public Space URL.
2. Upload one JPEG or PNG shelf photo.
3. Enter their BGG username.
4. Let the app fetch owned collection data and play history from BGG server-side.
5. Run ShelfHeat with CPU-safe defaults.
6. View summary counts.
7. Open/download the generated interactive heatmap HTML.
8. Download the results JSON.
9. Click/edit games in the generated heatmap and export/import edits.

Do not require accounts, payments, user-owned API tokens, BGG passwords, or user-managed collection files.

## Build Plan

0. Prove hosted CPU and BGG fetch feasibility before any BGG forum announcement: run one small photo plus BGG username job on a cold/new public Hugging Face CPU Basic Space, with `BGG_API_TOKEN` configured as a secret.
1. Extract a shared runner from `shelfheat/cli.py` so the CLI and web app use the same pipeline.
2. Update BGG API loading for public web use: Bearer token from environment, server-side only, 202 retry/backoff, throttling handling, friendly errors, and a tightly bounded in-memory response cache if needed.
3. Add a Gradio app entry point with photo upload, BGG username input, progress/status, and output downloads.
4. Use a per-job temporary directory and delete uploads/working files after each run. Temporary internal CSV/JSON normalization is allowed; user-facing CSV upload is not.
5. Add CPU-safe defaults: file-size limit, image downscale, one-job queue, and `no_images=True` by default so web jobs do not fetch BGG cover art, gallery images, CDN images, or per-user image caches.
6. Disable related-game play inheritance in web mode unless every `/thing` lookup is made tokenized, rate-limited, retried, and covered by tests. The MVP should avoid request-time BGG `/thing` calls during shelf matching.
7. Add Powered by BGG attribution and a link back to BGG on the public page.
8. Deploy to Hugging Face Spaces on free CPU Basic.
   - Use `uv run python scripts/deploy-hf-space.py --private` once local `.secrets` contains `BGG_API_TOKEN` and `HF_TOKEN`.
9. Smoke test cold start, warm run, invalid photo, invalid/private/unknown BGG username, output download, token absence, rate-limit handling, and privacy cleanup.
10. Add README "Try it online", known limitations, and the BGG forum post link placeholder.

## Release Gates

- Public URL opens without login.
- User can upload a photo and enter a BGG username without handling any CSV files.
- Hosted CPU Basic feasibility spike passes before the forum post goes live.
- Hosted Space has `BGG_API_TOKEN` configured as a secret and never exposes it client-side or in logs.
- BGG collection and plays requests include `Authorization: Bearer <token>` server-side.
- Web mode performs no unauthenticated BGG API calls, including hidden `/thing` lookups from related-game play inheritance.
- BGG 202 queued responses, throttling, bad usernames, private/unavailable collections, and token failures become friendly UI states.
- Default web mode uses `no_images=True` and does not call BGG image cache, gallery, or BGG/CDN image download paths.
- Small valid photo plus public BGG username produces HTML and JSON outputs.
- Generated HTML supports hover, edit, export, and import.
- Temporary files are cleaned after success and failure.
- UI states free-tier limits clearly: slow, beta, imperfect recognition.
- Existing tests pass, plus new BGG API, web-wrapper, cleanup, and hosted-smoke tests.

## Known Limitations To Say Out Loud

- It will be slow on free hosting.
- First request after cold start may be especially slow because models need to load/download.
- Recognition will be imperfect, especially for glare, odd angles, stacked boxes, and tiny spines.
- BGG collection import depends on BGG API availability, the approved app token, and the target user's collection visibility.
- Source uploads and temporary normalized collection files should be removed after each job. Generated HTML/JSON downloads are short-lived temp files and may contain the uploaded photo plus collection-derived results so the heatmap works offline. A process-local in-memory BGG response cache may be used for a short TTL, such as 15 minutes, to avoid hammering BGG during repeated attempts; it must not be written to persistent storage. Model caches are acceptable operational caches.

## Sources Checked

- [Hugging Face Spaces overview](https://huggingface.co/docs/hub/en/spaces-overview)
- [Hugging Face Gradio Spaces](https://huggingface.co/docs/hub/en/spaces-sdks-gradio)
- [Hugging Face Docker Spaces](https://huggingface.co/docs/hub/en/spaces-sdks-docker)
- [Hugging Face ZeroGPU Spaces](https://huggingface.co/docs/hub/en/spaces-zerogpu)
- [BoardGameGeek XML API guidance](https://boardgamegeek.com/using_the_xml_api)
- [BoardGameGeek XML API2 wiki](https://boardgamegeek.com/wiki/page/BGG_XML_API2)
- [Vercel Functions limits](https://vercel.com/docs/functions/limitations)

## Brief BGG Forum Post Draft

Subject: Free beta: ShelfHeat turns a shelf photo + BGG username into a play-recency heatmap

Post:

I made a small free tool called ShelfHeat. Upload a photo of your board game shelf, enter your BGG username, and it tries to identify the games and color them by play recency. Green means played recently, red means it has been a while, and purple is the shelf-of-shame zone.

It is a free beta, probably imperfect, and currently slow because it is running on free hosting. You can edit/export the results if it gets things wrong. Try it here: [URL TO ADD]
