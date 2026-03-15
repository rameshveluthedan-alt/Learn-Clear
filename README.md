# LearnClear — Deployment Guide (Google Cloud Run)
===================================================
Built by : [Your Full Name]
Telegram : https://t.me/learn_clear_bot


## Files in this folder

    main.py          — Bot application code
    Dockerfile       — Container build instructions
    requirements.txt — Python dependencies
    README.md        — This file


## One-Time Setup

### 1. Install Google Cloud CLI
Download from https://cloud.google.com/sdk/docs/install
Then run:
    gcloud auth login
    gcloud config set project YOUR_PROJECT_ID


## Deploy to Google Cloud Run

Run this single command from this folder:

    gcloud run deploy learn-clear \
      --source . \
      --region asia-south1 \
      --allow-unauthenticated \
      --min-instances 1 \
      --max-instances 3 \
      --memory 512Mi \
      --set-env-vars TELEGRAM_TOKEN=your_token_here,GEMINI_API_KEY=your_key_here

Cloud Run will build the Docker image, deploy it, and give you a URL like:
    https://learn-clear-xxxxxxxx-el.a.run.app


## Redeploy after code changes

Same command — Cloud Run creates a new revision automatically:

    gcloud run deploy learn-clear \
      --source . \
      --region asia-south1


## Environment Variables

    TELEGRAM_TOKEN   — Get from @BotFather on Telegram
    GEMINI_API_KEY   — Get from https://aistudio.google.com


## Useful Commands

    # View live logs
    gcloud run logs tail learn-clear --region asia-south1

    # View recent logs
    gcloud run logs read learn-clear --region asia-south1 --limit 50

    # Check service status
    gcloud run services describe learn-clear --region asia-south1

    # List all revisions
    gcloud run revisions list --service learn-clear --region asia-south1

    # Roll back to previous revision
    gcloud run services update-traffic learn-clear \
      --to-revisions REVISION_NAME=100 \
      --region asia-south1


## Health Check Endpoints

    /health  — Returns {"status": "ok"} — used by Cloud Run to verify container is healthy
    /wake    — Returns {"status": "awake"} — manual check endpoint


## Cost Estimate

With --min-instances 1 (always warm, no cold starts):
    Requests  : First 2 million/month free
    CPU/Memory: ~₹150-200/month for a low-traffic bot
    Total     : ₹150-200/month

With --min-instances 0 (cold starts possible, ~1-2 sec):
    Cost      : ₹0/month within free quota


## Monitoring

View in Google Cloud Console:
    https://console.cloud.google.com/run

Key metrics to watch:
    - Request count
    - Response latency
    - Error rate (should be 0%)
    - Container instance count


## Notes

- No UptimeRobot needed — Cloud Run with min-instances=1 never sleeps
- No watchdog thread needed — Cloud Run manages container health
- Logs are available in real-time via Cloud Console or gcloud CLI
- Each code deploy creates a new revision — easy to roll back if needed
