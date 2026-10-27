# Setup

## 1. Download your credentials
Open the [Credentials page](https://console.cloud.google.com/apis/credentials?project=youtube-scraper-512714) and sign in with the Google account that has access to the project `youtube-scraper-512714`.

1. Under **OAuth 2.0 Client IDs**, click the client name, then click **Download JSON**. The file is saved as `client_secret_....json`.
2. Under **API Keys**, click **Show key** next to the YouTube key and copy it.

If there is no **Download JSON** button, ask Nathan to send you the file.

## 2. Install
You need Python 3.10 or newer.

```
git clone https://github.com/njoseph789/YouTube-Thumbnail-CV.git
cd YouTube-Thumbnail-CV
python -m pip install google-auth-oauthlib google-cloud-bigquery requests
```

Don't install `requirements.txt` for this; it's the full ML stack and the scraper doesn't need it.

## 3. Add your secrets
Copy the `client_secret_....json` file into the project folder without renaming it. Then create a file named `.env` in the same folder:

```
YOUTUBE_API_KEY=your-key-here
CLAIMED_BY=YourFirstName
```

Git ignores both files.

## 4. Add creators to the Creator Tracker sheet
Open the [Creator Tracker](https://docs.google.com/spreadsheets/d/17Pwf7ZBwKnHGMreaRuwShroqn-lJvhY8GzqB0C4kXNI/edit) and add a row at the bottom:

| Creator | Channel URL |
|---|---|
| Matt Stonie | https://www.youtube.com/@mattstonie |

- **Creator** becomes the Drive folder name. Don't rename it after scraping.
- **Channel URL** must be the channel's `@handle` link.
- Leave Channel ID, Status, Claimed by and Date empty. The scraper fills them in.

## 5. Run the scraper
```
python scraper.py
```

**First run:** your browser opens a Google login.
1. Sign in with the Google account you gave Nathan.
2. If you see "Google hasn't verified this app", click **Advanced**, then **Go to ... (unsafe)**.
3. Tick every permission box (Drive, Sheets, BigQuery) and click **Continue**.
4. The page says "The authentication flow has completed". Close the tab and go back to the terminal.

This creates `token.json`. The login expires after 7 days, and the browser opens again when it does.

**Expected output:**
```
20 creators in the sheet, 1 folders already in Drive
Matt Stonie: already done, skipping
AM HOOPS: 612 videos
612 to do (0 already in BigQuery)
  1/612 abc123XYZ
  ...
  loaded batch to BigQuery (50/612)
AM HOOPS: done
all done
```

- **Status column:** the scraper sets a creator to `in progress` with your name before it starts, and to `done` when it finishes. It skips creators that are `done` or `in progress` under someone else's name, so several people can run it at once.
- **Stopping:** Ctrl+C is safe. Run `python scraper.py` again to continue where it stopped, as long as the Status still shows your name.
- **`FAILED`:** if a creator shows `FAILED (...)`, the run continues with the next creator. Fix the cause (usually a wrong URL in the sheet) and rerun.
- **Stuck `in progress` under someone else's name:** if their run crashed, clear the Status, Claimed by and Date cells for that row in the sheet and run the scraper again.
