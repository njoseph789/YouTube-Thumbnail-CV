import glob
import os
from datetime import date
from urllib.parse import urlparse

from dotenv import load_dotenv
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

load_dotenv()
SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/spreadsheets",
]
SHEET_ID = "17Pwf7ZBwKnHGMreaRuwShroqn-lJvhY8GzqB0C4kXNI"
HEADERS = ["Creator", "Channel URL", "Channel ID", "Status", "Claimed by", "Date"]
CLAIMED_BY = os.environ["CLAIMED_BY"]
YOUTUBE_API_KEY = os.environ["YOUTUBE_API_KEY"]


def claim_next_creator(sheets):
    values = sheets.spreadsheets().values()
    rows = values.get(spreadsheetId=SHEET_ID, range="A:F").execute().get("values", [])
    if not rows or len(rows[0]) < len(HEADERS):
        values.update(spreadsheetId=SHEET_ID, range="A1:F1",
                      valueInputOption="RAW", body={"values": [HEADERS]}).execute()
    for i, row in enumerate(rows[1:], start=2):
        row = row + [""] * (6 - len(row))
        if row[0] and not row[3]:
            # re-read the status cell right before claiming to shrink the race window
            fresh = values.get(spreadsheetId=SHEET_ID, range=f"D{i}").execute().get("values")
            if fresh and fresh[0] and fresh[0][0]:
                continue
            values.update(spreadsheetId=SHEET_ID, range=f"D{i}:F{i}", valueInputOption="RAW",
                          body={"values": [["in progress", CLAIMED_BY, str(date.today())]]}).execute()
            return i, row[0], row[1], row[2]
    return None


def channel_id_from_url(youtube, url):
    path = urlparse(url.strip()).path.strip("/")
    if path.startswith("channel/"):
        return path.split("/")[1]
    if path.startswith("@"):
        items = youtube.channels().list(part="id", forHandle=path.split("/")[0]).execute().get("items", [])
        return items[0]["id"] if items else None
    if path.startswith("user/"):
        items = youtube.channels().list(part="id", forUsername=path.split("/")[1]).execute().get("items", [])
        return items[0]["id"] if items else None
    return None


def lookup_channels(youtube, name):
    hits = youtube.search().list(part="snippet", q=name, type="channel", maxResults=3).execute()["items"]
    ids = [h["snippet"]["channelId"] for h in hits]
    if not ids:
        return []
    items = youtube.channels().list(part="snippet,statistics", id=",".join(ids)).execute()["items"]
    return sorted(items, key=lambda c: ids.index(c["id"]))


def choose_channel(name, candidates):
    print(f"\nCandidates for '{name}':")
    for n, c in enumerate(candidates, start=1):
        subs = c["statistics"].get("subscriberCount", "hidden")
        print(f"  {n}. {c['snippet']['title']}  ({c['snippet'].get('customUrl', 'no handle')})  subs: {subs}")
    while True:
        pick = input(f"Pick 1-{len(candidates)}: ").strip()
        if pick.isdigit() and 1 <= int(pick) <= len(candidates):
            return candidates[int(pick) - 1]["id"]


secret = glob.glob("client_secret*.json")[0]
creds = InstalledAppFlow.from_client_secrets_file(secret, SCOPES).run_local_server(port=0)
sheets = build("sheets", "v4", credentials=creds)
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
values = sheets.spreadsheets().values()

claimed = claim_next_creator(sheets)
if claimed is None:
    print("No unclaimed creators left.")
else:
    row_num, name, url, channel_id = claimed
    print("Claimed:", row_num, name)
    if not channel_id and url:
        channel_id = channel_id_from_url(youtube, url)
    if not channel_id:
        candidates = lookup_channels(youtube, name)
        if candidates:
            channel_id = choose_channel(name, candidates)
    if channel_id:
        values.update(spreadsheetId=SHEET_ID, range=f"C{row_num}", valueInputOption="RAW",
                      body={"values": [[channel_id]]}).execute()
        print("Channel ID:", channel_id)
    else:
        values.clear(spreadsheetId=SHEET_ID, range=f"D{row_num}:F{row_num}").execute()
        print("No channel found; released the claim.")
