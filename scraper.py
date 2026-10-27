"""Scrape every creator in the tracker sheet: thumbnails -> Drive subfolder per creator,
normalized video metadata -> BigQuery. Creators marked done in the sheet that also have a
Drive folder are skipped; anything else is scraped (or resumed)."""
import glob
import json
import os
import re
import threading
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from google.auth.transport.requests import AuthorizedSession, Request
from google.cloud import bigquery
from google.oauth2.credentials import Credentials

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = "youtube-scraper-512714"
DATASET = f"{PROJECT}.youtube_thumbnails"
PARENT_FOLDER = "1ZnJkfHHtUeDsuR80pDtq4AP-ikVhbIlS"  # "Thumbnail Images"
SHEET_ID = "17Pwf7ZBwKnHGMreaRuwShroqn-lJvhY8GzqB0C4kXNI"  # creator tracker, first tab
SHEETS = "https://sheets.googleapis.com/v4/spreadsheets"
SCOPES = ["https://www.googleapis.com/auth/drive",
          "https://www.googleapis.com/auth/spreadsheets",
          "https://www.googleapis.com/auth/bigquery"]
PARTS = ["snippet", "contentDetails", "statistics", "status", "topicDetails",
         "localizations", "player", "paidProductPlacementDetails"]
SIZES = ["fhd", "maxres", "standard", "high", "medium", "default"]  # best first
WORKERS = 6  # thumbnails downloaded/uploaded in parallel
TIMEOUT = 30  # seconds, for every network call
_local = threading.local()


def S(name, type_="STRING", mode="NULLABLE"):
    return bigquery.SchemaField(name, type_, mode=mode)


VIDEO_SCHEMA = [
    S("video_id", mode="REQUIRED"), S("channel_id"), S("channel_title"), S("title"), S("description"),
    S("published_at", "TIMESTAMP"), S("category_id"), S("live_broadcast_content"),
    S("default_language"), S("default_audio_language"),
    S("duration"), S("duration_seconds", "INTEGER"), S("dimension"), S("definition"),
    S("caption", "BOOLEAN"), S("licensed_content", "BOOLEAN"), S("projection"),
    S("upload_status"), S("privacy_status"), S("license"), S("embeddable", "BOOLEAN"),
    S("public_stats_viewable", "BOOLEAN"), S("made_for_kids", "BOOLEAN"),
    S("view_count", "INTEGER"), S("like_count", "INTEGER"), S("favorite_count", "INTEGER"),
    S("comment_count", "INTEGER"),
    S("has_paid_product_placement", "BOOLEAN"), S("embed_html"),
    S("drive_file_id"), S("drive_url"), S("thumbnail_size"), S("scraped_at", "TIMESTAMP"), S("scraped_by"),
]
CHILD_SCHEMAS = {
    "video_tags": [S("video_id", mode="REQUIRED"), S("position", "INTEGER"), S("tag")],
    "video_topics": [S("video_id", mode="REQUIRED"), S("kind"), S("value")],
    "video_thumbnails": [S("video_id", mode="REQUIRED"), S("size"), S("url"),
                         S("width", "INTEGER"), S("height", "INTEGER")],
    "video_localizations": [S("video_id", mode="REQUIRED"), S("language"), S("title"), S("description")],
    "video_content_ratings": [S("video_id", mode="REQUIRED"), S("rating_system"), S("rating")],
    "video_region_restrictions": [S("video_id", mode="REQUIRED"), S("restriction"), S("region")],
}
SCHEMAS = {"videos": VIDEO_SCHEMA, **CHILD_SCHEMAS}

# API fields the mapping knows about; anything else triggers a one-time warning.
KNOWN = {
    "snippet": {"publishedAt", "channelId", "title", "description", "thumbnails", "channelTitle", "tags",
                "categoryId", "liveBroadcastContent", "defaultLanguage", "localized", "defaultAudioLanguage"},
    "contentDetails": {"duration", "dimension", "definition", "caption", "licensedContent",
                       "regionRestriction", "contentRating", "projection"},
    "statistics": {"viewCount", "likeCount", "favoriteCount", "commentCount"},
    # selfDeclaredMadeForKids and containsSyntheticMedia are known but deliberately not stored (D-033)
    "status": {"uploadStatus", "privacyStatus", "license", "embeddable", "publicStatsViewable",
               "madeForKids", "selfDeclaredMadeForKids", "containsSyntheticMedia"},
    "topicDetails": {"topicIds", "relevantTopicIds", "topicCategories"},
    "player": {"embedHtml"},
    "paidProductPlacementDetails": {"hasPaidProductPlacement"},
    "localizations": None,  # map of language -> {title, description}; checked per language below
}
_warned = set()


def load_env():
    env = {}
    for line in open(os.path.join(HERE, ".env"), encoding="utf-8"):
        m = re.match(r"\s*([^#=\s]+)\s*=\s*(.*)", line)
        if m:
            env[m[1]] = m[2].strip().strip("\"'")
    return env


ENV = load_env()


def yt(endpoint, **params):
    params["key"] = ENV["YOUTUBE_API_KEY"]
    url = f"https://www.googleapis.com/youtube/v3/{endpoint}?" + urllib.parse.urlencode(params)
    return json.load(urllib.request.urlopen(url, timeout=TIMEOUT))


def list_video_ids(playlist_id):
    ids, token = [], None
    while True:
        page = yt("playlistItems", part="contentDetails", playlistId=playlist_id,
                  maxResults=50, **({"pageToken": token} if token else {}))
        ids += [i["contentDetails"]["videoId"] for i in page["items"]]
        token = page.get("nextPageToken")
        if not token:
            return ids


def find_or_create_folder(drive, name):
    q = (f"'{PARENT_FOLDER}' in parents and name = '{name.replace(chr(39), chr(92) + chr(39))}' "
         "and trashed = false "
         "and mimeType = 'application/vnd.google-apps.folder'")
    r = drive.get("https://www.googleapis.com/drive/v3/files", params={
        "q": q, "fields": "files(id)", "supportsAllDrives": "true",
        "includeItemsFromAllDrives": "true"})
    r.raise_for_status()
    if r.json()["files"]:
        return r.json()["files"][0]["id"]
    r = drive.post("https://www.googleapis.com/drive/v3/files", params={"supportsAllDrives": "true"},
                   json={"name": name, "mimeType": "application/vnd.google-apps.folder",
                         "parents": [PARENT_FOLDER]})
    r.raise_for_status()
    return r.json()["id"]


def existing_files(drive, folder_id):
    files, token = {}, None
    while True:
        r = drive.get("https://www.googleapis.com/drive/v3/files", params={
            "q": f"'{folder_id}' in parents and trashed = false",
            "fields": "nextPageToken, files(id,name,webViewLink)", "pageSize": 1000,
            "supportsAllDrives": "true", "includeItemsFromAllDrives": "true",
            **({"pageToken": token} if token else {})})
        r.raise_for_status()
        files.update({f["name"]: f for f in r.json()["files"]})
        token = r.json().get("nextPageToken")
        if not token:
            return files


def upload(drive, folder_id, name, data):
    b = "thumb_boundary"
    meta = json.dumps({"name": name, "parents": [folder_id]})
    body = (f"--{b}\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n{meta}\r\n"
            f"--{b}\r\nContent-Type: image/jpeg\r\n\r\n").encode() + data + f"\r\n--{b}--".encode()
    r = drive.post("https://www.googleapis.com/upload/drive/v3/files", data=body,
                   headers={"Content-Type": f"multipart/related; boundary={b}"},
                   params={"uploadType": "multipart", "supportsAllDrives": "true",
                           "fields": "id,webViewLink"}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def download_best_thumbnail(thumbnails):
    for size in SIZES:
        if size in thumbnails:
            try:
                return size, urllib.request.urlopen(thumbnails[size]["url"], timeout=TIMEOUT).read()
            except urllib.error.HTTPError:
                continue  # e.g. maxres listed but not actually available
    raise RuntimeError("no downloadable thumbnail")


def to_int(x):
    return int(x) if x is not None else None


def to_bool(x):
    if isinstance(x, str):
        return x.lower() == "true"
    return x


def duration_seconds(iso):
    m = re.fullmatch(r"P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?", iso or "")
    if not m:
        return None
    d, h, mi, s = (int(g or 0) for g in m.groups())
    return d * 86400 + h * 3600 + mi * 60 + s


def warn_unknown(vid, v):
    for key in v:
        if key not in KNOWN and key not in ("kind", "etag", "id") and ("video", key) not in _warned:
            _warned.add(("video", key))
            print(f"  WARNING: unmapped video part '{key}' (first seen on {vid})")
    for part, known in KNOWN.items():
        if known is None:
            continue
        for key in v.get(part, {}):
            if key not in known and (part, key) not in _warned:
                _warned.add((part, key))
                print(f"  WARNING: unmapped field {part}.{key} (first seen on {vid})")
    for lang, loc in v.get("localizations", {}).items():
        for key in loc:
            if key not in ("title", "description") and ("localizations", key) not in _warned:
                _warned.add(("localizations", key))
                print(f"  WARNING: unmapped field localizations.{lang}.{key} (first seen on {vid})")


def build_rows(v, cid, name, f, size):
    """Turn one API video resource into a videos row plus child-table rows."""
    vid = v["id"]
    warn_unknown(vid, v)
    sn, cd, st = v.get("snippet", {}), v.get("contentDetails", {}), v.get("status", {})
    stat, td = v.get("statistics", {}), v.get("topicDetails", {})
    video = {
        "video_id": vid, "channel_id": sn.get("channelId", cid),
        "channel_title": sn.get("channelTitle", name), "title": sn.get("title"),
        "description": sn.get("description"), "published_at": sn.get("publishedAt"),
        "category_id": sn.get("categoryId"), "live_broadcast_content": sn.get("liveBroadcastContent"),
        "default_language": sn.get("defaultLanguage"),
        "default_audio_language": sn.get("defaultAudioLanguage"),
        "duration": cd.get("duration"), "duration_seconds": duration_seconds(cd.get("duration")),
        "dimension": cd.get("dimension"), "definition": cd.get("definition"),
        "caption": to_bool(cd.get("caption")), "licensed_content": to_bool(cd.get("licensedContent")),
        "projection": cd.get("projection"),
        "upload_status": st.get("uploadStatus"), "privacy_status": st.get("privacyStatus"),
        "license": st.get("license"), "embeddable": st.get("embeddable"),
        "public_stats_viewable": st.get("publicStatsViewable"), "made_for_kids": st.get("madeForKids"),
        "view_count": to_int(stat.get("viewCount")), "like_count": to_int(stat.get("likeCount")),
        "favorite_count": to_int(stat.get("favoriteCount")),
        "comment_count": to_int(stat.get("commentCount")),
        "has_paid_product_placement":
            v.get("paidProductPlacementDetails", {}).get("hasPaidProductPlacement"),
        "embed_html": v.get("player", {}).get("embedHtml"),
        "drive_file_id": f["id"], "drive_url": f.get("webViewLink"), "thumbnail_size": size,
        "scraped_at": datetime.now(timezone.utc).isoformat(), "scraped_by": ENV.get("CLAIMED_BY"),
    }
    rr = cd.get("regionRestriction", {})
    children = {
        "video_tags": [{"video_id": vid, "position": i, "tag": t}
                       for i, t in enumerate(sn.get("tags", []))],
        "video_topics": [{"video_id": vid, "kind": kind,
                          # topic categories arrive as Wikipedia URLs; keep only the page name
                          "value": urllib.parse.unquote(val.rsplit("/", 1)[-1])
                          if kind == "topic_category" else val}
                         for kind, key in (("topic_category", "topicCategories"),
                                           ("relevant_topic_id", "relevantTopicIds"),
                                           ("topic_id", "topicIds"))
                         for val in td.get(key, [])],
        "video_thumbnails": [{"video_id": vid, "size": sz, "url": t.get("url"),
                              "width": t.get("width"), "height": t.get("height")}
                             for sz, t in sn.get("thumbnails", {}).items()],
        "video_localizations": [{"video_id": vid, "language": lang, "title": l.get("title"),
                                 "description": l.get("description")}
                                for lang, l in v.get("localizations", {}).items()],
        "video_content_ratings": [{"video_id": vid, "rating_system": system, "rating": str(rating)}
                                  for system, rating in cd.get("contentRating", {}).items()],
        "video_region_restrictions": [{"video_id": vid, "restriction": kind, "region": region}
                                      for kind in ("allowed", "blocked") for region in rr.get(kind, [])],
    }
    return video, children


def process_video(creds, folder_id, in_drive, cid, name, v):
    """Runs in a worker thread: upload the thumbnail (if needed) and build the BigQuery rows."""
    if not hasattr(_local, "drive"):  # one Drive session per thread (sessions aren't thread-safe)
        _local.drive = AuthorizedSession(creds)
    vid = v["id"]
    size, f = None, in_drive.get(f"{vid}.jpg")
    if not f:
        size, data = download_best_thumbnail(v["snippet"]["thumbnails"])
        f = upload(_local.drive, folder_id, f"{vid}.jpg", data)
    return build_rows(v, cid, name, f, size)


def ensure_tables(bq):
    for name, schema in SCHEMAS.items():
        ref = f"{DATASET}.{name}"
        try:
            t = bq.get_table(ref)
        except Exception:
            bq.create_table(bigquery.Table(ref, schema=schema))
            continue
        if [f.name for f in t.schema] != [f.name for f in schema]:
            if t.num_rows:
                raise SystemExit(f"Table {name} has rows and a different schema; refusing to drop it.")
            bq.delete_table(ref)
            bq.create_table(bigquery.Table(ref, schema=schema))


def load_rows(bq, name, rows):
    if rows:
        bq.load_table_from_json(rows, f"{DATASET}.{name}", job_config=bigquery.LoadJobConfig(
            schema=SCHEMAS[name], write_disposition="WRITE_APPEND")).result()


def existing_ids(bq, name):
    return {r.video_id for r in bq.query(f"SELECT DISTINCT video_id FROM `{DATASET}.{name}`").result()}


def get_credentials():
    """Load token.json, refreshing it; if missing or expired for good, open the browser login."""
    path = os.path.join(HERE, "token.json")
    creds = Credentials.from_authorized_user_file(path, SCOPES) if os.path.exists(path) else None
    if creds and not creds.valid:
        try:
            creds.refresh(Request())
        except Exception:  # refresh token expired or revoked (testing mode: every 7 days)
            creds = None
    if not creds:
        from google_auth_oauthlib.flow import InstalledAppFlow  # only needed for the browser login
        secrets = glob.glob(os.path.join(HERE, "client_secret*.json"))
        if not secrets:
            raise SystemExit("No client_secret*.json in the project folder. See SETUP.md.")
        creds = InstalledAppFlow.from_client_secrets_file(secrets[0], SCOPES).run_local_server(port=0)
    with open(path, "w") as f:
        f.write(creds.to_json())
    return creds


def norm(name):
    """Creator-name key for matching sheet rows to Drive folders (ignores case and extra spaces)."""
    return " ".join(name.split()).casefold()


def read_sheet(sess):
    """Return (tab title, [(sheet row number, creator, url, channel_id, status)]) for the first tab."""
    meta = sess.get(f"{SHEETS}/{SHEET_ID}", params={"fields": "sheets.properties(sheetId,title)"},
                    timeout=TIMEOUT)
    meta.raise_for_status()
    title = next(s["properties"]["title"] for s in meta.json()["sheets"]
                 if s["properties"]["sheetId"] == 0)
    r = sess.get(f"{SHEETS}/{SHEET_ID}/values/" + urllib.parse.quote(f"'{title}'!A2:F", safe=""),
                 timeout=TIMEOUT)
    r.raise_for_status()
    rows = []
    for i, row in enumerate(r.json().get("values", []), start=2):  # row 1 is the header
        row = [c.strip() for c in row] + [""] * (6 - len(row))
        if row[0] and row[1]:
            rows.append((i, row[0], row[1], row[2], row[3]))
    return title, rows


def write_sheet(sess, title, row_num, cells):
    """cells: {column letter: value}, written to that sheet row."""
    sess.post(f"{SHEETS}/{SHEET_ID}/values:batchUpdate", timeout=TIMEOUT, json={
        "valueInputOption": "RAW",
        "data": [{"range": f"'{title}'!{col}{row_num}", "values": [[val]]} for col, val in cells.items()],
    }).raise_for_status()


def resolve_channel_id(url):
    m = re.search(r"youtube\.com/channel/(UC[\w-]+)", url)
    if m:
        params = {"id": m[1]}
    else:
        m = re.search(r"youtube\.com/@([^/?#\s]+)", url)
        if not m:
            raise ValueError(f"no @handle or channel ID found in {url!r}")
        params = {"forHandle": urllib.parse.unquote(m[1])}
    items = yt("channels", part="id", **params).get("items")
    if not items:
        raise ValueError(f"channel not found for {url!r}")
    return items[0]["id"]


def list_creator_folders(drive):
    """{normalized creator name: folder id} for the subfolders of 'Thumbnail Images'."""
    folders, token = {}, None
    while True:
        r = drive.get("https://www.googleapis.com/drive/v3/files", params={
            "q": f"'{PARENT_FOLDER}' in parents and trashed = false "
                 "and mimeType = 'application/vnd.google-apps.folder'",
            "fields": "nextPageToken, files(id,name)", "pageSize": 1000,
            "supportsAllDrives": "true", "includeItemsFromAllDrives": "true",
            **({"pageToken": token} if token else {})}, timeout=TIMEOUT)
        r.raise_for_status()
        folders.update({norm(f["name"]): f["id"] for f in r.json()["files"]})
        token = r.json().get("nextPageToken")
        if not token:
            return folders


def scrape_creator(creds, drive, bq, pool, in_bq, in_child, creator, url):
    """Scrape one creator into Drive folder `creator` and BigQuery; return their channel ID."""
    cid = resolve_channel_id(url)
    ids = list_video_ids("UULF" + cid[2:])  # UULF = uploads without Shorts
    print(f"{creator}: {len(ids)} videos")

    folder_id = find_or_create_folder(drive, creator)
    in_drive = existing_files(drive, folder_id)
    todo = [i for i in ids if i not in in_bq]
    print(f"{len(todo)} to do ({len(ids) - len(todo)} already in BigQuery)")

    done = 0
    for start in range(0, len(todo), 50):
        batch = todo[start:start + 50]
        items = yt("videos", part=",".join(PARTS), id=",".join(batch))["items"]
        videos, children = [], {n: [] for n in CHILD_SCHEMAS}
        for video, kids in pool.map(
                lambda v: process_video(creds, folder_id, in_drive, cid, creator, v), items):
            videos.append(video)
            for n, rows in kids.items():
                children[n] += [r for r in rows if r["video_id"] not in in_child[n]]
            done += 1
            print(f"  {done}/{len(todo)} {video['video_id']}")
        for n, rows in children.items():  # child tables first, videos last (videos = "done" marker)
            load_rows(bq, n, rows)
            in_child[n] |= {r["video_id"] for r in rows}
        load_rows(bq, "videos", videos)
        in_bq |= {v["video_id"] for v in videos}
        print(f"  loaded batch to BigQuery ({done}/{len(todo)})")
    return cid


def main():
    creds = get_credentials()
    drive = AuthorizedSession(creds)
    bq = bigquery.Client(project=PROJECT, credentials=creds)
    ensure_tables(bq)

    title, creators = read_sheet(drive)
    folders = list_creator_folders(drive)
    in_bq = existing_ids(bq, "videos")
    in_child = {n: existing_ids(bq, n) for n in CHILD_SCHEMAS}  # guards against crash-time duplicates
    print(f"{len(creators)} creators in the sheet, {len(folders)} folders already in Drive")

    with ThreadPoolExecutor(WORKERS) as pool:
        for row_num, creator, url, sheet_cid, status in creators:
            try:
                # folder + Status done = finished. A folder without done (e.g. a crashed run) is resumed:
                # videos already in BigQuery and thumbnails already in Drive are skipped.
                if norm(creator) in folders and status.lower() == "done":
                    print(f"{creator}: already done, skipping")
                    if not sheet_cid:
                        write_sheet(drive, title, row_num, {"C": resolve_channel_id(url)})
                    continue
                cid = scrape_creator(creds, drive, bq, pool, in_bq, in_child, creator, url)
                write_sheet(drive, title, row_num, {
                    "C": cid, "D": "done", "E": ENV.get("CLAIMED_BY", ""),
                    "F": datetime.now().strftime("%Y-%m-%d")})
                print(f"{creator}: done")
            except Exception as e:  # leave this row unmarked and carry on with the next creator
                print(f"{creator}: FAILED ({e!r})")
    print("all done")


if __name__ == "__main__":
    main()
