#!/usr/bin/env python3
"""
PureTTS - Reddit Opportunity Radar
Scans Reddit for high-intent questions looking for free TTS, ElevenLabs alternatives,
or in-browser voice generators. Generates customized, non-spam replies ready to paste.
"""

import sys
import time
import json
import os
import re
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime

# Target subreddits and search RSS feeds
FEEDS = [
    {
        "name": "r/TextToSpeech (New Posts)",
        "url": "https://www.reddit.com/r/TextToSpeech/new.rss",
        "category": "general"
    },
    {
        "name": "r/LocalLLaMA (Kokoro / Local AI)",
        "url": "https://www.reddit.com/r/LocalLLaMA/search.rss?q=kokoro+OR+tts&sort=new&restrict_sr=1",
        "category": "technical"
    },
    {
        "name": "r/audiobooks (TTS / Readers)",
        "url": "https://www.reddit.com/r/audiobooks/search.rss?q=tts+OR+%22text+to+speech%22&sort=new&restrict_sr=1",
        "category": "audiobook"
    },
    {
        "name": "r/NewTubers (Voiceovers)",
        "url": "https://www.reddit.com/r/NewTubers/search.rss?q=voiceover+OR+tts&sort=new&restrict_sr=1",
        "category": "creator"
    },
    {
        "name": "Sitewide ElevenLabs Alternatives",
        "url": "https://www.reddit.com/search.rss?q=%22free+alternative+to+elevenlabs%22&sort=new",
        "category": "elevenlabs"
    }
]

# Keywords that indicate a great opportunity to recommend PureTTS
INTENT_KEYWORDS = [
    "free", "alternative", "unlimited", "browser", "online", "cheap", "no limit",
    "recommend", "elevenlabs", "speechify", "best tts", "voice generator",
    "natural sounding", "private", "local", "kokoro", "voiceover"
]

SEEN_CACHE_FILE = os.path.expanduser("~/.puretts_seen_reddit.json")

def load_seen():
    if os.path.exists(SEEN_CACHE_FILE):
        try:
            with open(SEEN_CACHE_FILE, "r") as f:
                return set(json.load(f))
        except Exception:
            return set()
    return set()

def save_seen(seen_ids):
    try:
        with open(SEEN_CACHE_FILE, "w") as f:
            json.dump(list(seen_ids)[-500:], f)
    except Exception:
        pass

def generate_reply(category, title):
    title_lower = title.lower()
    
    if "elevenlabs" in title_lower or category == "elevenlabs":
        return (
            "If you're looking for a free alternative to ElevenLabs without the 10,000 character limit, "
            "check out https://puretts.com. It runs the open-weight Kokoro-82M model directly in your browser "
            "via WebGPU/WASM, so it's 100% free with no signups, no character caps, and complete client-side privacy."
        )
    elif "browser" in title_lower or "web" in title_lower:
        return (
            "For a pure in-browser option, you can try https://puretts.com. It runs Kokoro-82M locally on your "
            "device using WebGPU/WASM. No accounts needed, no character limit, and you can export 16-bit WAV audio directly."
        )
    elif "kokoro" in title_lower or category == "technical":
        return (
            "If you want to test Kokoro-82M in the browser without installing Python/PyTorch locally, "
            "I built an in-browser WebGPU version with automatic sentence chunking and continuous WAV concatenation: "
            "https://puretts.com. Everything runs 100% client-side in a Web Worker."
        )
    elif category == "creator" or "youtube" in title_lower or "voiceover" in title_lower:
        return (
            "If you need clean AI voiceovers for videos without paying $22/mo for ElevenLabs, "
            "https://puretts.com is completely free with zero character limits. It generates uncompressed "
            "WAV audio you can drop straight into your video editor with no watermarks."
        )
    else:
        return (
            "I'd recommend checking out https://puretts.com. It's an in-browser tool powered by Kokoro-82M. "
            "Because it runs client-side on your device (WebGPU), it's completely free with no character limits, "
            "no signups, and your text is never stored on a server."
        )

def scan_reddit():
    seen = load_seen()
    headers = {"User-Agent": "python:puretts.radar:v1.0 (growth-assistant)"}
    
    print("\n🔍 PureTTS Reddit Radar — Scanning for High-Intent Opportunities...")
    print("=" * 70)
    
    opportunities = []

    for feed in FEEDS:
        print(f"📡 Checking {feed['name']}...")
        req = urllib.request.Request(feed["url"], headers=headers)
        
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                xml_data = resp.read()
                root = ET.fromstring(xml_data)
                
                for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
                    title_elem = entry.find("{http://www.w3.org/2005/Atom}title")
                    link_elem = entry.find("{http://www.w3.org/2005/Atom}link")
                    id_elem = entry.find("{http://www.w3.org/2005/Atom}id")
                    
                    if title_elem is None or link_elem is None:
                        continue
                        
                    title = title_elem.text or ""
                    link = link_elem.attrib.get("href", "")
                    entry_id = id_elem.text if id_elem is not None else link
                    
                    # Only look at actual comment/post threads
                    if "comments/" not in link:
                        continue
                        
                    # Check intent keywords
                    title_lower = title.lower()
                    matches = [k for k in INTENT_KEYWORDS if k in title_lower]
                    
                    if matches and entry_id not in seen:
                        reply = generate_reply(feed["category"], title)
                        opportunities.append({
                            "title": title,
                            "url": link,
                            "feed": feed["name"],
                            "matches": matches,
                            "reply": reply,
                            "id": entry_id
                        })
                        seen.add(entry_id)
        except Exception as e:
            print(f"   ⚠️ Could not fetch feed: {e}")
        
        # Respect Reddit's unauthenticated rate limits
        time.sleep(2.5)

    save_seen(seen)

    print("\n" + "=" * 70)
    if not opportunities:
        print("✅ No new unanswered high-intent posts found right now.")
        print("   (Radar automatically caches viewed posts to prevent duplicate replies).")
        print("   Run this script again in 1–2 hours to catch new threads as they appear!")
    else:
        print(f"🎯 Found {len(opportunities)} Hot Opportunities on Reddit:\n")
        for i, opp in enumerate(opportunities, 1):
            print(f"[{i}] {opp['title']}")
            print(f"    Subreddit Feed : {opp['feed']}")
            print(f"    Keywords Matched: {', '.join(opp['matches'])}")
            print(f"    Thread Link    : \033[94m{opp['url']}\033[0m")
            print(f"    Suggested Reply:\n    \033[92m\"{opp['reply']}\"\033[0m\n")
            print("-" * 70)

if __name__ == "__main__":
    scan_reddit()
