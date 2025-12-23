#!/usr/bin/env python3
"""
Setup Script

Initialize the FAISS index with papers from LLMSys repository.
"""

import re
import sys
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import get_settings
from core.logging import get_logger
from services.embedding_service import EmbeddingService

logger = get_logger("setup")


def fetch_llm_sys_papers() -> pd.DataFrame:
    """Fetch papers from LLMSys repository."""
    logger.info("Fetching papers from LLMSys repository...")
    
    url = "https://raw.githubusercontent.com/AmberLJC/LLMSys-PaperList/main/README.md"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    
    lines = response.text.splitlines()
    
    section_pattern = re.compile(r'^##\s+(.*)')
    subsection_pattern = re.compile(r'^###\s+(.*)')
    link_pattern = re.compile(r'- \[(.*?)\]\((.*?)\)')
    
    entries = []
    current_section = None
    current_subsection = None
    
    for line in lines:
        line = line.strip()
        
        sec_match = section_pattern.match(line)
        if sec_match:
            current_section = sec_match.group(1)
            current_subsection = None
            continue
        
        sub_match = subsection_pattern.match(line)
        if sub_match:
            current_subsection = sub_match.group(1)
            continue
        
        link_match = link_pattern.match(line)
        if link_match and current_section:
            title, url = link_match.groups()
            if url.startswith("http"):
                entries.append({
                    "section": current_section,
                    "subsection": current_subsection or "",
                    "title": title,
                    "url": url
                })
    
    df = pd.DataFrame(entries)
    logger.info(f"Found {len(df)} papers")
    return df


def fetch_arxiv_abstracts(urls: list[str]) -> list[str]:
    """Fetch abstracts from arXiv."""
    logger.info(f"Fetching abstracts for {len(urls)} papers...")
    
    abstracts = []
    for i, url in enumerate(urls):
        if i % 10 == 0:
            logger.info(f"Progress: {i}/{len(urls)}")
        
        # Extract arXiv ID
        match = re.search(r'arxiv\.org/(?:abs|pdf)/([0-9]+\.[0-9]+(?:v[0-9]+)?)', url)
        if not match:
            abstracts.append("")
            continue
        
        arxiv_id = match.group(1)
        api_url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}&max_results=1"
        
        try:
            response = requests.get(api_url, timeout=30)
            response.raise_for_status()
            
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            root = ET.fromstring(response.content)
            entry = root.find('atom:entry', ns)
            
            if entry is not None:
                summary = entry.find('atom:summary', ns)
                if summary is not None and summary.text:
                    abstracts.append(summary.text.strip())
                    continue
        except Exception as e:
            logger.warning(f"Failed to fetch {arxiv_id}: {e}")
        
        abstracts.append("")
    
    logger.info(f"Retrieved {sum(1 for a in abstracts if a)} abstracts")
    return abstracts


def main():
    """Run the setup process."""
    print("=" * 60)
    print("🚀 LLM Teaching Assistant - Setup")
    print("=" * 60)
    
    settings = get_settings()
    
    # Step 1: Fetch papers
    print("\n📚 Step 1: Fetching papers...")
    df = fetch_llm_sys_papers()
    
    # Filter arXiv URLs
    arxiv_mask = df['url'].str.contains(r'arxiv\.org/(abs|pdf)', na=False, regex=True)
    arxiv_urls = df.loc[arxiv_mask, 'url'].tolist()
    print(f"   Found {len(arxiv_urls)} arXiv papers")
    
    # Step 2: Fetch abstracts
    print("\n📝 Step 2: Fetching abstracts...")
    abstracts = fetch_arxiv_abstracts(arxiv_urls)
    
    # Filter out empty abstracts
    valid_pairs = [(url, abstract) for url, abstract in zip(arxiv_urls, abstracts) if abstract]
    valid_urls = [p[0] for p in valid_pairs]
    valid_abstracts = [p[1] for p in valid_pairs]
    print(f"   Valid abstracts: {len(valid_abstracts)}")
    
    # Step 3: Create embeddings
    print("\n🔢 Step 3: Creating embeddings...")
    embedding_service = EmbeddingService()
    embeddings = embedding_service.create_embeddings_batch(valid_abstracts)
    print(f"   Created {len(embeddings)} embeddings")
    
    # Step 4: Build FAISS index
    print("\n📊 Step 4: Building FAISS index...")
    embedding_service.build_index(embeddings, valid_urls)
    print(f"   Index saved to: {settings.faiss_index_path}")
    print(f"   URLs saved to: {settings.urls_json_path}")
    
    print("\n" + "=" * 60)
    print("✅ Setup complete!")
    print("=" * 60)
    print(f"\nYou can now run the server:")
    print("  uvicorn api.main:app --reload")
    print("\nOr test with:")
    print("  curl http://localhost:8000/health")


if __name__ == "__main__":
    main()
