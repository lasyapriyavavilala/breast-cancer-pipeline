"""
Agent 2: Entity Extraction & Summarization
Migrated from IBM watsonx to Anthropic Claude
Updated to support StandaloneScraper's multi-source format
Extracts entities, generates summaries, writes incrementally
"""

import os
import json
import re
from typing import List, Dict, Optional
from urllib.parse import urlparse
from pathlib import Path
import argparse

import dateparser
from anthropic import Anthropic
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


class ArticleExtractor:
    """Extract entities and generate summaries using Anthropic Claude"""
    
    # URL to company mapping
    URL_COMPANY_MAP = {
        "gene.com": "Genentech",
        "roche.com": "Roche",
        "novartis.com": "Novartis",
        "pfizer.com": "Pfizer",
        "astrazeneca.com": "AstraZeneca",
        "lilly.com": "Eli Lilly",
        "merck.com": "Merck",
        "sanofi.com": "Sanofi",
        "gilead.com": "Gilead",
        "bms.com": "BMS",
        "amgen.com": "Amgen",
        "gsk.com": "GSK",
        "bayer.com": "Bayer",
        "takeda.com": "Takeda",
        "boehringer-ingelheim.com": "Boehringer Ingelheim",
        "beigene.com": "BeiGene",
        "seagen.com": "Seagen",
        "sermonixpharma.com": "Sermonix Pharma",
        "janssen.com": "Janssen",
        "johnsonandjohnson.com": "Johnson & Johnson",
        "jnj.com": "Johnson & Johnson",
        "abbvie.com": "AbbVie",
        "abbott.com": "Abbott",
        "biogen.com": "Biogen",
        "celgene.com": "Celgene",
        "regeneron.com": "Regeneron",
        "modernatx.com": "Moderna",
        "nuvalent.com": "Nuvalent",
        "daiichisankyo.com": "Daiichi Sankyo",
        "eisai.com": "Eisai",
    }
    
    # Known breast cancer indications
    BREAST_CANCER_INDICATIONS = [
        "breast cancer", "metastatic breast cancer", "early breast cancer", 
        "advanced breast cancer", "HER2-positive breast cancer", "HER2+ breast cancer",
        "triple negative breast cancer", "TNBC", "hormone receptor positive breast cancer",
        "HR+ breast cancer", "ER+ breast cancer", "PR+ breast cancer",
        "inflammatory breast cancer", "ductal carcinoma", "lobular carcinoma",
        "locally advanced breast cancer", "HER2-low breast cancer",
        "HER2-negative breast cancer", "HER2- breast cancer"
    ]
    
    # Roman numeral mapping for trial phases
    ROMAN_MAP = {"I": "1", "II": "2", "III": "3", "IV": "4", "V": "5"}
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        temperature: float = 0.3,
        max_tokens: int = 300
    ):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
        
        self.client = Anthropic(api_key=self.api_key)
        logger.info(f"Anthropic client initialized with model: {self.model}")
    
    def _call_claude(self, prompt: str) -> str:
        """Call Claude API and return response"""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return ""
    
    @staticmethod
    def trim_tweet(text: str, max_chars: int = 280) -> str:
        """Trim text to Twitter length"""
        t = re.sub(r"\s+", " ", text.strip())
        if len(t) <= max_chars:
            return t
        cut = t[:max_chars]
        idx = max(cut.rfind("."), cut.rfind(";"), cut.rfind(","), cut.rfind(" "))
        if idx > max_chars * 0.6:
            return cut[:idx].rstrip()
        return cut.rstrip()
    
    def company_from_url(self, url: str, source: Optional[str] = None) -> Optional[str]:
        """
        Extract company name from URL or use source field from scraper.
        
        Args:
            url: Article URL
            source: Source name from StandaloneScraper (preferred)
        
        Returns:
            Company name or None
        """
        # If source is provided by scraper, use it directly
        if source and isinstance(source, str) and source.strip():
            # Clean up common news site suffixes
            source_clean = source.strip()
            for suffix in [" News", " Newsroom", " Press Releases", " Media", " - Press Release"]:
                if source_clean.endswith(suffix):
                    source_clean = source_clean[:-len(suffix)].strip()
            return source_clean
        
        # Fallback: extract from URL
        try:
            host = urlparse(url).netloc.lower()
            for prefix in ("www.", "amp.", "m.", "news.", "media.", "investor."):
                if host.startswith(prefix):
                    host = host[len(prefix):]
            
            if host in self.URL_COMPANY_MAP:
                return self.URL_COMPANY_MAP[host]
            
            parts = host.split(".")
            brand = parts[-2] if len(parts) >= 2 else parts[0]
            brand = brand.replace("-", " ").strip()
            return brand.capitalize() if brand else None
        except Exception:
            return None
    
    def extract_publication_date(
        self, 
        content: str, 
        headline: str, 
        scraped_date: Optional[str] = None
    ) -> Optional[str]:
        """
        Extract publication date from article.
        
        Args:
            content: Article content
            headline: Article headline
            scraped_date: Publication date from StandaloneScraper (preferred)
        
        Returns:
            Date in YYYY-MM-DD format or None
        """
        # If scraper already extracted date, use it
        if scraped_date and isinstance(scraped_date, str) and scraped_date.strip():
            # Validate format
            if re.match(r'^\d{4}-\d{2}-\d{2}', scraped_date):
                return scraped_date[:10]  # Take just YYYY-MM-DD part
            # Try parsing it
            parsed = dateparser.parse(scraped_date)
            if parsed:
                return parsed.strftime('%Y-%m-%d')
        
        # Fallback: extract from content
        text_for_date = (headline + " " + content[:2000])
        
        # Try regex patterns first
        date_patterns = [
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b',
            r'\b\d{4}-\d{2}-\d{2}\b',
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text_for_date, re.IGNORECASE)
            for match in matches:
                parsed_date = dateparser.parse(match)
                if parsed_date:
                    return parsed_date.strftime('%Y-%m-%d')
        
        # Claude fallback
        try:
            prompt = f"""Extract the publication date from the following text. Return ONLY the date in YYYY-MM-DD format or "Not found" if no clear date.

TEXT:
\"\"\"{text_for_date[:3000]}\"\"\"

Date (YYYY-MM-DD or "Not found"):"""
            
            response = self._call_claude(prompt)
            if response and response != "Not found" and re.match(r'^\d{4}-\d{2}-\d{2}$', response):
                return response
        except Exception:
            pass
        
        return None
    
    def extract_indication(self, headline: str, content: str) -> List[str]:
        """Extract disease indications"""
        text_all = (headline + " " + content).lower()
        found = []
        
        # Check known indications
        for indication in self.BREAST_CANCER_INDICATIONS:
            if indication.lower() in text_all:
                found.append(indication)
        
        # Claude assist for other mentions
        try:
            prompt = f"""Extract all specific disease indications, cancer subtypes, or medical conditions mentioned in this text.
Return ONLY a JSON array of strings with the specific indications found.

TEXT:
\"\"\"{text_all[:4000]}\"\"\"

JSON array:"""
            
            response = self._call_claude(prompt)
            match = re.search(r'\[.*\]', response, flags=re.S)
            if match:
                llm_list = json.loads(match.group(0))
                if isinstance(llm_list, list):
                    for ind in llm_list:
                        if isinstance(ind, str) and ind.strip():
                            if any(k in ind.lower() for k in ['cancer', 'carcinoma', 'tumor', 'metastatic', 'advanced']):
                                found.append(ind.strip())
        except Exception:
            pass
        
        # Deduplicate
        seen = set()
        out = []
        for x in found:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out
    
    def extract_drug_names(self, headline: str, content: str) -> List[str]:
        """Extract drug/product names (excluding therapy classes)"""
        snippet = (headline + "\n" + content).strip()
        if len(snippet) > 5000:
            snippet = snippet[:5000]
        
        prompt = f"""Extract drug or product names that appear **verbatim** in the text below.

RULES:
- Return ONLY names that literally appear in the text (headline or content).
- If a brand and its generic appear like "Itovebi (inavolisib)", return BOTH as separate strings.
- EXCLUDE therapy class terms like "chemotherapy", "immunotherapy", "targeted therapy", "radiotherapy", etc.
- Return a pure JSON array of strings (no commentary).

TEXT:
\"\"\"{snippet}\"\"\"

Return JSON array:"""
        
        raw = self._call_claude(prompt)
        match = re.search(r"\[\s*(?:\".*?\")\s*(?:,.*?)*\]", raw, flags=re.S)
        items = []
        if match:
            try:
                items = json.loads(match.group(0))
            except Exception:
                items = []
        
        if not isinstance(items, list):
            items = []
        items = [s.strip() for s in items if isinstance(s, str) and s.strip()]
        
        # Validate literal presence
        low = snippet.lower()
        banned = {"chemotherapy", "immunotherapy", "endocrine therapy", "hormonal therapy",
                  "targeted therapy", "radiation therapy", "radiotherapy", "adjuvant therapy"}
        validated = []
        for name in items:
            norm = re.sub(r"[™®]", "", name).strip()
            if norm and (norm.lower() in low) and (norm.lower() not in banned):
                validated.append(name)
        
        # Deduplicate
        seen = set()
        out = []
        for x in validated:
            k = x.lower()
            if k not in seen:
                seen.add(k)
                out.append(x)
        return out
    
    def extract_trial_phase(self, headline: str, content: str) -> List[str]:
        """Extract and normalize trial phases"""
        text = f"{headline}\n{content}"
        phases = set()
        
        # Regex patterns
        patterns = [
            r'\b[Pp]hase\s*(I{1,3}V?|V|1|2|3|4)(?:\s*[/\-]\s*(I{1,3}V?|V|1|2|3|4))?\s*([a-dA-D])?\b',
            r'\b[Pp](?:h|H)?\s*(\d)(?:\s*/\s*(\d))?\b',
        ]
        
        for pat in patterns:
            for m in re.finditer(pat, text):
                grp = [g for g in m.groups() if g]
                if not grp:
                    continue
                nums = []
                suffix = ""
                for g in grp:
                    G = g.upper()
                    if G in self.ROMAN_MAP:
                        nums.append(self.ROMAN_MAP[G])
                    elif re.fullmatch(r"\d", G):
                        nums.append(G)
                    elif re.fullmatch(r"[A-D]", G):
                        suffix = G.lower()
                
                if len(nums) == 1:
                    phases.add(f"Phase {nums[0]}{suffix}")
                elif len(nums) >= 2:
                    phases.add(f"Phase {nums[0]}/{nums[1]}{suffix}")
        
        # Claude fallback
        if not phases:
            try:
                snippet = text[:4500]
                prompt = f"""From the text, extract the clinical trial PHASE if mentioned.
Return ONLY a JSON array of normalized strings like "Phase 1", "Phase 2/3", "Phase 3b".
If none, return [].

TEXT:
\"\"\"{snippet}\"\"\"

JSON array:"""
                
                resp = self._call_claude(prompt)
                mm = re.search(r'\[.*\]', resp, flags=re.S)
                if mm:
                    arr = json.loads(mm.group(0))
                    if isinstance(arr, list):
                        for x in arr:
                            if isinstance(x, str) and x.strip().lower().startswith("phase"):
                                phases.add(x.strip())
            except Exception:
                pass
        
        out = []
        seen = set()
        for p in phases:
            if p not in seen:
                seen.add(p)
                out.append(p)
        return out
    
    def extract_trial_names(self, headline: str, content: str) -> List[str]:
        """Extract proper study names (e.g., KEYNOTE-522)"""
        text = f"{headline}\n{content}"
        names = set()
        
        # Strict patterns requiring letters
        pats = [
            r'[""]?([A-Z][A-Z0-9]+(?:[-–][A-Za-z0-9]+){0,3})[""]?\s+(?:trial|study)\b',
            r'\b(?:trial|study)\s+[""]?([A-Z][A-Z0-9]+(?:[-–][A-Za-z0-9]+){0,3})[""]?\b',
        ]
        
        for pat in pats:
            for m in re.finditer(pat, text):
                cand = re.sub(r'\s+', ' ', m.group(1).strip())
                if re.search(r'[A-Za-z]', cand) and not re.fullmatch(r'[IVX]+', cand, flags=re.I):
                    names.add(cand)
        
        # Claude fallback
        try:
            snippet = text[:4500]
            prompt = f"""If the text mentions a named clinical study/trial (e.g., KEYNOTE-522, TROPION-Breast01), return ONLY a pure JSON array of those names.
If none, return [].

TEXT:
\"\"\"{snippet}\"\"\"

JSON array:"""
            
            resp = self._call_claude(prompt)
            mm = re.search(r'\[.*\]', resp, flags=re.S)
            if mm:
                arr = json.loads(mm.group(0))
                if isinstance(arr, list):
                    for cand in arr:
                        if isinstance(cand, str):
                            c = re.sub(r'\s+', ' ', cand.strip())
                            if c and re.search(r'[A-Za-z]', c) and not re.fullmatch(r'[IVX]+', c, flags=re.I):
                                names.add(c)
        except Exception:
            pass
        
        out = []
        seen = set()
        for n in names:
            if n not in seen:
                seen.add(n)
                out.append(n)
        return out
    
    def generate_summary_280(self, headline: str, content: str) -> str:
        """Generate tweet-length summary"""
        snippet = (headline + "\n" + content).strip()
        if len(snippet) > 6000:
            snippet = snippet[:6000]
        
        prompt = f"""Write ONE tweet-style summary (<= 280 characters) strictly based on the article text below.
REQUIREMENTS:
- Mention concrete specifics present in the text (e.g., drug name(s), indication, setting/line, and endpoint if mentioned).
- No emojis, no hashtags, no marketing language, no invented details.
- One concise sentence or two short clauses.

TEXT:
\"\"\"{snippet}\"\"\"

Tweet (<=280 chars):"""
        
        raw = self._call_claude(prompt)
        tweet = raw.splitlines()[0].strip() if raw else ""
        return self.trim_tweet(tweet, max_chars=280)
    
    def process_article(self, article: Dict) -> Dict:
        """
        Process single article from StandaloneScraper format.
        
        Args:
            article: Article dict from Agent 1 (StandaloneScraper)
                Expected fields:
                - title: Article headline
                - url: Article URL
                - content: Full article text
                - source: Source name (e.g., "Pfizer", "Roche")
                - publication_date: Date string (optional)
                - content_type: "rss", "api", "scrape"
        
        Returns:
            Enhanced article dict for Agent 3
        """
        # Handle both old and new field names
        headline = (
            article.get("title") or 
            article.get("headline") or 
            ""
        ).strip()
        
        url = (
            article.get("url") or 
            article.get("link") or 
            ""
        ).strip()
        
        content = (article.get("content") or "").strip()
        
        # NEW: Get source and publication_date from scraper
        source = article.get("source")
        scraped_date = article.get("publication_date")
        content_type = article.get("content_type", "unknown")
        
        logger.info(f"Processing: {headline[:80]}... [{source or 'unknown'}]")
        
        # Extract all fields (with scraper-provided data as preferred)
        published_date = self.extract_publication_date(content, headline, scraped_date)
        indications = self.extract_indication(headline, content)
        drug_names = self.extract_drug_names(headline, content)
        trial_phases = self.extract_trial_phase(headline, content)
        trial_names = self.extract_trial_names(headline, content)
        summary = self.generate_summary_280(headline, content)
        company_name = self.company_from_url(url, source)
        
        # Build output record (schema for Agent 3 & 4)
        record = {
            "published_date": published_date,
            "content": content,
            "entities": {
                "drug_names": drug_names,
                "company_name": company_name,  # Single string (not list)
                "trial_phases": trial_phases,
                "trial_names": trial_names,
                "indications": list(dict.fromkeys(indications))  # deduplicate
            },
            "summary_280": summary,
            "url": url,
            "headline": headline,
            # Preserve scraper metadata
            "source": source,
            "content_type": content_type,
            "scraped_at": article.get("scraped_at")
        }
        
        return record
    
    def process_articles_batch(
        self,
        articles: List[Dict],
        output_json: str = "data/processed/enhanced_articles.json",
        output_ndjson: str = "data/processed/enhanced_articles.ndjson",
        incremental: bool = True
    ) -> List[Dict]:
        """
        Process batch of articles with optional incremental writes.
        
        Args:
            articles: List of article dicts from Agent 1 (StandaloneScraper)
            output_json: Path for final JSON array
            output_ndjson: Path for line-delimited JSON
            incremental: Write after each article
        
        Returns:
            List of processed articles
        """
        processed = []
        
        # Ensure output directory exists
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        
        # Start fresh NDJSON file
        if incremental:
            open(output_ndjson, "w", encoding="utf-8").close()
        
        for i, article in enumerate(articles, 1):
            try:
                result = self.process_article(article)
                processed.append(result)
                
                # Incremental write
                if incremental:
                    # Append to NDJSON
                    with open(output_ndjson, "a", encoding="utf-8") as f:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        f.flush()
                        os.fsync(f.fileno())
                    
                    # Overwrite JSON array atomically
                    tmp = output_json + ".tmp"
                    with open(tmp, "w", encoding="utf-8") as f:
                        json.dump(processed, f, ensure_ascii=False, indent=2)
                        f.flush()
                        os.fsync(f.fileno())
                    os.replace(tmp, output_json)
                
                if i % 10 == 0:
                    logger.info(f"Processed {i}/{len(articles)} articles")
            
            except Exception as e:
                logger.error(f"Error processing article {i}: {e}")
                continue
        
        # Final write if not incremental
        if not incremental:
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(processed, f, ensure_ascii=False, indent=2)
        
        logger.success(f"✅ Processed {len(processed)}/{len(articles)} articles")
        self._print_stats(processed)
        
        return processed
    
    def _print_stats(self, processed: List[Dict]):
        """Print extraction statistics"""
        from collections import Counter
        
        logger.info("=" * 50)
        logger.info("EXTRACTION STATS")
        logger.info("=" * 50)
        logger.info(f"Total processed: {len(processed)}")
        logger.info(f"With published date: {sum(1 for a in processed if a.get('published_date'))}")
        logger.info(f"With drug names: {sum(1 for a in processed if a['entities'].get('drug_names'))}")
        logger.info(f"With company name: {sum(1 for a in processed if a['entities'].get('company_name'))}")
        logger.info(f"With trial phases: {sum(1 for a in processed if a['entities'].get('trial_phases'))}")
        logger.info(f"With trial names: {sum(1 for a in processed if a['entities'].get('trial_names'))}")
        logger.info(f"With indications: {sum(1 for a in processed if a['entities'].get('indications'))}")
        
        # Content type distribution
        content_types = [a.get("content_type") for a in processed if a.get("content_type")]
        if content_types:
            logger.info("\nBy Content Type:")
            for ct, count in Counter(content_types).most_common():
                logger.info(f"  {ct}: {count}")
        
        # Source distribution
        sources = [a.get("source") for a in processed if a.get("source")]
        if sources:
            logger.info("\nBy Source:")
            for src, count in Counter(sources).most_common(10):
                logger.info(f"  {src}: {count}")


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(description="Agent 2: Entity Extraction & Summarization")
    parser.add_argument("--input", required=True, help="Input JSON from Agent 1 (StandaloneScraper)")
    parser.add_argument("--output-json", default="data/processed/enhanced_articles.json",
                        help="Output JSON array path")
    parser.add_argument("--output-ndjson", default="data/processed/enhanced_articles.ndjson",
                        help="Output NDJSON path")
    parser.add_argument("--no-incremental", action="store_true",
                        help="Disable incremental writes")
    
    args = parser.parse_args()
    
    # Load articles
    logger.info(f"Loading articles from {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        articles = json.load(f)
    
    logger.info(f"Loaded {len(articles)} articles")
    
    # Initialize extractor
    extractor = ArticleExtractor()
    
    # Process
    processed = extractor.process_articles_batch(
        articles,
        output_json=args.output_json,
        output_ndjson=args.output_ndjson,
        incremental=not args.no_incremental
    )
    
    logger.success(f"✅ Agent 2 complete: {args.output_json}")
    print(f"OUTPUT_FILE={args.output_json}")


if __name__ == "__main__":
    main()