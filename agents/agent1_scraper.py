"""
Standalone Multi-Source Breast Cancer News Scraper
Works independently - just needs CSV files and environment variables
"""

import os
import json
import time
import random
import requests
import re
import io
from pathlib import Path
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Optional
from loguru import logger
import pandas as pd
import feedparser
import PyPDF2
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium_stealth import stealth
from dotenv import load_dotenv


class StandaloneScraper:
    """
    Complete standalone scraper for breast cancer news
    Handles RSS feeds, APIs, and traditional web scraping
    """
    
    USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    ]
    
    def __init__(self, urls_csv: str, keywords_csv: str, days_back: int = 90, 
             target_articles: int = 50, output_dir: str = "output"):
            
            """
            Initialize scraper with CSV files.
            
            CSV Format (flexible - handles extra columns):
            Required columns: url, source_name, source_type, priority
            Optional columns: tier, category, notes (will be ignored)
            """
            self.days_back = days_back
            self.target_articles = target_articles
            self.output_dir = output_dir
            
            # Load sources - UPDATED to handle extra columns
            try:
                self.sources_df = pd.read_csv(urls_csv)
                
                # Validate required columns exist
                required_cols = ['url', 'source_name', 'source_type', 'priority']
                missing_cols = [col for col in required_cols if col not in self.sources_df.columns]
                
                if missing_cols:
                    raise ValueError(f"Missing required columns in {urls_csv}: {missing_cols}")
                
                # Keep only required columns (ignore tier, category, notes)
                self.sources_df = self.sources_df[required_cols]
                
                # Sort by priority
                self.sources_df = self.sources_df.sort_values('priority', ascending=False)
                
                logger.info(f"Loaded {len(self.sources_df)} sources from {urls_csv}")
                
            except Exception as e:
                logger.error(f"Failed to load URLs CSV: {e}")
                raise
            
            # Load keywords
            try:
                keywords_df = pd.read_csv(keywords_csv)
                self.keywords = keywords_df['keyword'].str.lower().tolist()
                logger.info(f"Loaded {len(self.keywords)} keywords")
            except Exception as e:
                logger.error(f"Failed to load keywords CSV: {e}")
                raise
            
            # Rest of initialization...
            self.cutoff_date = datetime.now() - timedelta(days=days_back)
            self.articles = []
            self.driver = None
            
            # Stats
            self.stats = {
                'total_scraped': 0,
                'by_source': {},
                'by_type': {'rss': 0, 'api': 0, 'scrape': 0}
            }
            
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            logger.info(f"Scraper initialized: target={target_articles}, days_back={days_back}")

            
    def _load_keywords(self, keywords_csv: str) -> List[str]:
        """Load keywords from CSV"""
        try:
            df = pd.read_csv(keywords_csv)
            keywords = df['keyword'].dropna().str.lower().tolist()
            return keywords
        except Exception as e:
            print(f"⚠️  Error loading keywords: {e}")
            return ["breast cancer", "her2", "tnbc", "triple negative"]
    
    def _setup_driver(self) -> webdriver.Chrome:
        """Setup headless Chrome with stealth mode"""
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument(f"user-agent={random.choice(self.USER_AGENTS)}")
        chrome_options.add_argument("--window-size=1920,1080")
        
        try:
            # Try system chromedriver first
            service = Service()
            driver = webdriver.Chrome(service=service, options=chrome_options)
        except Exception:
            # Fallback to specific path (update as needed)
            service = Service(executable_path="C:\\Chromedriver\\chromedriver.exe")
            driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # Apply stealth settings
        stealth(
            driver,
            languages=["en-US", "en"],
            vendor="Google Inc.",
            platform="Win32",
            webgl_vendor="Intel Inc.",
            renderer="Intel Iris OpenGL Engine",
            fix_hairline=True,
        )
        
        driver.set_page_load_timeout(30)
        return driver
            
    def scrape_all(self) -> List[Dict]:
        """Main scraping method - handles all source types"""
        
        print("\n" + "=" * 70)
        print("🔍 STARTING MULTI-SOURCE SCRAPE")
        print("=" * 70)
        
        for idx, source in self.sources_df.iterrows():
            if len(self.articles) >= self.target_articles:
                print(f"\n✅ Target of {self.target_articles} articles reached!")
                break
            
            source_name = source['source_name']
            source_type = source['source_type']
            url = source['url']
            priority = source['priority']
            
            print(f"\n[Priority {priority}] {source_name} ({source_type})")
            print(f"   URL: {url}")
            
            try:
                if source_type == 'rss':
                    results = self._scrape_rss(url, source_name)
                elif source_type == 'api':
                    results = self._scrape_api(url, source_name)
                else:  # scrape
                    results = self._scrape_website(url, source_name)
                
                # Filter by keywords and date
                filtered = self._filter_articles(results)
                self.articles.extend(filtered)
                
                print(f"   ✓ Found {len(filtered)} relevant articles (Total: {len(self.articles)}/{self.target_articles})")
                
                # Rate limiting
                time.sleep(random.uniform(1, 3))
                
            except Exception as e:
                print(f"   ✗ Error: {e}")
        
        print("\n" + "=" * 70)
        print(f"✅ SCRAPING COMPLETE: {len(self.articles)} articles collected")
        print("=" * 70)
        
        return self.articles
    
    # ==================== RSS SCRAPING ====================
    
    def _scrape_rss(self, url: str, source_name: str) -> List[Dict]:
        """Scrape RSS feed"""
        articles = []
        
        try:
            feed = feedparser.parse(url)
            
            for entry in feed.entries[:20]:  # Limit per feed
                # Extract date
                pub_date = None
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    pub_date = datetime(*entry.published_parsed[:6])
                elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                    pub_date = datetime(*entry.updated_parsed[:6])
                
                article = {
                    'source': source_name,
                    'url': entry.get('link', ''),
                    'title': entry.get('title', ''),
                    'content': entry.get('summary', entry.get('description', '')),
                    'publication_date': pub_date.strftime('%Y-%m-%d') if pub_date else None,
                    'content_type': 'rss',
                    'scraped_at': datetime.now().isoformat()
                }
                
                articles.append(article)
            
        except Exception as e:
            print(f"   RSS error: {e}")
        
        return articles
    
    # ==================== API SCRAPING ====================
    
    def _scrape_api(self, url: str, source_name: str) -> List[Dict]:
        """Scrape API endpoints"""
        
        if 'clinicaltrials.gov' in url:
            return self._scrape_clinicaltrials()
        elif 'medrxiv' in url or 'biorxiv' in url:
            return self._scrape_preprints(url, source_name)
        
        return []
    
    def _scrape_clinicaltrials(self) -> List[Dict]:
        """Scrape ClinicalTrials.gov API"""
        articles = []
        
        try:
            url = "https://clinicaltrials.gov/api/v2/studies"
            
            params = {
                "query.cond": "Breast Cancer",
                "query.intr": "Drug",
                "filter.lastUpdatePostDate": f"{(datetime.now() - timedelta(days=self.days_back)).strftime('%Y-%m-%d')},{datetime.now().strftime('%Y-%m-%d')}",
                "pageSize": 20,
                "format": "json"
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            for study in data.get('studies', []):
                protocol = study.get('protocolSection', {})
                ident = protocol.get('identificationModule', {})
                desc = protocol.get('descriptionModule', {})
                status = protocol.get('statusModule', {})
                
                nct_id = ident.get('nctId', '')
                
                article = {
                    'source': 'ClinicalTrials.gov',
                    'url': f"https://clinicaltrials.gov/study/{nct_id}",
                    'title': ident.get('briefTitle', ''),
                    'content': desc.get('briefSummary', ''),
                    'publication_date': status.get('lastUpdatePostDate', ''),
                    'content_type': 'clinical_trial',
                    'scraped_at': datetime.now().isoformat()
                }
                
                articles.append(article)
                
        except Exception as e:
            print(f"   ClinicalTrials.gov error: {e}")
        
        return articles
    
    def _scrape_preprints(self, url: str, source_name: str) -> List[Dict]:
        """Scrape preprint servers (medRxiv, bioRxiv)"""
        articles = []
        
        # This would require specific API implementation
        # Placeholder for now
        print(f"   Preprint scraping not yet implemented for {source_name}")
        
        return articles
    
    # ==================== WEB SCRAPING ====================
    
    def _scrape_website(self, url: str, source_name: str) -> List[Dict]:
        """Traditional web scraping"""
        articles = []
        
        try:
            # Initialize driver if needed
            if self.driver is None:
                self.driver = self._setup_driver()
            
            self.driver.get(url)
            time.sleep(3)
            
            # Scroll to load dynamic content
            for _ in range(2):
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(1)
            
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            
            # Find article links
            links = self._find_article_links(soup, url)
            
            # Extract content from each link (limit to avoid timeouts)
            for link in links[:5]:  # Limit per source
                if len(articles) >= 5:
                    break
                
                content, pub_date = self._extract_article_content(link['url'])
                
                if content:
                    article = {
                        'source': source_name,
                        'url': link['url'],
                        'title': link['title'],
                        'content': content,
                        'publication_date': pub_date.strftime('%Y-%m-%d') if pub_date else None,
                        'content_type': link['type'],
                        'scraped_at': datetime.now().isoformat()
                    }
                    articles.append(article)
                
                time.sleep(random.uniform(1, 2))
        
        except Exception as e:
            print(f"   Web scraping error: {e}")
        
        return articles
    
    def _find_article_links(self, soup: BeautifulSoup, base_url: str) -> List[Dict]:
        """Find article links on a page"""
        links = []
        seen = set()
        domain = urlparse(base_url).netloc
        
        for a in soup.find_all('a', href=True):
            href = a['href']
            text = a.get_text(strip=True).lower()
            title_attr = a.get('title', '').lower()
            
            # Check if link text or title contains keywords
            combined_text = text + ' ' + title_attr
            if not any(kw in combined_text for kw in self.keywords):
                continue
            
            # Build full URL
            full_url = urljoin(base_url, href)
            
            # Only same domain, avoid duplicates
            if urlparse(full_url).netloc != domain:
                continue
            
            if full_url in seen:
                continue
            
            seen.add(full_url)
            
            links.append({
                'url': full_url,
                'title': a.get_text(strip=True)[:200],
                'type': 'pdf' if self._is_pdf_url(full_url) else 'html'
            })
        
        return links
    
    def _extract_article_content(self, url: str) -> tuple:
        """Extract content and date from article URL"""
        
        if self._is_pdf_url(url):
            content = self._extract_pdf_text(url)
            date = self._extract_date_from_text(content)
            return content, date
        
        try:
            self.driver.get(url)
            time.sleep(2)
            
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            date = self._extract_date(soup, self.driver.page_source)
            
            # Try common article selectors
            content = None
            for selector in ['article', 'main', '.content', '.article-body', '.post-content', '.entry-content']:
                el = soup.select_one(selector)
                if el:
                    content = el.get_text(' ', strip=True)
                    break
            
            if not content:
                # Fallback: get all text
                content = soup.get_text(' ', strip=True)
            
            return content, date
            
        except Exception as e:
            print(f"      Content extraction failed: {e}")
            return "", None
    
    def _is_pdf_url(self, url: str) -> bool:
        """Check if URL points to a PDF"""
        return url.lower().endswith('.pdf') or 'pdf' in urlparse(url).query.lower()
    
    def _extract_pdf_text(self, url: str) -> str:
        """Extract text from PDF URL"""
        try:
            headers = {'User-Agent': random.choice(self.USER_AGENTS)}
            r = requests.get(url, headers=headers, timeout=30)
            r.raise_for_status()
            
            reader = PyPDF2.PdfReader(io.BytesIO(r.content))
            text = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
            
            return '\n'.join(text).strip()
            
        except Exception as e:
            print(f"      PDF extraction failed: {e}")
            return ""
    
    def _extract_date(self, soup: BeautifulSoup, html: str) -> Optional[datetime]:
        """Extract publication date from HTML"""
        
        # Try <time> tag first
        time_tag = soup.find('time')
        if time_tag:
            datetime_attr = time_tag.get('datetime')
            if datetime_attr:
                try:
                    # Handle various ISO formats
                    date_str = datetime_attr[:10]  # Get YYYY-MM-DD
                    return datetime.fromisoformat(date_str)
                except:
                    pass
        
        # Try common meta tags
        for meta_name in ['article:published_time', 'datePublished', 'date', 'publish-date']:
            meta = soup.find('meta', property=meta_name) or soup.find('meta', attrs={'name': meta_name})
            if meta and meta.get('content'):
                try:
                    date_str = meta['content'][:10]
                    return datetime.fromisoformat(date_str)
                except:
                    pass
        
        # Try regex patterns
        patterns = [
            r'(\d{4}-\d{2}-\d{2})',  # YYYY-MM-DD
            r'(\w+ \d{1,2},? \d{4})',  # Month DD, YYYY
            r'(\d{1,2} \w+ \d{4})'  # DD Month YYYY
        ]
        
        for pattern in patterns:
            match = re.search(pattern, html)
            if match:
                try:
                    date_str = match.group(1).replace(',', '')
                    # Try different formats
                    for fmt in ['%Y-%m-%d', '%B %d %Y', '%d %B %Y']:
                        try:
                            return datetime.strptime(date_str, fmt)
                        except:
                            continue
                except:
                    pass
        
        return None
    
    def _extract_date_from_text(self, text: str) -> Optional[datetime]:
        """Extract date from plain text"""
        if not text:
            return None
        
        patterns = [
            r'(\d{4}-\d{2}-\d{2})',
            r'(\w+ \d{1,2},? \d{4})',
            r'(\d{1,2} \w+ \d{4})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text[:2000])  # Check first 2000 chars
            if match:
                try:
                    date_str = match.group(1).replace(',', '')
                    for fmt in ['%Y-%m-%d', '%B %d %Y', '%d %B %Y']:
                        try:
                            return datetime.strptime(date_str, fmt)
                        except:
                            continue
                except:
                    pass
        
        return None
    
    # ==================== FILTERING ====================
    
    def _filter_articles(self, articles: List[Dict]) -> List[Dict]:
        """Filter articles by keywords and date"""
        filtered = []
        
        for article in articles:
            # Check content length
            content = article.get('content', '')
            if len(content) < 100:
                continue
            
            # Check keywords
            text = (article.get('title', '') + ' ' + content).lower()
            if not any(kw in text for kw in self.keywords):
                continue
            
            # Check date
            pub_date_str = article.get('publication_date')
            if pub_date_str:
                try:
                    pub_date = datetime.fromisoformat(pub_date_str)
                    if pub_date < self.cutoff_date:
                        continue
                except:
                    pass
            
            filtered.append(article)
        
        return filtered
    
    # ==================== SAVE RESULTS ====================
    
    def save(self, filename: Optional[str] = None) -> str:
        """Save articles to JSON file"""
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"breast_cancer_articles_{timestamp}.json"
        
        # FIX: Ensure output_dir is a Path object
        filepath = Path(self.output_dir) / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.articles, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Saved {len(self.articles)} articles → {filepath}")
        
        # Print statistics
        self._print_stats()
        
        return str(filepath)
    
    def _print_stats(self):
        """Print collection statistics"""
        
        print("\n" + "=" * 70)
        print("📊 COLLECTION STATISTICS")
        print("=" * 70)
        
        # By source
        sources = {}
        for article in self.articles:
            source = article['source']
            sources[source] = sources.get(source, 0) + 1
        
        print("\nBy Source:")
        for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
            print(f"  {source}: {count}")
        
        # By content type
        types = {}
        for article in self.articles:
            ctype = article['content_type']
            types[ctype] = types.get(ctype, 0) + 1
        
        print("\nBy Type:")
        for ctype, count in types.items():
            print(f"  {ctype}: {count}")
        
        # With dates
        with_dates = sum(1 for a in self.articles if a.get('publication_date'))
        print(f"\nArticles with dates: {with_dates}/{len(self.articles)}")
        
        print("=" * 70)
    
    def cleanup(self):
        """Cleanup resources"""
        if self.driver:
            self.driver.quit()
            print("\n🔒 Browser closed")


# ==================== MAIN FUNCTION ====================

def main():
    """Standalone CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Standalone Breast Cancer News Scraper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python standalone_scraper.py --urls urls.csv --keywords keywords.csv
  
  # Custom target and lookback
  python standalone_scraper.py --urls urls.csv --keywords keywords.csv --target 100 --days-back 30
  
  # Save to specific directory
  python standalone_scraper.py --urls urls.csv --keywords keywords.csv --output-dir my_data
        """
    )
    
    parser.add_argument('--urls', required=True, help='Path to URLs CSV file')
    parser.add_argument('--keywords', required=True, help='Path to keywords CSV file')
    parser.add_argument('--target', type=int, default=50, help='Target number of articles (default: 50)')
    parser.add_argument('--days-back', type=int, default=90, help='Days to look back (default: 90)')
    parser.add_argument('--output-dir', default='output', help='Output directory (default: output)')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not Path(args.urls).exists():
        print(f"❌ Error: URLs file not found: {args.urls}")
        return
    
    if not Path(args.keywords).exists():
        print(f"❌ Error: Keywords file not found: {args.keywords}")
        return
    
    # Create scraper
    scraper = StandaloneScraper(
        urls_csv=args.urls,
        keywords_csv=args.keywords,
        days_back=args.days_back,
        target_articles=args.target,
        output_dir=args.output_dir
    )
    
    try:
        # Scrape
        articles = scraper.scrape_all()
        
        # Save
        if articles:
            output_file = scraper.save()
            print(f"\n✅ SUCCESS! Output file: {output_file}")
        else:
            print("\n⚠️  No articles found")
    
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        scraper.cleanup()


if __name__ == "__main__":
    main()