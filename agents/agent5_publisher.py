"""
Agent 5: Unified Twitter/X Publisher
Posts both questions AND summaries with rate limiting and dry-run support.
"""

import os
import json
import time
import sqlite3
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from pathlib import Path

import tweepy
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


class UnifiedTwitterPublisher:
    """
    Publish both questions and summaries to Twitter/X with scheduling and rate limiting.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        access_token: Optional[str] = None,
        access_secret: Optional[str] = None,
        bearer_token: Optional[str] = None,
        db_path: str = "data/pharma_news.db",
        dry_run: bool = True,
        post_interval_minutes: int = 60,
        max_posts_per_day: int = 20
    ):
        self.dry_run = dry_run
        self.post_interval_minutes = post_interval_minutes
        self.max_posts_per_day = max_posts_per_day
        self.db_path = db_path
        
        # Initialize database
        self._init_db()
        
        # Initialize Twitter API if not dry run
        if not dry_run:
            api_key = api_key or os.getenv("TWITTER_API_KEY")
            api_secret = api_secret or os.getenv("TWITTER_API_SECRET")
            access_token = access_token or os.getenv("TWITTER_ACCESS_TOKEN")
            access_secret = access_secret or os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
            bearer_token = bearer_token or os.getenv("TWITTER_BEARER_TOKEN")
            
            if not all([api_key, api_secret, access_token, access_secret]):
                raise ValueError("Twitter API credentials not found")
            
            # Setup Tweepy v4 client
            self.client = tweepy.Client(
                bearer_token=bearer_token,
                consumer_key=api_key,
                consumer_secret=api_secret,
                access_token=access_token,
                access_token_secret=access_secret,
                wait_on_rate_limit=True
            )
            
            logger.info("Twitter API initialized")
        else:
            self.client = None
            logger.info("DRY RUN MODE - No actual tweets will be posted")
    
    def _init_db(self):
        """Initialize SQLite database for tracking posts"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Table for both questions and summaries
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS published_content (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content_type TEXT NOT NULL,
                content_text TEXT NOT NULL,
                article_url TEXT,
                article_headline TEXT,
                category TEXT,
                tweet_id TEXT,
                posted_at TIMESTAMP,
                dry_run BOOLEAN DEFAULT 0,
                status TEXT DEFAULT 'pending'
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_posted_at ON published_content(posted_at)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_content_type ON published_content(content_type)
        """)
        
        conn.commit()
        conn.close()
        logger.info(f"Database initialized: {self.db_path}")
    
    def _get_posts_today(self) -> int:
        """Count posts made today"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        cursor.execute("""
            SELECT COUNT(*) FROM published_content
            WHERE posted_at >= ? AND status = 'posted'
        """, (today,))
        
        count = cursor.fetchone()[0]
        conn.close()
        
        return count
    
    def _can_post_now(self) -> bool:
        """Check if we can post based on rate limits"""
        # Check daily limit
        posts_today = self._get_posts_today()
        if posts_today >= self.max_posts_per_day:
            logger.warning(f"Daily limit reached: {posts_today}/{self.max_posts_per_day}")
            return False
        
        # Check interval
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT posted_at FROM published_content
            WHERE status = 'posted'
            ORDER BY posted_at DESC
            LIMIT 1
        """)
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            last_post_time = datetime.fromisoformat(result[0])
            time_since_last = datetime.now() - last_post_time
            if time_since_last < timedelta(minutes=self.post_interval_minutes):
                logger.info(f"Too soon since last post ({time_since_last.seconds // 60} min)")
                return False
        
        return True
    
    def _log_post(
        self, 
        content_type: str,
        content_text: str,
        article_url: str = "",
        article_headline: str = "",
        category: str = "",
        tweet_id: Optional[str] = None,
        status: str = "posted"
    ):
        """Log posted content to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO published_content 
            (content_type, content_text, article_url, article_headline, category,
             tweet_id, posted_at, dry_run, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            content_type,
            content_text,
            article_url,
            article_headline,
            category,
            tweet_id,
            datetime.now(),
            self.dry_run,
            status
        ))
        
        conn.commit()
        conn.close()
    
    @staticmethod
    def safe_trim(text: str, max_chars: int = 280) -> str:
        """Trim text to fit Twitter limit"""
        return text if len(text) <= max_chars else text[:max_chars-3] + "..."
    
    def post_question(self, question: Dict) -> Optional[str]:
        """
        Post a question as a single tweet.
        
        Args:
            question: Question dict from Agent 4
        
        Returns:
            Tweet ID if posted, None otherwise
        """
        question_text = question.get("question", "")
        article_url = question.get("article_url", "")
        article_headline = question.get("article_headline", "")
        category = question.get("category", "")
        
        # Build tweet text
        tweet_text = f"{question_text}\n\n{article_url}"
        
        if len(tweet_text) > 280:
            # Truncate question if needed
            max_q_len = 280 - len(article_url) - 3
            question_text = question_text[:max_q_len] + "..."
            tweet_text = f"{question_text}\n\n{article_url}"
        
        if self.dry_run:
            logger.info(f"[DRY RUN] Would post QUESTION ({len(tweet_text)} chars):")
            logger.info(f"  {tweet_text}")
            self._log_post("question", question_text, article_url, article_headline, category, status="dry_run")
            return None
        else:
            try:
                response = self.client.create_tweet(text=tweet_text)
                tweet_id = response.data["id"]
                logger.success(f"✓ Posted question (ID: {tweet_id})")
                logger.info(f"  {tweet_text}")
                self._log_post("question", question_text, article_url, article_headline, category, tweet_id, "posted")
                return tweet_id
            except Exception as e:
                logger.error(f"Failed to post question: {e}")
                self._log_post("question", question_text, article_url, article_headline, category, status="failed")
                return None
    
    def post_summary_thread(self, article: Dict) -> Optional[str]:
        """
        Post a summary as a 2-tweet thread.
        
        Args:
            article: Article dict from Agent 2
        
        Returns:
            Parent tweet ID if posted, None otherwise
        """
        url = (article.get("url") or "").strip()
        summary = (article.get("summary_280") or "").strip()
        headline = (article.get("headline") or "").strip()
        entities = article.get("entities") or {}
        company = (entities.get("company_name") or "").strip()
        category = article.get("categorization", {}).get("primary_category", "")
        
        # Build tweet 1 (URL + Company)
        parts1 = []
        if url:
            parts1.append(url)
        if company:
            parts1.append(company)
        tweet1 = "\n".join(parts1)
        tweet1 = self.safe_trim(tweet1)
        
        # Build tweet 2 (Summary)
        tweet2 = self.safe_trim(summary)
        
        if self.dry_run:
            logger.info(f"[DRY RUN] Would post SUMMARY THREAD:")
            logger.info(f"  Tweet 1: {tweet1}")
            logger.info(f"  Tweet 2: {tweet2}")
            self._log_post("summary", summary, url, headline, category, status="dry_run")
            return None
        else:
            try:
                # Post parent tweet
                resp1 = self.client.create_tweet(text=tweet1)
                parent_id = resp1.data["id"]
                
                # Post summary as reply (creates thread)
                resp2 = self.client.create_tweet(
                    text=tweet2,
                    in_reply_to_tweet_id=parent_id
                )
                
                logger.success(f"✓ Posted summary thread (ID: {parent_id})")
                logger.info(f"  Tweet 1: {tweet1}")
                logger.info(f"  Tweet 2: {tweet2}")
                self._log_post("summary", summary, url, headline, category, parent_id, "posted")
                return parent_id
            except Exception as e:
                logger.error(f"Failed to post summary thread: {e}")
                self._log_post("summary", summary, url, headline, category, status="failed")
                return None
    
    def publish_batch(
        self,
        items: List[Dict],
        content_type: str = "auto",
        respect_rate_limits: bool = True
    ) -> Dict:
        """
        Publish batch of questions or summaries.
        
        Args:
            items: List of questions or articles
            content_type: "question", "summary", or "auto" (detect from dict keys)
            respect_rate_limits: Whether to respect rate limits
        
        Returns:
            Summary dict
        """
        posted = 0
        skipped = 0
        failed = 0
        
        for i, item in enumerate(items, 1):
            # Check rate limits
            if respect_rate_limits and not self._can_post_now():
                logger.info(f"Rate limit reached. Skipping remaining {len(items) - i + 1} items")
                skipped += len(items) - i + 1
                break
            
            # Auto-detect content type if needed
            if content_type == "auto":
                if "question" in item:
                    detected_type = "question"
                elif "summary_280" in item:
                    detected_type = "summary"
                else:
                    logger.warning(f"Cannot detect content type for item {i}, skipping")
                    skipped += 1
                    continue
            else:
                detected_type = content_type
            
            # Post based on type
            if detected_type == "question":
                result = self.post_question(item)
            elif detected_type == "summary":
                result = self.post_summary_thread(item)
            else:
                logger.warning(f"Unknown content type: {detected_type}")
                skipped += 1
                continue
            
            if result or self.dry_run:
                posted += 1
            else:
                failed += 1
            
            # Wait between posts (if not dry run and not last item)
            if not self.dry_run and i < len(items):
                wait_seconds = self.post_interval_minutes * 60
                logger.info(f"Waiting {self.post_interval_minutes} min until next post...")
                time.sleep(wait_seconds)
        
        summary = {
            "total": len(items),
            "posted": posted,
            "skipped": skipped,
            "failed": failed,
            "dry_run": self.dry_run
        }
        
        logger.info("=" * 50)
        logger.info("PUBLISHING SUMMARY")
        logger.info("=" * 50)
        for key, value in summary.items():
            logger.info(f"{key}: {value}")
        
        return summary
    
    def get_posting_stats(self) -> Dict:
        """Get statistics from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM published_content WHERE status = 'posted'")
        total_posted = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM published_content WHERE status = 'dry_run'")
        total_dry_run = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT content_type, COUNT(*) as count 
            FROM published_content 
            WHERE status = 'posted'
            GROUP BY content_type
        """)
        by_type = dict(cursor.fetchall())
        
        cursor.execute("""
            SELECT category, COUNT(*) as count 
            FROM published_content 
            WHERE status = 'posted'
            GROUP BY category
            ORDER BY count DESC
        """)
        by_category = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            "total_posted": total_posted,
            "total_dry_run": total_dry_run,
            "by_type": by_type,
            "by_category": by_category
        }


def main():
    """CLI entry point for Unified Agent 5"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified Twitter/X Publisher (Agent 5)")
    parser.add_argument("--input", required=True, help="Input JSON file")
    parser.add_argument("--type", choices=["question", "summary", "auto"], default="auto",
                        help="Content type (auto-detect by default)")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    parser.add_argument("--post-interval", type=int, default=60, help="Minutes between posts")
    parser.add_argument("--max-per-day", type=int, default=20, help="Max posts per day")
    parser.add_argument("--limit", type=int, help="Limit number of items to post")
    parser.add_argument("--stats", action="store_true", help="Show stats and exit")
    parser.add_argument("--db-path", default="data/pharma_news.db", help="Database path")
    
    args = parser.parse_args()
    
    # Initialize publisher
    publisher = UnifiedTwitterPublisher(
        db_path=args.db_path,
        dry_run=args.dry_run,
        post_interval_minutes=args.post_interval,
        max_posts_per_day=args.max_per_day
    )
    
    # Show stats if requested
    if args.stats:
        stats = publisher.get_posting_stats()
        logger.info("=" * 50)
        logger.info("POSTING STATISTICS")
        logger.info("=" * 50)
        logger.info(f"Total posted: {stats['total_posted']}")
        logger.info(f"Total dry run: {stats['total_dry_run']}")
        logger.info("\nBy content type:")
        for ctype, count in stats['by_type'].items():
            logger.info(f"  {ctype}: {count}")
        logger.info("\nBy category:")
        for cat, count in stats['by_category'].items():
            logger.info(f"  {cat}: {count}")
        return
    
    # Load content
    logger.info(f"Loading from {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        items = json.load(f)
    
    # Apply limit if specified
    if args.limit:
        items = items[:args.limit]
    
    logger.info(f"Loaded {len(items)} items (type: {args.type})")
    
    # Publish
    summary = publisher.publish_batch(items, content_type=args.type)
    
    logger.success(f"✅ Publishing complete!")
    return summary


if __name__ == "__main__":
    main()