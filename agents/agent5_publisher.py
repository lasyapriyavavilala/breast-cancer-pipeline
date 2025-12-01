"""
Agent 5: Twitter/X Publisher with Native Polls
Posts polls as native Twitter polls in threaded format:
Tweet 1: URL + Company
Tweet 2 (reply): Summary
Tweet 3 (reply): Native poll with 4 voting options
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


class TwitterPollPublisher:
    """
    Publish polls to Twitter/X as native polls with rate limiting.
    """
    
    # Common pharma companies for extraction
    PHARMA_COMPANIES = [
        'Pfizer', 'Roche', 'Novartis', 'AstraZeneca', 'Merck', 'Lilly',
        'Eli Lilly', 'Genentech', 'Daiichi Sankyo', 'Seagen', 'Gilead', 
        'BMS', 'Bristol Myers Squibb', 'Bristol-Myers Squibb', 'Amgen', 
        'GSK', 'Sanofi', 'Bayer', 'AbbVie', 'Takeda', 'Eisai',
        'ImmunoGen', 'Regeneron'
    ]
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        access_token: Optional[str] = None,
        access_secret: Optional[str] = None,
        bearer_token: Optional[str] = None,
        db_path: str = "data/pharma_news.db",
        dry_run: bool = True,
        post_interval_minutes: int = 3,
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
        
        # Table for polls
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS posts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                article_url TEXT NOT NULL,
                poll_question TEXT NOT NULL,
                tweet_id TEXT,
                posted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                category TEXT,
                poll_type TEXT,
                grounding_score REAL
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_article_url ON posts(article_url)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_posted_at ON posts(posted_at)
        """)
        
        conn.commit()
        conn.close()
        logger.info(f"Database initialized: {self.db_path}")
    
    def _save_post(
        self, 
        article_url: str,
        poll_question: str,
        tweet_id: Optional[str] = None,
        category: str = "",
        poll_type: str = "",
        grounding_score: float = 0.0
    ):
        """Save posted poll to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO posts 
            (article_url, poll_question, tweet_id, category, poll_type, grounding_score)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (article_url, poll_question, tweet_id, category, poll_type, grounding_score))
        
        conn.commit()
        conn.close()
    
    def _get_posts_today(self) -> int:
        """Count posts made today"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        cursor.execute("""
            SELECT COUNT(*) FROM posts
            WHERE posted_at >= ? AND tweet_id IS NOT NULL
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
            SELECT posted_at FROM posts
            WHERE tweet_id IS NOT NULL
            ORDER BY posted_at DESC
            LIMIT 1
        """)
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            last_post_time = datetime.fromisoformat(result[0])
            time_since_last = datetime.now() - last_post_time
            if time_since_last < timedelta(minutes=self.post_interval_minutes):
                minutes_ago = time_since_last.seconds // 60
                logger.info(f"Too soon since last post ({minutes_ago} min ago)")
                return False
        
        return True
    
    def _extract_company(self, headline: str) -> str:
        """Extract company name from headline"""
        headline_lower = headline.lower()
        
        for company in self.PHARMA_COMPANIES:
            if company.lower() in headline_lower:
                return company
        
        # Try to extract from possessive forms
        import re
        match = re.search(r"(\w+)'s", headline)
        if match:
            potential_company = match.group(1)
            if len(potential_company) > 3:  # Avoid very short matches
                return potential_company
        
        return ""
    
    @staticmethod
    def _safe_trim(text: str, max_chars: int = 280) -> str:
        """Trim text to fit Twitter limit"""
        if len(text) <= max_chars:
            return text
        return text[:max_chars-3] + "..."
    
    def post_poll_thread(self, poll_data: dict) -> Optional[str]:
        """
        Post a native Twitter poll in a 3-tweet thread:
        Tweet 1: URL + Company
        Tweet 2 (reply): Summary/Headline
        Tweet 3 (reply): Native poll with 4 voting options
        
        Args:
            poll_data: Poll dict from Agent 4
        
        Returns:
            Poll tweet ID if posted, None otherwise
        """
        question = poll_data.get('question', '')
        options = poll_data.get('options', [])
        article_url = poll_data.get('article_url', '')
        headline = poll_data.get('article_headline', '')
        category = poll_data.get('category', '')
        poll_type = poll_data.get('poll_type', '')
        grounding_score = poll_data.get('grounding_score', {}).get('overall', 0.0)
        
        if not question or len(options) < 2:
            logger.warning(f"Skipping invalid poll: missing question or options")
            return None
        
        # Limit to 4 options (Twitter requirement)
        options = options[:4]
        
        # Ensure we have at least 2 options
        if len(options) < 2:
            logger.warning(f"Skipping poll: need at least 2 options, got {len(options)}")
            return None
        
        # Extract company from headline
        company = self._extract_company(headline)
        
        # Build thread
        try:
            # ===== TWEET 1: URL + Company =====
            tweet1_parts = []
            if article_url:
                tweet1_parts.append(article_url)
            if company:
                tweet1_parts.append(company)
            
            tweet1_text = "\n".join(tweet1_parts) if tweet1_parts else article_url
            tweet1_text = self._safe_trim(tweet1_text, 280)
            
            if self.dry_run:
                logger.info(f"[DRY RUN] Would post 3-tweet poll thread:")
                logger.info(f"  Tweet 1: {tweet1_text}")
                logger.info(f"  Tweet 2: {self._safe_trim(headline, 275)}")
                logger.info(f"  Tweet 3 (POLL): {question}")
                for i, opt in enumerate(options, 1):
                    logger.info(f"    {i}. {opt}")
                self._save_post(article_url, question, None, category, poll_type, grounding_score)
                return "DRY_RUN"
            
            # Post tweet 1 (URL + company)
            response1 = self.client.create_tweet(text=tweet1_text)
            tweet1_id = response1.data['id']
            logger.info(f"✓ Posted tweet 1 (URL + company) - ID: {tweet1_id}")
            
            # ===== TWEET 2: Summary (280 chars from Agent 2) =====
            # Use summary_280 from Agent 2, fallback to headline if not available
            summary = poll_data.get('article_summary', '')  # ← GET SUMMARY
            tweet2_text = summary if summary else headline
            tweet2_text = self._safe_trim(tweet2_text, 275)

            response2 = self.client.create_tweet(
                text=tweet2_text,
                in_reply_to_tweet_id=tweet1_id
            )
            tweet2_id = response2.data['id']
            logger.info(f"✓ Posted tweet 2 (summary) - ID: {tweet2_id}")
            logger.info(f"  Summary: {tweet2_text[:100]}...")  # ← LOG SUMMARY
            
            # ===== TWEET 3: Native Poll (reply to tweet 2) =====
            poll_response = self.client.create_tweet(
                text=question,
                poll_options=options,
                poll_duration_minutes=1440,  # 24 hours
                in_reply_to_tweet_id=tweet2_id
            )
            poll_id = poll_response.data['id']
            
            logger.success(f"✓ Posted poll (ID: {poll_id})")
            logger.info(f"  Q: {question}")
            for i, opt in enumerate(options, 1):
                logger.info(f"  {i}. {opt}")
            logger.info(f"  URL: {article_url}")
            
            # Save to database
            self._save_post(article_url, question, poll_id, category, poll_type, grounding_score)
            
            return poll_id
            
        except Exception as e:
            logger.error(f"Failed to post poll thread: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def publish_batch(
        self,
        polls: List[Dict],
        limit: Optional[int] = None,
        respect_rate_limits: bool = True
    ) -> Dict:
        """
        Publish batch of polls.
        
        Args:
            polls: List of poll dicts from Agent 4
            limit: Max number to post (None = all)
            respect_rate_limits: Whether to respect rate limits
        
        Returns:
            Summary dict with stats
        """
        if limit:
            polls = polls[:limit]
        
        posted = 0
        skipped = 0
        failed = 0
        
        logger.info(f"Publishing {len(polls)} polls...")
        
        for i, poll in enumerate(polls, 1):
            # Check rate limits
            if respect_rate_limits and not self._can_post_now():
                logger.info(f"Rate limit reached. Skipping remaining {len(polls) - i + 1} polls")
                skipped += len(polls) - i + 1
                break
            
            # Post poll
            result = self.post_poll_thread(poll)
            
            if result:
                posted += 1
            elif result is None:
                failed += 1
            else:
                skipped += 1
            
            # Wait between posts (if not dry run and not last item)
            if not self.dry_run and i < len(polls) and result:
                wait_seconds = self.post_interval_minutes * 60
                logger.info(f"Waiting {self.post_interval_minutes} min until next post...")
                time.sleep(wait_seconds)
        
        summary = {
            "total": len(polls),
            "posted": posted if not self.dry_run else 0,
            "skipped": skipped,
            "failed": failed,
            "dry_run": self.dry_run
        }
        
        logger.info("=" * 50)
        logger.info("PUBLISHING SUMMARY")
        logger.info("=" * 50)
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")
        
        return summary
    
    def get_stats(self) -> Dict:
        """Get statistics from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM posts WHERE tweet_id IS NOT NULL")
        total_posted = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT category, COUNT(*) as count 
            FROM posts 
            WHERE tweet_id IS NOT NULL
            GROUP BY category
            ORDER BY count DESC
        """)
        by_category = dict(cursor.fetchall())
        
        cursor.execute("""
            SELECT poll_type, COUNT(*) as count 
            FROM posts 
            WHERE tweet_id IS NOT NULL
            GROUP BY poll_type
            ORDER BY count DESC
        """)
        by_type = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            "total_posted": total_posted,
            "by_category": by_category,
            "by_type": by_type
        }


def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Twitter Poll Publisher (Agent 5)")
    parser.add_argument("--input", required=True, help="Input JSON file (polls from Agent 4)")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    parser.add_argument("--post-interval", type=int, default=60, help="Minutes between posts")
    parser.add_argument("--max-per-day", type=int, default=20, help="Max posts per day")
    parser.add_argument("--limit", type=int, help="Limit number of polls to post")
    parser.add_argument("--stats", action="store_true", help="Show stats and exit")
    parser.add_argument("--db-path", default="data/pharma_news.db", help="Database path")
    
    args = parser.parse_args()
    
    # Initialize publisher
    publisher = TwitterPollPublisher(
        db_path=args.db_path,
        dry_run=args.dry_run,
        post_interval_minutes=args.post_interval,
        max_posts_per_day=args.max_per_day
    )
    
    # Show stats if requested
    if args.stats:
        stats = publisher.get_stats()
        logger.info("=" * 50)
        logger.info("POSTING STATISTICS")
        logger.info("=" * 50)
        logger.info(f"Total posted: {stats['total_posted']}")
        logger.info("\nBy category:")
        for cat, count in stats['by_category'].items():
            logger.info(f"  {cat}: {count}")
        logger.info("\nBy poll type:")
        for ptype, count in stats['by_type'].items():
            logger.info(f"  {ptype}: {count}")
        return
    
    # Load polls
    logger.info(f"Loading polls from {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        polls = json.load(f)
    
    logger.info(f"Loaded {len(polls)} polls")
    
    # Publish
    summary = publisher.publish_batch(
        polls,
        limit=args.limit,
        respect_rate_limits=True
    )
    
    logger.success(f"✅ Publishing complete!")
    return summary


if __name__ == "__main__":
    main()