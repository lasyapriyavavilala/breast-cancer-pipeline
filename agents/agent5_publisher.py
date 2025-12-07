"""
Agent 5: Twitter/X Publisher with Grounding Threshold Filtering
SIMPLIFIED VERSION - Twitter Only
Posts only polls above grounding threshold
"""

import os
import json
import time
import sqlite3
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path

import tweepy
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


class TwitterPublisher:
    """
    Publish polls to Twitter/X with grounding score filtering.
    Simplified version - Twitter only, no LinkedIn.
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
        # Twitter credentials
        twitter_api_key: Optional[str] = None,
        twitter_api_secret: Optional[str] = None,
        twitter_access_token: Optional[str] = None,
        twitter_access_secret: Optional[str] = None,
        twitter_bearer_token: Optional[str] = None,
        # Configuration
        db_path: str = "data/pharma_news.db",
        grounding_threshold: float = 0.60,
        dry_run: bool = True,
        post_interval_minutes: int = 3,
        max_posts_per_day: int = 20
    ):
        self.grounding_threshold = grounding_threshold
        self.dry_run = dry_run
        self.post_interval_minutes = post_interval_minutes
        self.max_posts_per_day = max_posts_per_day
        self.db_path = db_path
        
        # Initialize database
        self._init_db()
        
        # Initialize Twitter API
        self.twitter_client = None
        if not dry_run:
            twitter_api_key = twitter_api_key or os.getenv("TWITTER_API_KEY")
            twitter_api_secret = twitter_api_secret or os.getenv("TWITTER_API_SECRET")
            twitter_access_token = twitter_access_token or os.getenv("TWITTER_ACCESS_TOKEN")
            twitter_access_secret = twitter_access_secret or os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
            twitter_bearer_token = twitter_bearer_token or os.getenv("TWITTER_BEARER_TOKEN")
            
            if all([twitter_api_key, twitter_api_secret, twitter_access_token, twitter_access_secret]):
                self.twitter_client = tweepy.Client(
                    bearer_token=twitter_bearer_token,
                    consumer_key=twitter_api_key,
                    consumer_secret=twitter_api_secret,
                    access_token=twitter_access_token,
                    access_token_secret=twitter_access_secret,
                    wait_on_rate_limit=True
                )
                logger.info("✅ Twitter API initialized")
            else:
                logger.error("❌ Twitter credentials incomplete")
                raise ValueError("Missing Twitter credentials")
        
        if dry_run:
            logger.info("🧪 DRY RUN MODE - No actual posts will be made")
        
        logger.info(f"📊 Grounding threshold: {grounding_threshold}")
        logger.info(f"🐦 Twitter: Enabled")
    
    def _init_db(self):
        """Initialize SQLite database for tracking posts"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS posts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                article_url TEXT NOT NULL,
                poll_question TEXT NOT NULL,
                platform TEXT NOT NULL,
                post_id TEXT,
                posted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                category TEXT,
                poll_type TEXT,
                grounding_score REAL,
                grounding_semantic REAL,
                grounding_entity REAL
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
        logger.info(f"✅ Database initialized: {self.db_path}")
    
    def _save_post(
        self, 
        article_url: str,
        poll_question: str,
        post_id: Optional[str] = None,
        category: str = "",
        poll_type: str = "",
        grounding_score: float = 0.0,
        grounding_semantic: float = 0.0,
        grounding_entity: float = 0.0
    ):
        """Save posted poll to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO posts 
            (article_url, poll_question, platform, post_id, category, poll_type, 
             grounding_score, grounding_semantic, grounding_entity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (article_url, poll_question, "twitter", post_id, category, poll_type, 
              grounding_score, grounding_semantic, grounding_entity))
        
        conn.commit()
        conn.close()
    
    def _get_posts_today(self) -> int:
        """Count posts made today"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        cursor.execute("""
            SELECT COUNT(*) FROM posts
            WHERE posted_at >= ? AND post_id IS NOT NULL
        """, (today,))
        
        count = cursor.fetchone()[0]
        conn.close()
        
        return count
    
    def _can_post_now(self) -> bool:
        """Check if we can post based on rate limits"""
        # Check daily limit
        posts_today = self._get_posts_today()
        if posts_today >= self.max_posts_per_day:
            logger.warning(f"⏰ Daily limit reached: {posts_today}/{self.max_posts_per_day}")
            return False
        
        # Check interval
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT posted_at FROM posts
            WHERE post_id IS NOT NULL
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
                logger.info(f"⏰ Too soon since last post ({minutes_ago} min ago)")
                return False
        
        return True
    
    def _extract_company(self, headline: str) -> str:
        """Extract company name from headline"""
        headline_lower = headline.lower()
        
        for company in self.PHARMA_COMPANIES:
            if company.lower() in headline_lower:
                return company
        
        import re
        match = re.search(r"(\w+)'s", headline)
        if match:
            potential_company = match.group(1)
            if len(potential_company) > 3:
                return potential_company
        
        return ""
    
    @staticmethod
    def _safe_trim(text: str, max_chars: int = 280) -> str:
        """Trim text to fit character limit"""
        if len(text) <= max_chars:
            return text
        return text[:max_chars-3] + "..."
    
    def _check_grounding_threshold(self, poll_data: Dict) -> Tuple[bool, str]:
        """
        Check if poll meets grounding threshold.
        
        Returns:
            Tuple of (passes_threshold, reason)
        """
        grounding_score_data = poll_data.get('grounding_score', {})
        overall_score = grounding_score_data.get('overall', 0.0)
        
        if overall_score < self.grounding_threshold:
            reason = f"Below threshold ({overall_score:.3f} < {self.grounding_threshold})"
            return False, reason
        
        return True, f"Passed (score: {overall_score:.3f})"
    
    def post_twitter_thread(self, poll_data: Dict) -> Optional[str]:
        """
        Post a native Twitter poll in a 3-tweet thread:
        Tweet 1: URL + Company
        Tweet 2 (reply): Summary/Headline
        Tweet 3 (reply): Native poll with 4 voting options
        
        Returns:
            Poll tweet ID if posted, None otherwise
        """
        question = poll_data.get('question', '')
        options = poll_data.get('options', [])
        article_url = poll_data.get('article_url', '')
        headline = poll_data.get('article_headline', '')
        summary = poll_data.get('article_summary', '')
        category = poll_data.get('category', '')
        poll_type = poll_data.get('poll_type', '')
        grounding_data = poll_data.get('grounding_score', {})
        
        if not question or len(options) < 2:
            logger.warning(f"⏭️  Skipping invalid poll: missing question or options")
            return None
        
        # Limit to 4 options (Twitter requirement)
        options = options[:4]
        
        # Extract company
        company = self._extract_company(headline)
        
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
                logger.info(f"[DRY RUN - TWITTER] 3-tweet thread:")
                logger.info(f"  Tweet 1: {tweet1_text}")
                logger.info(f"  Tweet 2: {self._safe_trim(summary or headline, 275)}")
                logger.info(f"  Tweet 3 (POLL): {question}")
                for i, opt in enumerate(options, 1):
                    logger.info(f"    {i}. {opt}")
                
                self._save_post(
                    article_url, question, "DRY_RUN", category, poll_type,
                    grounding_data.get('overall', 0.0),
                    grounding_data.get('semantic', 0.0),
                    grounding_data.get('entity', 0.0)
                )
                return "DRY_RUN_TWITTER"
            
            # Post tweet 1 (URL + company)
            response1 = self.twitter_client.create_tweet(text=tweet1_text)
            tweet1_id = response1.data['id']
            logger.info(f"✓ Twitter Tweet 1 posted - ID: {tweet1_id}")
            
            # ===== TWEET 2: Summary =====
            tweet2_text = summary if summary else headline
            tweet2_text = self._safe_trim(tweet2_text, 275)
            
            response2 = self.twitter_client.create_tweet(
                text=tweet2_text,
                in_reply_to_tweet_id=tweet1_id
            )
            tweet2_id = response2.data['id']
            logger.info(f"✓ Twitter Tweet 2 posted - ID: {tweet2_id}")
            
            # ===== TWEET 3: Native Poll =====
            poll_response = self.twitter_client.create_tweet(
                text=question,
                poll_options=options,
                poll_duration_minutes=1440,  # 24 hours
                in_reply_to_tweet_id=tweet2_id
            )
            poll_id = poll_response.data['id']
            
            logger.success(f"✅ Twitter poll posted - ID: {poll_id}")
            logger.info(f"  Q: {question}")
            
            # Save to database
            self._save_post(
                article_url, question, poll_id, category, poll_type,
                grounding_data.get('overall', 0.0),
                grounding_data.get('semantic', 0.0),
                grounding_data.get('entity', 0.0)
            )
            
            return poll_id
            
        except Exception as e:
            logger.error(f"❌ Twitter posting failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def publish_poll(self, poll_data: Dict) -> Dict[str, Optional[str]]:
        """
        Publish poll to Twitter after checking grounding threshold.
        
        Returns:
            Dict with result
        """
        # Check grounding threshold FIRST
        passes, reason = self._check_grounding_threshold(poll_data)
        if not passes:
            logger.warning(f"⏭️  SKIPPED: {reason}")
            logger.warning(f"   Question: {poll_data.get('question', '')[:60]}...")
            return {"skipped": reason}
        
        logger.info(f"✅ Grounding check passed: {reason}")
        
        # Post to Twitter
        twitter_id = self.post_twitter_thread(poll_data)
        
        return {"twitter": twitter_id}
    
    def publish_batch(
        self,
        polls: List[Dict],
        limit: Optional[int] = None,
        respect_rate_limits: bool = True
    ) -> Dict:
        """
        Publish batch of polls with grounding threshold filtering.
        
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
        skipped_threshold = 0
        skipped_rate_limit = 0
        failed = 0
        
        logger.info(f"📊 Processing {len(polls)} polls...")
        logger.info(f"🎯 Grounding threshold: {self.grounding_threshold}")
        
        for i, poll in enumerate(polls, 1):
            # Check rate limits
            if respect_rate_limits and not self._can_post_now():
                logger.info(f"⏰ Rate limit reached. Skipping remaining {len(polls) - i + 1} polls")
                skipped_rate_limit += len(polls) - i + 1
                break
            
            # Check grounding threshold
            passes, reason = self._check_grounding_threshold(poll)
            if not passes:
                logger.info(f"[{i}/{len(polls)}] ⏭️  Skipped: {reason}")
                skipped_threshold += 1
                continue
            
            # Post to Twitter
            logger.info(f"[{i}/{len(polls)}] 📤 Publishing poll...")
            results = self.publish_poll(poll)
            
            if 'skipped' in results:
                skipped_threshold += 1
            elif results.get('twitter'):
                posted += 1
            else:
                failed += 1
            
            # Wait between posts (if not dry run and not last item)
            if not self.dry_run and i < len(polls) and results.get('twitter'):
                wait_seconds = self.post_interval_minutes * 60
                logger.info(f"⏳ Waiting {self.post_interval_minutes} min until next post...")
                time.sleep(wait_seconds)
        
        summary = {
            "total": len(polls),
            "posted": posted if not self.dry_run else 0,
            "skipped_threshold": skipped_threshold,
            "skipped_rate_limit": skipped_rate_limit,
            "failed": failed,
            "grounding_threshold": self.grounding_threshold,
            "dry_run": self.dry_run
        }
        
        logger.info("=" * 70)
        logger.info("📊 PUBLISHING SUMMARY")
        logger.info("=" * 70)
        logger.info(f"  Total polls: {summary['total']}")
        logger.info(f"  Posted to Twitter: {summary['posted']}")
        logger.info(f"  Skipped (below threshold): {summary['skipped_threshold']}")
        logger.info(f"  Skipped (rate limit): {summary['skipped_rate_limit']}")
        logger.info(f"  Failed: {summary['failed']}")
        logger.info(f"  Grounding threshold: {summary['grounding_threshold']}")
        logger.info(f"  Dry run: {summary['dry_run']}")
        logger.info("=" * 70)
        
        return summary


def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Twitter Publisher (Agent 5)")
    parser.add_argument("--input", required=True, help="Input JSON file (polls from Agent 4)")
    parser.add_argument("--grounding-threshold", type=float, default=0.60,
                        help="Minimum grounding score to post (default: 0.60)")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Dry run mode (default: True)")
    parser.add_argument("--no-dry-run", action="store_false", dest="dry_run",
                        help="Actually post to Twitter")
    parser.add_argument("--post-interval", type=int, default=3,
                        help="Minutes between posts (default: 3)")
    parser.add_argument("--max-per-day", type=int, default=20,
                        help="Max posts per day (default: 20)")
    parser.add_argument("--limit", type=int,
                        help="Limit number of polls to process")
    parser.add_argument("--db-path", default="data/pharma_news.db",
                        help="Database path")
    
    args = parser.parse_args()
    
    # Initialize publisher
    publisher = TwitterPublisher(
        db_path=args.db_path,
        grounding_threshold=args.grounding_threshold,
        dry_run=args.dry_run,
        post_interval_minutes=args.post_interval,
        max_posts_per_day=args.max_per_day
    )
    
    # Load polls
    logger.info(f"📂 Loading polls from {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        polls = json.load(f)
    
    logger.info(f"📊 Loaded {len(polls)} polls")
    
    # Show distribution by grounding score
    above_threshold = sum(1 for p in polls if p.get('grounding_score', {}).get('overall', 0) >= args.grounding_threshold)
    below_threshold = len(polls) - above_threshold
    
    logger.info(f"  ✅ Above threshold ({args.grounding_threshold}): {above_threshold}")
    logger.info(f"  ⚠️  Below threshold: {below_threshold}")
    
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