"""
Complete Integrated Pipeline Runner
Runs all 5 agents sequentially with proper path handling for Windows
Agent 1: StandaloneScraper → Agent 2: Entities → Agent 3: Categorization → Agent 4: Polls → Agent 5: Publisher
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict

from loguru import logger
from dotenv import load_dotenv

# Import all agent classes
try:
    from agents.agent1_scraper import StandaloneScraper
except ImportError:
    logger.warning("standalone_scraper.py not found in current directory")
    StandaloneScraper = None

try:
    from agents.agent2_summarizer import ArticleExtractor
except ImportError:
    logger.warning("agent2_entity_extraction.py not found in current directory")
    ArticleExtractor = None

try:
    from agents.agent3_categorizer import ArticleCategorizer
except ImportError:
    logger.warning("agent3_categorizer.py not found in current directory")
    ArticleCategorizer = None

try:
    from agents.agent4_question_gen import PollGenerator
except ImportError:
    logger.warning("agent4_poll_generator.py not found in current directory")
    PollGenerator = None

try:
    from agents.agent5_publisher import UnifiedTwitterPublisher
except ImportError:
    logger.warning("agent5_publisher.py not found in current directory")
    UnifiedTwitterPublisher = None


class IntegratedPipelineRunner:
    """Orchestrates the entire 5-agent pipeline with proper path handling"""
    
    def __init__(
        self,
        base_dir: str = None,  # ← Changed from hardcoded path
        target_articles: int = 50,
        days_back: int = 90,
        polls_per_article: int = 3,
        post_polls: bool = False,
        post_limit: Optional[int] = None,
        dry_run: bool = True
    ):
        # Use current working directory if not specified (works on Windows & Linux)
        if base_dir is None:
            import os
            base_dir = os.getcwd()
        
        self.base_dir = Path(base_dir)
        self.target_articles = target_articles
        self.days_back = days_back
        self.polls_per_article = polls_per_article
        self.post_polls = post_polls
        self.post_limit = post_limit
        self.dry_run = dry_run
        
        # Setup directory structure
        self.data_dir = self.base_dir / "data"
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.outputs_dir = self.data_dir / "outputs"
        self.embeddings_dir = self.data_dir / "embeddings"
        
        # Create all directories
        for d in [self.raw_dir, self.processed_dir, self.outputs_dir, self.embeddings_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # File paths (will be set during execution)
        self.scraped_file = None
        self.enhanced_file = None
        self.categorized_file = None
        self.polls_file = None
        
        logger.info("="*70)
        logger.info("INTEGRATED PIPELINE INITIALIZED")
        logger.info("="*70)
        logger.info(f"Base directory: {self.base_dir}")
        logger.info(f"Target articles: {target_articles}")
        logger.info(f"Days back: {days_back}")
        logger.info(f"Polls per article: {polls_per_article}")
        logger.info(f"Dry run: {dry_run}")
    
    def run_agent1_scraper(self) -> str:
        """Agent 1: StandaloneScraper - Multi-source news scraping"""
        logger.info("\n" + "="*70)
        logger.info("AGENT 1: STANDALONE MULTI-SOURCE SCRAPER")
        logger.info("="*70)
        
        if StandaloneScraper is None:
            raise ImportError("StandaloneScraper not found. Please ensure standalone_scraper.py is in the current directory.")
        
        # Try data directory first, then fall back to base directory
        urls_csv = self.data_dir / "pharma_urls.csv"
        keywords_csv = self.data_dir / "keywords.csv"
        
        logger.info(f"🔍 Looking for CSV files...")
        logger.info(f"   Trying data directory: {self.data_dir}")
        logger.info(f"   Base directory: {self.base_dir}")
        
        # If files don't exist in data/, check base directory
        if not urls_csv.exists():
            logger.info(f"   URLs not found in data/, checking base directory...")
            urls_csv = self.base_dir / "pharma_urls.csv"
        if not keywords_csv.exists():
            logger.info(f"   Keywords not found in data/, checking base directory...")
            keywords_csv = self.base_dir / "keywords.csv"
        
        # Validate CSV files exist
        if not urls_csv.exists():
            raise FileNotFoundError(
                f"Missing {urls_csv}. Please create this file with columns: source_name,source_type,url,priority. "
                f"Searched in: {self.data_dir} and {self.base_dir}"
            )
        if not keywords_csv.exists():
            raise FileNotFoundError(
                f"Missing {keywords_csv}. Please create this file with column: keyword. "
                f"Searched in: {self.data_dir} and {self.base_dir}"
            )
        
        logger.info(f"📂 Loading configuration:")
        logger.info(f"   URLs: {urls_csv}")
        logger.info(f"   Keywords: {keywords_csv}")
        
        # Initialize scraper
        scraper = StandaloneScraper(
            urls_csv=str(urls_csv),
            keywords_csv=str(keywords_csv),
            days_back=self.days_back,
            target_articles=self.target_articles,
            output_dir=str(self.raw_dir)
        )
        
        # Scrape
        logger.info(f"🔍 Scraping up to {self.target_articles} articles from last {self.days_back} days...")
        articles = scraper.scrape_all()
        
        # Save
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = scraper.save(f"scraped_articles_{timestamp}.json")
        scraper.cleanup()
        
        self.scraped_file = output_file
        logger.success(f"✅ Agent 1 complete: {len(articles)} articles scraped")
        logger.info(f"📄 Output: {output_file}")
        
        return self.scraped_file
    
    def run_agent2_extraction(self) -> str:
        """Agent 2: Entity Extraction & Summarization using Anthropic"""
        logger.info("\n" + "="*70)
        logger.info("AGENT 2: ENTITY EXTRACTION & SUMMARIZATION")
        logger.info("="*70)
        
        if ArticleExtractor is None:
            raise ImportError("ArticleExtractor not found. Please ensure agent2_entity_extraction.py is in the current directory.")
        
        if not self.scraped_file:
            raise ValueError("Agent 1 must run first. No scraped file found.")
        
        # Load articles
        logger.info(f"📂 Loading articles from {self.scraped_file}")
        with open(self.scraped_file, "r", encoding="utf-8") as f:
            articles = json.load(f)
        
        logger.info(f"📊 Processing {len(articles)} articles from StandaloneScraper")
        
        # Initialize extractor with Anthropic
        extractor = ArticleExtractor(
            model="claude-sonnet-4-20250514",  # Agent 2 model
            temperature=0.3,
            max_tokens=300
        )
        
        # Output paths
        output_json = self.processed_dir / "enhanced_articles.json"
        output_ndjson = self.processed_dir / "enhanced_articles.ndjson"
        
        # Process with incremental writes
        logger.info(f"🔄 Extracting entities and generating summaries...")
        processed = extractor.process_articles_batch(
            articles,
            output_json=str(output_json),
            output_ndjson=str(output_ndjson),
            incremental=True
        )
        
        self.enhanced_file = str(output_json)
        logger.success(f"✅ Agent 2 complete: {len(processed)} articles enhanced")
        logger.info(f"📄 JSON output: {output_json}")
        logger.info(f"📄 NDJSON output: {output_ndjson}")
        
        return self.enhanced_file
    
    def run_agent3_categorization(self) -> str:
        """Agent 3: Article Categorization using FAISS + Anthropic"""
        logger.info("\n" + "="*70)
        logger.info("AGENT 3: ARTICLE CATEGORIZATION")
        logger.info("="*70)
        
        if ArticleCategorizer is None:
            raise ImportError("ArticleCategorizer not found. Please ensure agent3_categorizer.py is in the current directory.")
        
        if not self.enhanced_file:
            raise ValueError("Agent 2 must run first. No enhanced file found.")
        
        # Load articles
        logger.info(f"📂 Loading articles from {self.enhanced_file}")
        with open(self.enhanced_file, "r", encoding="utf-8") as f:
            articles = json.load(f)
        
        logger.info(f"📊 Categorizing {len(articles)} articles")
        
        # Initialize categorizer
        categorizer = ArticleCategorizer(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            index_path=str(self.embeddings_dir / "articles.index"),
            metadata_path=str(self.embeddings_dir / "metadata.pkl"),
            similarity_threshold=0.75
        )
        
        # Output path
        output_file = self.processed_dir / "categorized_articles.json"
        
        # Categorize
        logger.info(f"🔄 Categorizing with FAISS embeddings + Anthropic Claude...")
        categorized = categorizer.categorize_batch(
            articles,
            output_path=str(output_file),
            save_index=True
        )
        
        self.categorized_file = str(output_file)
        logger.success(f"✅ Agent 3 complete: {len(categorized)} articles categorized")
        logger.info(f"📄 Output: {output_file}")
        logger.info(f"📄 FAISS index: {self.embeddings_dir / 'articles.index'}")
        
        return self.categorized_file
    
    def run_agent4_polls(self) -> str:
        """Agent 4: HCP Poll Generation using Anthropic"""
        logger.info("\n" + "="*70)
        logger.info("AGENT 4: HCP POLL GENERATION")
        logger.info("="*70)
        
        if PollGenerator is None:
            raise ImportError("PollGenerator not found. Please ensure agent4_poll_generator.py is in the current directory.")
        
        if not self.categorized_file:
            raise ValueError("Agent 3 must run first. No categorized file found.")
        
        # Load articles
        logger.info(f"📂 Loading articles from {self.categorized_file}")
        with open(self.categorized_file, "r", encoding="utf-8") as f:
            articles = json.load(f)
        
        logger.info(f"📊 Generating polls for {len(articles)} articles")
        
        # Initialize poll generator with Anthropic
        generator = PollGenerator(
            model="claude-sonnet-4-5-20250929",  # Agent 4 model
            polls_per_article=self.polls_per_article,
            temperature=0.8,
            max_tokens=480,
            passes=3,  # Multi-pass generation
            similarity_threshold=0.95,  # Semantic dedup
            grounding_threshold=0.75,  # Flag polls below this
            entity_weight=0.30  # Entity weight in grounding
        )
        
        # Output path
        output_file = self.outputs_dir / "twitter_polls.json"
        
        # Generate
        logger.info(f"🔄 Running {self.polls_per_article} polls per article with 3-pass generation")
        polls = generator.generate_batch(
            articles,
            output_path=str(output_file)
        )
        
        self.polls_file = str(output_file)
        logger.success(f"✅ Agent 4 complete: {len(polls)} polls generated")
        logger.info(f"📄 Output: {output_file}")
        
        # Show flagged polls count
        flagged = sum(1 for p in polls if p.get("grounding_score", {}).get("needs_review", False))
        if flagged > 0:
            logger.warning(f"⚠️  {flagged} polls flagged for review (below grounding threshold)")
        
        return self.polls_file
    
    def run_agent5_publisher(self):
        """Agent 5: Twitter/X Publisher"""
        logger.info("\n" + "="*70)
        logger.info("AGENT 5: TWITTER/X PUBLISHER")
        logger.info("="*70)
        
        if UnifiedTwitterPublisher is None:
            raise ImportError("UnifiedTwitterPublisher not found. Please ensure agent5_publisher.py is in the current directory.")
        
        if not self.post_polls:
            logger.info("⏭️  Skipping Agent 5 (posting disabled)")
            logger.info("💡 Use --post-polls flag to enable Twitter/X posting")
            return
        
        if not self.polls_file:
            raise ValueError("Agent 4 must run first. No polls file found.")
        
        # Initialize publisher
        logger.info(f"📤 Initializing Twitter/X publisher (dry_run={self.dry_run})")
        publisher = UnifiedTwitterPublisher(
            dry_run=self.dry_run,
            db_path=str(self.data_dir / "pharma_news.db"),
            post_interval_minutes=60,
            max_posts_per_day=20
        )
        
        # Load polls
        logger.info(f"📂 Loading polls from {self.polls_file}")
        with open(self.polls_file, "r", encoding="utf-8") as f:
            polls = json.load(f)
        
        logger.info(f"📊 Total polls available: {len(polls)}")
        
        # Convert polls to publishable format
        publishable_polls = []
        for poll in polls:
            # Reconstruct poll text
            question = poll.get("question", "")
            options = poll.get("options", [])
            poll_text = f"Q: {question}\n" + "\n".join([f"- {opt}" for opt in options])
            
            publishable_polls.append({
                "question": poll_text,
                "article_url": poll.get("article_url", ""),
                "article_headline": poll.get("article_headline", ""),
                "category": poll.get("category", ""),
                "grounding_score": poll.get("grounding_score", {}).get("overall", 0.0)
            })
        
        # Limit if specified
        if self.post_limit:
            publishable_polls = publishable_polls[:self.post_limit]
            logger.info(f"📊 Limiting to {self.post_limit} polls")
        
        # Publish
        logger.info(f"🚀 Publishing {len(publishable_polls)} polls...")
        summary = publisher.publish_batch(
            publishable_polls,
            content_type="question",
            respect_rate_limits=True
        )
        
        logger.success(f"✅ Agent 5 complete")
        logger.info(f"📊 Posted: {summary['posted']}/{summary['total']}")
        logger.info(f"⏭️  Skipped: {summary['skipped']}")
        if summary.get('failed', 0) > 0:
            logger.warning(f"❌ Failed: {summary['failed']}")
    
    def print_summary(self):
        """Print pipeline execution summary"""
        logger.info("\n" + "="*70)
        logger.info("📊 PIPELINE EXECUTION SUMMARY")
        logger.info("="*70)
        
        summary_data = []
        
        # Agent 1
        if self.scraped_file and Path(self.scraped_file).exists():
            try:
                with open(self.scraped_file, encoding="utf-8") as f:
                    articles = json.load(f)
                    count = len(articles)
                    # Count by content type
                    from collections import Counter
                    types = Counter(a.get("content_type", "unknown") for a in articles)
                    type_str = ", ".join([f"{t}: {c}" for t, c in types.most_common(3)])
                    summary_data.append(f"Agent 1 (Scraper):        {count:>4} articles ({type_str})")
            except:
                pass
        
        # Agent 2
        if self.enhanced_file and Path(self.enhanced_file).exists():
            try:
                with open(self.enhanced_file, encoding="utf-8") as f:
                    articles = json.load(f)
                    count = len(articles)
                    with_drugs = sum(1 for a in articles if a.get("entities", {}).get("drug_names"))
                    with_trials = sum(1 for a in articles if a.get("entities", {}).get("trial_names"))
                    summary_data.append(f"Agent 2 (Extraction):     {count:>4} enhanced ({with_drugs} drugs, {with_trials} trials)")
            except:
                pass
        
        # Agent 3
        if self.categorized_file and Path(self.categorized_file).exists():
            try:
                with open(self.categorized_file, encoding="utf-8") as f:
                    articles = json.load(f)
                    count = len(articles)
                    from collections import Counter
                    cats = Counter(a.get("categorization", {}).get("primary_category") for a in articles)
                    top_cat = cats.most_common(1)[0] if cats else ("N/A", 0)
                    summary_data.append(f"Agent 3 (Categorization): {count:>4} categorized (top: {top_cat[0]}, {top_cat[1]})")
            except:
                pass
        
        # Agent 4
        if self.polls_file and Path(self.polls_file).exists():
            try:
                with open(self.polls_file, encoding="utf-8") as f:
                    polls = json.load(f)
                    count = len(polls)
                    flagged = sum(1 for p in polls if p.get("grounding_score", {}).get("needs_review", False))
                    avg_score = sum(p.get("grounding_score", {}).get("overall", 0) for p in polls) / count if count > 0 else 0
                    summary_data.append(f"Agent 4 (Polls):          {count:>4} polls (avg score: {avg_score:.2f}, flagged: {flagged})")
            except:
                pass
        
        # Agent 5
        if self.post_polls:
            summary_data.append(f"Agent 5 (Publisher):      {'Dry run' if self.dry_run else 'Live posting'}")
        
        for line in summary_data:
            logger.info(line)
        
        logger.info("\n" + "="*70)
        logger.info("📁 OUTPUT FILES")
        logger.info("="*70)
        
        if self.scraped_file:
            logger.info(f"Scraped:      {self.scraped_file}")
        if self.enhanced_file:
            logger.info(f"Enhanced:     {self.enhanced_file}")
        if self.categorized_file:
            logger.info(f"Categorized:  {self.categorized_file}")
        if self.polls_file:
            logger.info(f"Polls:        {self.polls_file}")
        
        logger.info("="*70)


def main():
    """CLI entry point"""
    import argparse
    
    load_dotenv()
    
    parser = argparse.ArgumentParser(
        description="Integrated 5-Agent Pipeline for Breast Cancer News Processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (scrape 50 articles, no posting)
  python run_integrated_pipeline.py
  
  # Custom article count with poll generation
  python run_integrated_pipeline.py --target 20 --polls-per-article 4
  
  # Full pipeline with Twitter posting (dry run)
  python run_integrated_pipeline.py --target 30 --post-polls --dry-run
  
  # Actual posting (remove dry-run flag)
  python run_integrated_pipeline.py --target 10 --post-polls --post-limit 5 --no-dry-run
        """
    )
    
    # Core options
    parser.add_argument("--base-dir", default=None,
                    help="Base directory for all data (default: current directory)")
    parser.add_argument("--target", type=int, default=50,
                        help="Number of articles to scrape (default: 50)")
    parser.add_argument("--days-back", type=int, default=90,
                        help="Days to look back (default: 90)")
    parser.add_argument("--polls-per-article", type=int, default=3,
                        help="Polls to generate per article (default: 3)")
    
    # Publishing options
    parser.add_argument("--post-polls", action="store_true",
                        help="Post polls to Twitter/X")
    parser.add_argument("--post-limit", type=int,
                        help="Limit number of polls to post")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Dry run mode (default: True)")
    parser.add_argument("--no-dry-run", action="store_false", dest="dry_run",
                        help="Actually post to Twitter/X")
    
    # Agent skip options
    parser.add_argument("--skip-agent1", action="store_true",
                        help="Skip scraping (use existing data)")
    parser.add_argument("--skip-agent2", action="store_true",
                        help="Skip entity extraction")
    parser.add_argument("--skip-agent3", action="store_true",
                        help="Skip categorization")
    parser.add_argument("--skip-agent4", action="store_true",
                        help="Skip poll generation")
    parser.add_argument("--skip-agent5", action="store_true",
                        help="Skip publishing")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    runner = IntegratedPipelineRunner(
        base_dir=args.base_dir,
        target_articles=args.target,
        days_back=args.days_back,
        polls_per_article=args.polls_per_article,
        post_polls=args.post_polls and not args.skip_agent5,
        post_limit=args.post_limit,
        dry_run=args.dry_run
    )
    
    # Run pipeline
    start_time = time.time()
    
    try:
        logger.info("\n🚀 STARTING PIPELINE EXECUTION\n")
        
        if not args.skip_agent1:
            runner.run_agent1_scraper()
        else:
            logger.info("⏭️  Skipping Agent 1 (Scraper)")
        
        if not args.skip_agent2:
            runner.run_agent2_extraction()
        else:
            logger.info("⏭️  Skipping Agent 2 (Extraction)")
        
        if not args.skip_agent3:
            runner.run_agent3_categorization()
        else:
            logger.info("⏭️  Skipping Agent 3 (Categorization)")
        
        if not args.skip_agent4:
            runner.run_agent4_polls()
        else:
            logger.info("⏭️  Skipping Agent 4 (Polls)")
        
        if not args.skip_agent5:
            runner.run_agent5_publisher()
        else:
            logger.info("⏭️  Skipping Agent 5 (Publisher)")
        
        elapsed = time.time() - start_time
        
        logger.info("\n" + "="*70)
        logger.success(f"✅ PIPELINE COMPLETE in {elapsed/60:.1f} minutes")
        logger.info("="*70)
        
        runner.print_summary()
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()