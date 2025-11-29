"""
Agent 3: Article Categorizer
Categorizes articles using semantic embeddings (FAISS) and Claude for labeling.
"""

import os
import json
import pickle
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


class ArticleCategorizer:
    """
    Categorize articles using embeddings and LLM-based classification.
    """
    
    CATEGORIES = [
        "Clinical Trial Results",
        "FDA Approval/Regulatory",
        "Drug Development Pipeline",
        "Scientific Research",
        "Partnership/Collaboration",
        "Market/Commercial",
        "Patient Access/Advocacy",
        "Safety/Adverse Events"
    ]
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_path: str = "data/embeddings/articles.index",
        metadata_path: str = "data/embeddings/metadata.pkl",
        anthropic_api_key: Optional[str] = None,
        similarity_threshold: float = 0.75
    ):
        self.embedding_model_name = embedding_model
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.similarity_threshold = similarity_threshold
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.encoder = SentenceTransformer(embedding_model)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        
        # Initialize Anthropic
        api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found")
        self.client = Anthropic(api_key=api_key)
        
        # Initialize or load FAISS index
        self.index = None
        self.metadata = []
        self._load_or_create_index()
        
        logger.info("Categorizer initialized")
    
    def _load_or_create_index(self):
        """Load existing FAISS index or create new one"""
        Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
        
        if os.path.exists(self.index_path):
            logger.info(f"Loading existing FAISS index from {self.index_path}")
            self.index = faiss.read_index(self.index_path)
            
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, "rb") as f:
                    self.metadata = pickle.load(f)
                logger.info(f"Loaded {len(self.metadata)} metadata entries")
        else:
            logger.info("Creating new FAISS index")
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.metadata = []
    
    def _save_index(self):
        """Save FAISS index and metadata"""
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)
        logger.info(f"Saved index ({self.index.ntotal} vectors) and metadata")
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        return self.encoder.encode(text, convert_to_numpy=True)
    
    def _llm_categorize(self, headline: str, summary: str, entities: Dict) -> Dict:
        """Use Claude to categorize article"""
        prompt = f"""Categorize this pharmaceutical article into ONE or MORE of these categories:

{chr(10).join(f"- {cat}" for cat in self.CATEGORIES)}

Also provide 3-5 relevant tags (e.g., "metastatic", "CDK4/6 inhibitor", "Phase 3", etc.)

ARTICLE:
Headline: {headline}
Summary: {summary}
Drug Names: {', '.join(entities.get('drug_names', [])) or 'None'}
Indications: {', '.join(entities.get('indications', [])) or 'None'}
Trial Phases: {', '.join(entities.get('trial_phases', [])) or 'None'}

Return ONLY a JSON object with this structure:
{{
  "primary_category": "...",
  "secondary_categories": ["..."],
  "tags": ["tag1", "tag2", "tag3"],
  "confidence": 0.95
}}"""
        
        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                temperature=0.2,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = message.content[0].text.strip()
            # Extract JSON
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            else:
                logger.warning("No JSON found in LLM response")
                return {
                    "primary_category": self.CATEGORIES[0],
                    "secondary_categories": [],
                    "tags": [],
                    "confidence": 0.5
                }
        except Exception as e:
            logger.error(f"LLM categorization failed: {e}")
            return {
                "primary_category": self.CATEGORIES[0],
                "secondary_categories": [],
                "tags": [],
                "confidence": 0.0
            }
    
    def _find_similar_articles(self, embedding: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
        """Find k most similar articles"""
        if self.index.ntotal == 0:
            return []
        
        # Search
        distances, indices = self.index.search(
            embedding.reshape(1, -1).astype('float32'),
            min(k, self.index.ntotal)
        )
        
        # Convert distances to similarities (L2 distance -> similarity)
        similarities = 1 / (1 + distances[0])
        
        results = [(int(idx), float(sim)) for idx, sim in zip(indices[0], similarities)]
        return results
    
    def categorize_article(self, article: Dict, add_to_index: bool = True) -> Dict:
        """
        Categorize a single article.
        
        Args:
            article: Article dict with headline, summary_280, entities
            add_to_index: Whether to add this article to the index
        
        Returns:
            Article with categorization added
        """
        headline = article.get("headline", "")
        summary = article.get("summary_280", "")
        entities = article.get("entities", {})
        
        logger.info(f"Categorizing: {headline[:60]}...")
        
        # Generate embedding
        text_for_embedding = f"{headline} {summary}"
        embedding = self._generate_embedding(text_for_embedding)
        
        # Find similar articles
        similar = self._find_similar_articles(embedding, k=5)
        
        # Get categories from similar articles
        similar_categories = []
        for idx, sim in similar:
            if sim >= self.similarity_threshold and idx < len(self.metadata):
                meta = self.metadata[idx]
                if "categorization" in meta:
                    similar_categories.append(meta["categorization"])
        
        # Use LLM for categorization
        categorization = self._llm_categorize(headline, summary, entities)
        
        # Add similar articles info
        categorization["similar_articles"] = [
            {
                "index": idx,
                "similarity": sim,
                "headline": self.metadata[idx].get("headline", "")[:60] if idx < len(self.metadata) else ""
            }
            for idx, sim in similar[:3]
        ]
        
        # Add categorization to article
        article["categorization"] = categorization
        
        # Add to index
        if add_to_index:
            self.index.add(embedding.reshape(1, -1).astype('float32'))
            self.metadata.append({
                "headline": headline,
                "url": article.get("url", ""),
                "categorization": categorization
            })
        
        return article
    
    def categorize_batch(
        self,
        articles: List[Dict],
        output_path: str = "/data/processed/categorized_articles.json",
        save_index: bool = True
    ) -> List[Dict]:
        """
        Categorize batch of articles.
        
        Args:
            articles: List of article dicts
            output_path: Output file path
            save_index: Whether to save the FAISS index after processing
        
        Returns:
            List of categorized articles
        """
        categorized = []
        
        for i, article in enumerate(articles, 1):
            try:
                categorized_article = self.categorize_article(article, add_to_index=True)
                categorized.append(categorized_article)
                
                if i % 10 == 0:
                    logger.info(f"Progress: {i}/{len(articles)} articles categorized")
                    if save_index:
                        self._save_index()
                        
            except Exception as e:
                logger.error(f"Failed to categorize article {i}: {e}")
                categorized.append(article)  # Add uncategorized
                continue
        
        # Final save
        if save_index:
            self._save_index()
        
        # Write output
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(categorized, f, ensure_ascii=False, indent=2)
        
        logger.success(f"✅ Categorized {len(categorized)} articles → {output_path}")
        self._print_stats(categorized)
        
        return categorized
    
    @staticmethod
    def _print_stats(articles: List[Dict]):
        """Print categorization statistics"""
        from collections import Counter
        
        primary_cats = [
            a.get("categorization", {}).get("primary_category")
            for a in articles
            if a.get("categorization")
        ]
        
        logger.info("=" * 50)
        logger.info("CATEGORIZATION STATS")
        logger.info("=" * 50)
        logger.info(f"Total categorized: {len([a for a in articles if a.get('categorization')])}")
        logger.info("\nPrimary Categories:")
        for cat, count in Counter(primary_cats).most_common():
            logger.info(f"  {cat}: {count}")


def main():
    """CLI entry point for Agent 3"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Article Categorizer (Agent 3)")
    parser.add_argument("--input", required=True, help="Input JSON from Agent 2")
    parser.add_argument("--output", default="/data/processed/categorized_articles.json")
    parser.add_argument("--index-path", default="/data/embeddings/articles.index")
    parser.add_argument("--similarity-threshold", type=float, default=0.75)
    
    args = parser.parse_args()
    
    # Load input
    logger.info(f"Loading articles from {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        articles = json.load(f)
    logger.info(f"Loaded {len(articles)} articles")
    
    # Categorize
    categorizer = ArticleCategorizer(
        index_path=args.index_path,
        similarity_threshold=args.similarity_threshold
    )
    
    categorized = categorizer.categorize_batch(articles, output_path=args.output)
    
    logger.success(f"✅ Complete! Output: {args.output}")
    print(f"OUTPUT_FILE={args.output}")
    return args.output


if __name__ == "__main__":
    main()
