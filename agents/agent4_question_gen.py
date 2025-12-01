"""
Agent 4: HCP Poll Generator for Twitter/X
Generates HCP-focused polls based on categorized articles using Anthropic Claude.
Migrated from IBM Watson to Anthropic - includes complete Streamlit app logic:
- Semantic deduplication
- Diversity selection by bucket (practice impact, intent to use, patient selection, endpoints)
- Grounding validation (semantic similarity + entity overlap)
- Multi-pass generation with temperature jitter
"""

import os
import json
import re
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime

import numpy as np
from anthropic import Anthropic
from loguru import logger
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util

load_dotenv()


class PollGenerator:
    """
    Generate HCP-focused Twitter/X polls based on article content using Anthropic Claude.
    Includes semantic deduplication, diversity selection, and grounding validation.
    """
    
    MAX_POLL_CHARS = 280
    MAX_POLL_OPTIONS = 4
    MAX_POLLS_PER_ARTICLE = 4
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-5-20250929",
        polls_per_article: int = 3,
        temperature: float = 0.8,
        max_tokens: int = 480,
        similarity_threshold: float = 0.95,
        grounding_threshold: float = 0.75,
        entity_weight: float = 0.30,
        passes: int = 3
    ):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self.polls_per_article = min(polls_per_article, self.MAX_POLLS_PER_ARTICLE)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.similarity_threshold = similarity_threshold
        self.grounding_threshold = grounding_threshold
        self.entity_weight = entity_weight
        self.passes = passes
        
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
        
        self.client = Anthropic(api_key=self.api_key)
        
        # Initialize embedder for semantic operations
        logger.info("Loading sentence transformer for semantic operations...")
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        logger.info(f"Poll Generator initialized with {model}")
        logger.info(f"  - Polls per article: {self.polls_per_article}")
        logger.info(f"  - Generation passes: {self.passes}")
        logger.info(f"  - Similarity threshold: {self.similarity_threshold}")
        logger.info(f"  - Grounding threshold: {self.grounding_threshold}")
        logger.info(f"  - Entity weight: {self.entity_weight}")
    
    def _call_claude(self, prompt: str, pass_number: int = 1) -> str:
        """Call Anthropic API with temperature jitter for diversity"""
        try:
            # Add slight jitter to temperature based on pass number for diversity
            seed_jitter = (abs(hash(prompt[:100] + str(pass_number))) % 1000) / 1000.0
            jitter_temp = min(1.2, max(0.2, self.temperature + (seed_jitter - 0.5) * 0.15))
            
            message = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=jitter_temp,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text.strip()
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return ""
    
    @staticmethod
    def _normalize_ws(s: str) -> str:
        """Normalize whitespace"""
        return re.sub(r"[ \t]+", " ", s.strip())
    
    @staticmethod
    def _enforce_neutrality(s: str) -> str:
        """Remove promotional/hype phrases"""
        AVOID_PHRASES = [
            r"\bbreakthrough\b", r"\bmiracle\b", r"\bgame[- ]?changing\b",
            r"\bcure\b", r"\bguarantee(d)?\b", r"\bperfect\b", r"\bmust[- ]?have\b",
        ]
        for pat in AVOID_PHRASES:
            s = re.sub(pat, "", s, flags=re.IGNORECASE)
        return s
    
    @staticmethod
    def _trim_to_limit(s: str, limit: int = 280) -> str:
        """Trim poll to character limit while preserving structure"""
        if len(s) <= limit:
            return s
        
        lines = s.splitlines()
        if not lines:
            return s[:limit]
        
        q = lines[0]
        opts = [ln for ln in lines[1:] if ln.strip()]
        
        # Trim question if too long
        while len("\n".join([q] + opts)) > limit and len(q) > 40:
            q = q[:-10].rstrip(" .,:;")
        
        # Shorten options if needed
        def shorten(op):
            return (op[:80] + "…") if len(op) > 80 else op
        opts = [shorten(op) for op in opts]
        
        # Reduce number of options if still too long
        if len("\n".join([q] + opts)) > limit and len(opts) > 3:
            opts = opts[:3]
        
        out = "\n".join([q] + opts)
        return out[:limit]
    
    def _parse_polls_from_response(self, text: str) -> List[str]:
        """Parse poll blocks from LLM response"""
        text = text.strip()
        if not text:
            return []
        
        # Try to split by Q: patterns
        parts = re.split(r"\n(?=Q\d*\s*:|Question\s*:|Q\s*:)", text, flags=re.I)
        blocks = []
        
        for part in parts:
            p = part.strip()
            if p and re.match(r"^(Q\d*|Question|Q)\s*:", p, flags=re.I):
                blocks.append(p)
        
        # Fallback: look for question marks
        if not blocks:
            lines = [ln.rstrip() for ln in text.splitlines()]
            cur = []
            collecting = False
            
            for ln in lines:
                if ln.strip().endswith("?"):
                    if cur:
                        blocks.append("\n".join(cur))
                        cur = []
                    cur = [f"Q: {self._normalize_ws(ln)}"]
                    collecting = True
                elif collecting and re.match(r"^\s*([-•—]|\d+\.|[A-D]\))", ln):
                    cur.append("- " + re.sub(r"^\s*([-•—]|\d+\.|[A-D]\))\s*", "", ln).strip())
                elif collecting and ln.strip() == "":
                    if cur:
                        blocks.append("\n".join(cur))
                        cur = []
                    collecting = False
            
            if cur:
                blocks.append("\n".join(cur))
        
        # Clean and validate
        cleaned = []
        for b in blocks:
            lines = [self._normalize_ws(ln) for ln in b.splitlines() if ln.strip()]
            if not lines:
                continue
            
            # Ensure Q: prefix
            if not re.match(r"^Q\d*\s*:", lines[0], flags=re.I):
                lines[0] = "Q: " + re.sub(r"^(Q\d*|Question|Q)\s*:\s*", "", lines[0], flags=re.I)
            
            qline = lines[0]
            opts = [ln for ln in lines[1:] if re.match(r"^(-|•|—|\d+\.|[A-D]\))\s*", ln)]
            opts = ["- " + re.sub(r"^(-|•|—|\d+\.|[A-D]\))\s*", "", o).strip() for o in opts]
            
            # Must have at least 3 options
            if len(opts) < 3:
                continue
            
            # Limit to max options
            opts = opts[:self.MAX_POLL_OPTIONS]
            
            poll = "\n".join([qline] + opts)
            poll = self._enforce_neutrality(poll)
            poll = self._trim_to_limit(poll, self.MAX_POLL_CHARS)
            cleaned.append(poll)
        
        return cleaned
    
    def _dedup_semantic(self, blocks: List[str]) -> List[str]:
        """Remove semantically similar polls using embeddings"""
        if len(blocks) <= 1:
            return blocks
        
        embs = self.embedder.encode(blocks, normalize_embeddings=True)
        keep, kept = [], []
        
        for i, b in enumerate(blocks):
            if not kept:
                keep.append(b)
                kept.append(embs[i])
                continue
            
            sims = util.cos_sim(embs[i], np.stack(kept)).cpu().numpy()[0]
            if sims.max() < self.similarity_threshold:
                keep.append(b)
                kept.append(embs[i])
        
        return keep
    
    @staticmethod
    def _stem(line: str) -> str:
        """Stem a question line for similarity comparison"""
        s = line.lower()
        s = re.sub(r"^q\s*:\s*", "", s)
        s = re.sub(r"[^a-z0-9 ]+", " ", s)
        # Remove common stop words
        s = re.sub(r"\b(the|a|an|for|of|to|in|on|with|about|is|are|this|that|how|will|your|does|do)\b", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s
    
    @staticmethod
    def _bucket(q_text: str) -> str:
        """Classify question into a bucket for diversity"""
        t = q_text.lower()
        
        if "aware" in t or "awareness" in t:
            return "awareness"
        if ("practice-changing" in t) or ("practice changing" in t) or ("practice-informing" in t) or ("impact" in t):
            return "practice_impact"
        if ("will you use" in t) or ("use this" in t) or ("in your practice" in t) or ("consider using" in t):
            return "intent_to_use"
        if any(k in t for k in ["which patients", "patient", "line of therapy", "first-line", "adjuvant", "setting", "subtype", "biomarker"]):
            return "patient_selection"
        if any(k in t for k in ["pfs", "os", "orr", "endpoint", "discussions"]):
            return "endpoints"
        
        return "other"
    
    def _select_diverse_top4(self, polls: List[str]) -> List[str]:
        """
        Select up to MAX_POLLS_PER_ARTICLE diverse polls by:
        1. Semantic deduplication
        2. Bucket-based diversity (practice_impact, intent_to_use, patient_selection, endpoints)
        3. Stem-based uniqueness check
        """
        if not polls:
            return []
        
        # Normalize and deduplicate
        uniq = list(dict.fromkeys([self._normalize_ws(p) for p in polls]))
        uniq = self._dedup_semantic(uniq)
        
        # Prioritize buckets
        priority = ["practice_impact", "intent_to_use", "patient_selection", "endpoints", "other"]
        chosen: List[str] = []
        seen_stems: List[str] = []
        
        # Group by bucket
        by_bucket: Dict[str, List[str]] = {}
        for p in uniq:
            qline = p.splitlines()[0] if p else ""
            bucket = self._bucket(qline)
            by_bucket.setdefault(bucket, []).append(p)
        
        # Pick one from each bucket (priority order)
        for b in priority:
            for candidate in by_bucket.get(b, []):
                stem = self._stem(candidate.splitlines()[0])
                
                if not seen_stems:
                    chosen.append(candidate)
                    seen_stems.append(stem)
                    break
                
                # Check stem similarity with existing
                stem_emb = self.embedder.encode(stem, normalize_embeddings=True)
                seen_embs = self.embedder.encode(seen_stems, normalize_embeddings=True)
                if util.cos_sim(stem_emb, seen_embs).max().item() < 0.90:
                    chosen.append(candidate)
                    seen_stems.append(stem)
                    break
                
                if len(chosen) >= self.MAX_POLLS_PER_ARTICLE:
                    break
            
            if len(chosen) >= self.MAX_POLLS_PER_ARTICLE:
                break
        
        # Fill remaining slots if needed
        if len(chosen) < self.MAX_POLLS_PER_ARTICLE:
            for p in uniq:
                if p in chosen:
                    continue
                
                stem = self._stem(p.splitlines()[0])
                stem_emb = self.embedder.encode(stem, normalize_embeddings=True)
                seen_embs = self.embedder.encode(seen_stems, normalize_embeddings=True)
                
                if util.cos_sim(stem_emb, seen_embs).max().item() < 0.90:
                    chosen.append(p)
                    seen_stems.append(stem)
                
                if len(chosen) >= self.MAX_POLLS_PER_ARTICLE:
                    break
        
        return chosen[:self.MAX_POLLS_PER_ARTICLE]
    
    def _build_awareness_poll(self, entities: Dict) -> str:
        """Build simple awareness poll from entities (JSON-only, no LLM)"""
        drugs = entities.get("drug_names") or []
        inds = entities.get("indications") or []
        
        drug = drugs[0] if drugs else None
        ind = inds[0] if inds else None
        
        if drug and ind:
            q = f"Q: Were you aware of this {drug} development for {ind} before reading?"
        elif ind:
            q = f"Q: Were you aware of this {ind} development before reading?"
        else:
            q = f"Q: Were you aware of this development before reading?"
        
        poll = q + "\n- Yes, was aware\n- No, new information\n- Somewhat aware\n- Will research further"
        poll = self._enforce_neutrality(poll)
        poll = self._trim_to_limit(poll, self.MAX_POLL_CHARS)
        return poll
    
    # ==================== GROUNDING VALIDATION ====================
    
    @staticmethod
    def _chunk_text_for_similarity(text: str, chunk_size: int = 480, overlap: int = 120) -> List[str]:
        """Chunk text for semantic similarity calculation"""
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) <= chunk_size:
            return [text] if text else []
        
        chunks = []
        start = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            chunks.append(text[start:end])
            if end == len(text):
                break
            start = max(end - overlap, start + 1)
        
        return chunks
    
    @staticmethod
    def _normalize(txt: str) -> str:
        """Normalize text for comparison"""
        return re.sub(r"\s+", " ", (txt or "").strip().lower())
    
    @staticmethod
    def _contains_phrase(q: str, phrase: str) -> bool:
        """Check if normalized question contains phrase"""
        qn = PollGenerator._normalize(q)
        pn = PollGenerator._normalize(phrase)
        return pn and pn in qn
    
    def _entity_overlap_score(self, question_text: str, json_entities: Dict) -> float:
        """
        Calculate weighted entity overlap between question and article entities.
        Uses JSON-provided entities only (no inference).
        
        Weights:
        - drug_names: 0.5
        - indications: 0.3
        - trial_phases: 0.2
        """
        q = self._normalize(question_text)
        w_drug, w_ind, w_ph = 0.5, 0.3, 0.2
        total_w = 0.0
        score = 0.0
        
        # Drug names
        drugs = [d for d in (json_entities.get("drug_names") or []) if isinstance(d, str) and d.strip()]
        if drugs:
            total_w += w_drug
            if any(self._contains_phrase(q, d) for d in drugs):
                score += w_drug
        
        # Indications
        inds = [i for i in (json_entities.get("indications") or []) if isinstance(i, str) and i.strip()]
        if inds:
            total_w += w_ind
            present = any(self._contains_phrase(q, ind) or "tnbc" in q for ind in inds)
            if present:
                score += w_ind
        
        # Trial phases
        phases = [p for p in (json_entities.get("trial_phases") or []) if isinstance(p, str) and p.strip()]
        if phases:
            total_w += w_ph
            present = any(self._contains_phrase(q, p) or self._contains_phrase(q, p.replace("Phase ", "")) for p in phases)
            if present:
                score += w_ph
        
        # If no entities present, return neutral score
        if total_w == 0.0:
            return 0.5
        
        return score / total_w
    
    def _grounding_semantic_similarity(self, question_text: str, grounding_text: str) -> float:
        """Calculate semantic similarity between question and article content"""
        q = question_text.strip()
        if not q or not grounding_text:
            return 0.0
        
        q_emb = self.embedder.encode([q], normalize_embeddings=True)
        chunks = self._chunk_text_for_similarity(grounding_text, chunk_size=480, overlap=120)
        
        if not chunks:
            return 0.0
        
        c_embs = self.embedder.encode(chunks, normalize_embeddings=True)
        sims = util.cos_sim(q_emb, c_embs).cpu().numpy()[0]
        
        return float(np.max(sims))
    
    def _grounding_overall_score(self, question_text: str, grounding_text: str, json_entities: Dict) -> Dict[str, float]:
        """
        Calculate overall grounding score as weighted combination of:
        - Semantic similarity (question vs content)
        - Entity overlap (question vs JSON entities)
        
        Formula: overall = (1 - entity_weight) * semantic + entity_weight * entity
        """
        sem = self._grounding_semantic_similarity(question_text, grounding_text)
        ent = self._entity_overlap_score(question_text, json_entities)
        overall = (1.0 - self.entity_weight) * sem + self.entity_weight * ent
        
        return {
            "semantic": round(sem, 4),
            "entity": round(ent, 4),
            "overall": round(overall, 4)
        }
    
    # ==================== POLL GENERATION ====================
    
    def generate_polls(self, article: Dict) -> Tuple[List[Dict], List[Dict]]:
        """
        Generate HCP-focused polls for an article with grounding validation.
        
        Args:
            article: Article dict with headline, summary, entities, categorization, content
        
        Returns:
            Tuple of (poll_dicts, grounding_scores)
        """
        headline = article.get("headline", "")
        summary = article.get("summary_280", "")
        content = article.get("content", "")
        entities = article.get("entities", {})
        category = article.get("categorization", {}).get("primary_category", "")
        url = article.get("url", "")
        
        logger.info(f"Generating polls for: {headline[:60]}...")
        
        # Use content if available, otherwise summary
        grounding_text = content if content else summary
        if len(grounding_text) > 1600:
            grounding_text = grounding_text[:1600]
        
        # Build facts payload (JSON entities only, no inference)
        facts_payload = {
            "headline": headline,
            "url": url,
            "entities": {
                "drug_names": entities.get("drug_names", [])[:6],
                "company_name": entities.get("company_name", [])[:6] if isinstance(entities.get("company_name"), list) else ([entities.get("company_name")] if entities.get("company_name") else []),
                "trial_names": entities.get("trial_names", [])[:6],
                "trial_phases": entities.get("trial_phases", [])[:6],
                "indications": entities.get("indications", [])[:6],
            }
        }
        facts_str = json.dumps(facts_payload, ensure_ascii=False)
        
        # Build prompt
        prompt = f"""You are an assistant that writes Twitter/X poll questions to understand a **doctor's (HCP) perspective** on ONE breast cancer news article.

STRICT CONTENT RULES — READ CAREFULLY:
- USE ONLY information that appears in the headline or article content below (or in Key Facts, which directly mirrors the provided JSON). Do NOT invent drugs, companies, endpoints, biomarkers, or settings.
- Each poll MUST clearly reference the specific context present in the article (e.g., drug name, indication, phase/setting, relevant endpoint if mentioned).
- Produce AT MOST **{self.polls_per_article}** polls and make them mutually DISTINCT in focus (avoid paraphrases):
  • practice impact (practice-informing vs practice-changing)
  • intent to use in practice
  • patient selection / setting (line of therapy, subtype, biomarker)
  • endpoints / treatment discussions (e.g., PFS/OS/ORR if mentioned)
- Keep each poll UNDER 280 characters total (question + options) and include 3–4 concise options.
- Return COMPLETE polls only (no truncated options).
- Purpose: elicit the **doctor's perspective** (practice intent), not generic Q&A.

KEY FACTS (from JSON; do not add new facts):
{facts_str}

ARTICLE CONTENT:
{grounding_text}

OUTPUT FORMAT (up to {self.polls_per_article} polls; no extra text):
Q: <HCP-perspective question that strictly reflects article content>
- Option 1
- Option 2
- Option 3
- Option 4 (optional)
"""
        
        # Multi-pass generation
        all_blocks: List[str] = []
        for i in range(self.passes):
            logger.info(f"  Generation pass {i+1}/{self.passes}...")
            try:
                response = self._call_claude(prompt, pass_number=i+1)
                polls = self._parse_polls_from_response(response)
                all_blocks.extend(polls)
            except Exception as e:
                logger.warning(f"  Pass {i+1} failed: {e}")
                continue
        
        # Select diverse polls
        llm_polls = self._select_diverse_top4(all_blocks)
        
        # Always add awareness poll as first item
        awareness = self._build_awareness_poll(entities)
        all_polls = [awareness] + llm_polls
        
        # Deduplicate and limit
        seen = set()
        unique_polls = []
        for p in all_polls:
            normalized = self._normalize_ws(p)
            if normalized not in seen:
                seen.add(normalized)
                unique_polls.append(p)
        
        # Limit to max polls
        final_polls = unique_polls[:self.MAX_POLLS_PER_ARTICLE]
        
        # Calculate grounding scores
        grounding_scores = []
        poll_dicts = []
        
        for i, poll in enumerate(final_polls, 1):
            lines = [ln for ln in poll.splitlines() if ln.strip()]
            if not lines:
                continue
            
            question = lines[0].replace("Q:", "").strip()
            options = [ln[2:].strip() for ln in lines[1:] if ln.startswith("- ")]
            
            # Calculate grounding
            scores = self._grounding_overall_score(question, grounding_text, entities)
            needs_review = scores["overall"] < self.grounding_threshold
            
            poll_dict = {
                "poll_number": i,
                "question": question,
                "options": options,
                "article_url": url,
                "article_headline": headline,
                "article_summary": article.get("summary_280", ""),  # ← ADD THIS LINE
                "category": category,
                "char_count": len(poll),
                "poll_type": "awareness" if i == 1 else self._bucket(question),
                "grounding_score": {
                    **scores,
                    "threshold": self.grounding_threshold,
                    "needs_review": needs_review
                }
            }

            poll_dicts.append(poll_dict)
            grounding_scores.append(scores)
        
        logger.info(f"  Generated {len(poll_dicts)} polls")
        return poll_dicts, grounding_scores
    
    def generate_batch(
        self,
        articles: List[Dict],
        output_path: str = "data/outputs/twitter_polls.json"
    ) -> List[Dict]:
        """
        Generate polls for batch of articles with grounding validation.
        
        Args:
            articles: List of categorized articles
            output_path: Output file path
        
        Returns:
            List of all generated polls with grounding scores
        """
        all_polls = []
        flagged_count = 0
        
        for i, article in enumerate(articles, 1):
            try:
                polls, scores = self.generate_polls(article)
                all_polls.extend(polls)
                
                # Count flagged polls
                flagged = sum(1 for p in polls if p["grounding_score"]["needs_review"])
                flagged_count += flagged
                
                if i % 10 == 0:
                    logger.info(f"Progress: {i}/{len(articles)} articles, {len(all_polls)} polls, {flagged_count} flagged")
                    
            except Exception as e:
                logger.error(f"Failed to generate polls for article {i}: {e}")
                continue
        
        # Save output
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_polls, f, ensure_ascii=False, indent=2)
        
        logger.success(f"✅ Generated {len(all_polls)} polls → {output_path}")
        logger.warning(f"⚠️  {flagged_count} polls flagged for review (below grounding threshold {self.grounding_threshold})")
        self._print_stats(all_polls)
        
        return all_polls
    
    @staticmethod
    def _print_stats(polls: List[Dict]):
        """Print generation statistics"""
        from collections import Counter
        
        poll_types = [p.get("poll_type") for p in polls]
        categories = [p.get("category") for p in polls]
        flagged = [p for p in polls if p.get("grounding_score", {}).get("needs_review", False)]
        
        logger.info("=" * 50)
        logger.info("POLL GENERATION STATS")
        logger.info("=" * 50)
        logger.info(f"Total polls: {len(polls)}")
        logger.info(f"Flagged for review: {len(flagged)} ({len(flagged)/len(polls)*100:.1f}%)")
        
        avg_length = sum(p.get("char_count", 0) for p in polls) / len(polls) if polls else 0
        logger.info(f"Average length: {avg_length:.0f} chars")
        
        avg_semantic = sum(p.get("grounding_score", {}).get("semantic", 0) for p in polls) / len(polls) if polls else 0
        avg_entity = sum(p.get("grounding_score", {}).get("entity", 0) for p in polls) / len(polls) if polls else 0
        avg_overall = sum(p.get("grounding_score", {}).get("overall", 0) for p in polls) / len(polls) if polls else 0
        
        logger.info("\nGrounding Scores (avg):")
        logger.info(f"  Semantic similarity: {avg_semantic:.3f}")
        logger.info(f"  Entity overlap: {avg_entity:.3f}")
        logger.info(f"  Overall: {avg_overall:.3f}")
        
        logger.info("\nPoll Types:")
        for ptype, count in Counter(poll_types).most_common():
            logger.info(f"  {ptype}: {count}")
        
        logger.info("\nBy Category:")
        for cat, count in Counter(categories).most_common():
            logger.info(f"  {cat}: {count}")


def main():
    """CLI entry point for Agent 4"""
    import argparse
    
    parser = argparse.ArgumentParser(description="HCP Poll Generator (Agent 4)")
    parser.add_argument("--input", required=True, help="Input JSON from Agent 3")
    parser.add_argument("--output", default="data/outputs/twitter_polls.json")
    parser.add_argument("--polls-per-article", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--passes", type=int, default=3, help="Number of generation passes per article")
    parser.add_argument("--similarity-threshold", type=float, default=0.95)
    parser.add_argument("--grounding-threshold", type=float, default=0.75)
    parser.add_argument("--entity-weight", type=float, default=0.30)
    
    args = parser.parse_args()
    
    # Load input
    logger.info(f"Loading articles from {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        articles = json.load(f)
    logger.info(f"Loaded {len(articles)} articles")
    
    # Generate polls
    generator = PollGenerator(
        polls_per_article=args.polls_per_article,
        temperature=args.temperature,
        passes=args.passes,
        similarity_threshold=args.similarity_threshold,
        grounding_threshold=args.grounding_threshold,
        entity_weight=args.entity_weight
    )
    
    polls = generator.generate_batch(articles, output_path=args.output)
    
    logger.success(f"✅ Complete! Output: {args.output}")
    print(f"OUTPUT_FILE={args.output}")
    return args.output


if __name__ == "__main__":
    main()