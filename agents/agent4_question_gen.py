"""
Agent 4: Poll Generator for Twitter/X
Generates polls based on categorized articles using Anthropic Claude.

Routing by primary_category:

HCP polls (persona="hcp"):
- "Clinical Trial Results"  (with extra trial summary bullets)
- "FDA Approval/Regulatory"
- "Drug Development Pipeline"
- "Scientific Research"
- "Safety/Adverse Events"

Patient polls + highlights (persona="patient"):
- "Patient Access/Advocacy"

Meta only (no polls; just summary/entities/url for next agent):
- "Partnership/Collaboration"
- "Market/Commercial"
"""

import os
import json
import re
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import numpy as np
from anthropic import Anthropic
from loguru import logger
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util

load_dotenv()


class PollGenerator:
    """
    Generate Twitter/X polls based on article content using Anthropic Claude.
    Includes semantic deduplication, diversity selection, and grounding validation.

    Modes:
    - "hcp_polls"      : HCP-focused polls for clinical / scientific news
    - "patient_polls"  : Patient-focused highlights + poll
    - "meta_only"      : No polls; pass through metadata for next agent
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
        passes: int = 3,
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

        logger.info("Loading sentence transformer for semantic operations...")
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        logger.info(f"Poll Generator initialized with {model}")
        logger.info(f"  - Polls per article: {self.polls_per_article}")
        logger.info(f"  - Generation passes: {self.passes}")
        logger.info(f"  - Similarity threshold: {self.similarity_threshold}")
        logger.info(f"  - Grounding threshold: {self.grounding_threshold}")
        logger.info(f"  - Entity weight: {self.entity_weight}")

    # ==================== LLM CALL ====================

    def _call_claude(self, prompt: str, pass_number: int = 1) -> str:
        """Call Anthropic API with temperature jitter for diversity"""
        try:
            seed_jitter = (abs(hash(prompt[:100] + str(pass_number))) % 1000) / 1000.0
            jitter_temp = min(
                1.2, max(0.2, self.temperature + (seed_jitter - 0.5) * 0.15)
            )

            message = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=jitter_temp,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text.strip()
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return ""

    # ==================== TEXT NORMALIZATION HELPERS ====================

    @staticmethod
    def _normalize_ws(s: str) -> str:
        return re.sub(r"[ \t]+", " ", s.strip())

    @staticmethod
    def _enforce_neutrality(s: str) -> str:
        AVOID_PHRASES = [
            r"\bbreakthrough\b",
            r"\bmiracle\b",
            r"\bgame[- ]?changing\b",
            r"\bcure\b",
            r"\bguarantee(d)?\b",
            r"\bperfect\b",
            r"\bmust[- ]?have\b",
        ]
        for pat in AVOID_PHRASES:
            s = re.sub(pat, "", s, flags=re.IGNORECASE)
        return s

    @staticmethod
    def _trim_to_limit(s: str, limit: int = 280) -> str:
        if len(s) <= limit:
            return s

        lines = s.splitlines()
        if not lines:
            return s[:limit]

        q = lines[0]
        opts = [ln for ln in lines[1:] if ln.strip()]

        while len("\n".join([q] + opts)) > limit and len(q) > 40:
            q = q[:-10].rstrip(" .,:;")

        def shorten(op):
            return (op[:80] + "…") if len(op) > 80 else op

        opts = [shorten(op) for op in opts]

        if len("\n".join([q] + opts)) > limit and len(opts) > 3:
            opts = opts[:3]

        out = "\n".join([q] + opts)
        return out[:limit]

    # ==================== PROMPT BUILDERS ====================

    def _build_hcp_prompt(
        self,
        polls_per_article: int,
        facts_str: str,
        grounding_text: str,
        category: str,
    ) -> str:
        """
        Prompt for HCP-focused polls.
        For "Clinical Trial Results", also ask for a TRIAL_SUMMARY block.
        """
        base_rules = f"""You are an assistant that writes Twitter/X poll questions to understand a **doctor's (HCP) perspective** on ONE oncology news article.

STRICT CONTENT RULES — READ CAREFULLY:
- USE ONLY information that appears in the headline or article content below (or in Key Facts, which directly mirrors the provided JSON). Do NOT invent drugs, companies, endpoints, biomarkers, or settings.
- Each poll MUST clearly reference the specific context present in the article (e.g., drug name, indication, phase/setting, relevant endpoint if mentioned).
- Produce AT MOST **{polls_per_article}** polls and make them mutually DISTINCT in focus (avoid paraphrases):
  • practice impact (practice-informing vs practice-changing)
  • intent to use in practice
  • patient selection / setting (line of therapy, subtype, biomarker)
  • endpoints / treatment discussions (e.g., PFS/OS/ORR if mentioned)
- Keep each poll UNDER 280 characters total (question + options) and include 3–4 concise options.
- Return COMPLETE polls only (no truncated options).
- Purpose: elicit the **doctor's perspective** (practice intent), not generic Q&A.
"""

        if category == "Clinical Trial Results":
            extra = """
Because this article is about CLINICAL TRIAL RESULTS, do the following:

1) First output a brief TRIAL_SUMMARY section:
   - Primary endpoint: the main primary endpoint of the trial (e.g., "PFS", "OS"). If not clearly stated, write "not specified".
   - Patient population: the main population enrolled (e.g., "adult patients with metastatic TNBC previously treated with..."). If not clearly stated, write "not specified".

2) Then output the polls as usual.

KEY FACTS (from JSON; do not add new facts):
{facts_str}

ARTICLE CONTENT:
{grounding_text}

OUTPUT FORMAT (TRIALS):

TRIAL_SUMMARY:
- Primary endpoint: <short text>
- Patient population: <short text>

POLLS:
Q: <HCP-perspective question that strictly reflects article content>
- Option 1
- Option 2
- Option 3
- Option 4 (optional)
"""
            return base_rules + extra.format(
                facts_str=facts_str, grounding_text=grounding_text
            )

        # Non-trial HCP content: no trial summary
        return base_rules + f"""

KEY FACTS (from JSON; do not add new facts):
{facts_str}

ARTICLE CONTENT:
{grounding_text}

OUTPUT FORMAT (up to {polls_per_article} polls; no extra text):
Q: <HCP-perspective question that strictly reflects article content>
- Option 1
- Option 2
- Option 3
- Option 4 (optional)
"""

    def _build_patient_prompt(
        self,
        polls_per_article: int,
        facts_str: str,
        grounding_text: str,
    ) -> str:
        """Patient-story prompt: highlights + one poll about treatment change."""
        return f"""You are an assistant that reads a single patient story about cancer and summarizes the **patient experience**.

AUDIENCE:
- People living with the condition described (patients) and caregivers.
- Output must be easy to understand, avoiding heavy clinical jargon.

STRICT CONTENT RULES — READ CAREFULLY:
- USE ONLY information that appears in the patient story or Key Facts (JSON). Do NOT invent new drugs, side effects, or treatments.
- If a detail is not clearly stated, say "not specified" rather than guessing.
- Do NOT add extra commentary, advice, or emojis.

Your task has TWO parts:

1) PATIENT HIGHLIGHTS (bullet points)
   Summarize the story using exactly these four bullets:
   - Treatment: the current treatment / regimen the patient is on (drug name or regimen name). If unclear, write "not specified".
   - Duration: how long they have been on this current treatment (e.g., "6 months", "recently started"). If unclear, write "not specified".
   - Disease progression: whether the disease is described as improving, stable, or progressing/relapsing. If unclear, write "not specified".
   - Main concern: the main concern(s) the patient expresses about the current treatment (e.g., side effects, quality of life, access/affordability, emotional impact). If unclear, write "not specified".

2) ONE Twitter/X POLL about treatment change
   - The poll should ask whether patients in a similar situation:
     • feel open to switching treatment, AND/OR
     • have had their doctor suggest a change in treatment.
   - The poll is for patients to answer about their own experience.
   - Provide 3–4 concise answer options.
   - Keep the question + options UNDER 280 characters total.
   - The poll must stay neutral and must not recommend any specific drug.

KEY FACTS (from JSON; do not add new facts):
{facts_str}

PATIENT STORY CONTENT:
{grounding_text}

OUTPUT FORMAT — MUST MATCH EXACTLY:

PATIENT_HIGHLIGHTS:
- Treatment: <short text>
- Duration: <short text>
- Disease progression: <short text>
- Main concern: <short text>

POLL:
Q: <one question about switching / changing treatment, based on the story>
- Option 1
- Option 2
- Option 3
- Option 4 (optional)
"""

    # ==================== PARSING & DEDUP ====================

    def _parse_polls_from_response(self, text: str) -> List[str]:
        """Parse poll blocks from LLM response"""
        text = text.strip()
        if not text:
            return []

        parts = re.split(r"\n(?=Q\d*\s*:|Question\s*:|Q\s*:)", text, flags=re.I)
        blocks = []

        for part in parts:
            p = part.strip()
            if p and re.match(r"^(Q\d*|Question|Q)\s*:", p, flags=re.I):
                blocks.append(p)

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
                    cur.append(
                        "- "
                        + re.sub(r"^\s*([-•—]|\d+\.|[A-D]\))\s*", "", ln).strip()
                    )
                elif collecting and ln.strip() == "":
                    if cur:
                        blocks.append("\n".join(cur))
                        cur = []
                    collecting = False

            if cur:
                blocks.append("\n".join(cur))

        cleaned = []
        for b in blocks:
            lines = [self._normalize_ws(ln) for ln in b.splitlines() if ln.strip()]
            if not lines:
                continue

            if not re.match(r"^Q\d*\s*:", lines[0], flags=re.I):
                lines[0] = "Q: " + re.sub(
                    r"^(Q\d*|Question|Q)\s*:\s*", "", lines[0], flags=re.I
                )

            qline = lines[0]
            opts = [
                ln
                for ln in lines[1:]
                if re.match(r"^(-|•|—|\d+\.|[A-D]\))\s*", ln)
            ]
            opts = [
                "- " + re.sub(r"^(-|•|—|\d+\.|[A-D]\))\s*", "", o).strip()
                for o in opts
            ]

            if len(opts) < 3:
                continue

            opts = opts[: self.MAX_POLL_OPTIONS]

            poll = "\n".join([qline] + opts)
            poll = self._enforce_neutrality(poll)
            poll = self._trim_to_limit(poll, self.MAX_POLL_CHARS)
            cleaned.append(poll)

        return cleaned

    def _parse_patient_highlights(self, text: str) -> Optional[Dict[str, str]]:
        """Extract patient highlights block from patient-story output."""
        m = re.search(
            r"PATIENT_HIGHLIGHTS:\s*(.*?)\n\s*POLL:",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not m:
            return None

        block = m.group(1)
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]

        data = {
            "treatment": None,
            "duration": None,
            "disease_progression": None,
            "main_concern": None,
        }

        for ln in lines:
            low = ln.lower()
            if low.startswith("- treatment:"):
                data["treatment"] = ln.split(":", 1)[1].strip()
            elif low.startswith("- duration:"):
                data["duration"] = ln.split(":", 1)[1].strip()
            elif low.startswith("- disease progression:"):
                data["disease_progression"] = ln.split(":", 1)[1].strip()
            elif low.startswith("- main concern:"):
                data["main_concern"] = ln.split(":", 1)[1].strip()

        for k in data:
            if not data[k]:
                data[k] = "not specified"

        return data

    def _parse_trial_summary(self, text: str) -> Optional[Dict[str, str]]:
        """Extract trial summary bullets from Clinical Trial Results output."""
        m = re.search(
            r"TRIAL_SUMMARY:\s*(.*?)\n\s*POLLS:",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not m:
            return None

        block = m.group(1)
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]

        data = {
            "primary_endpoint": None,
            "patient_population": None,
        }

        for ln in lines:
            low = ln.lower()
            if low.startswith("- primary endpoint:"):
                data["primary_endpoint"] = ln.split(":", 1)[1].strip()
            elif low.startswith("- patient population:"):
                data["patient_population"] = ln.split(":", 1)[1].strip()

        for k in data:
            if not data[k]:
                data[k] = "not specified"

        return data

    def _dedup_semantic(self, blocks: List[str]) -> List[str]:
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
        s = line.lower()
        s = re.sub(r"^q\s*:\s*", "", s)
        s = re.sub(r"[^a-z0-9 ]+", " ", s)
        s = re.sub(
            r"\b(the|a|an|for|of|to|in|on|with|about|is|are|this|that|how|will|your|does|do)\b",
            " ",
            s,
        )
        s = re.sub(r"\s+", " ", s).strip()
        return s

    # ==================== BUCKETING (PERSONA-AWARE) ====================

    @staticmethod
    def _bucket(q_text: str, persona: str = "hcp") -> str:
        t = q_text.lower()

        if persona == "patient":
            if any(
                k in t
                for k in [
                    "symptom",
                    "side effect",
                    "side-effect",
                    "fatigue",
                    "pain",
                    "nausea",
                    "sleep",
                    "insomnia",
                    "appetite",
                ]
            ):
                return "symptom_burden"
            if any(
                k in t
                for k in [
                    "quality of life",
                    "daily life",
                    "day-to-day",
                    "day to day",
                    "activities",
                    "routine",
                ]
            ):
                return "quality_of_life"
            if any(
                k in t
                for k in [
                    "satisfied",
                    "satisfaction",
                    "helpful",
                    "working",
                    "control",
                    "benefit",
                    "helping",
                ]
            ):
                return "treatment_satisfaction"
            if "switch" in t or "change treatment" in t or "different treatment" in t:
                return "treatment_change"
            if any(
                k in t
                for k in [
                    "doctor",
                    "oncologist",
                    "nurse",
                    "hcp",
                    "healthcare team",
                    "discuss",
                    "talked about",
                    "conversation",
                ]
            ):
                return "doctor_communication"
            return "other_patient"

        if "aware" in t or "awareness" in t:
            return "awareness"
        if (
            "practice-changing" in t
            or "practice changing" in t
            or "practice-informing" in t
            or "impact" in t
        ):
            return "practice_impact"
        if (
            "will you use" in t
            or "use this" in t
            or "in your practice" in t
            or "consider using" in t
        ):
            return "intent_to_use"
        if any(
            k in t
            for k in [
                "which patients",
                "patient",
                "line of therapy",
                "first-line",
                "first line",
                "adjuvant",
                "setting",
                "subtype",
                "biomarker",
            ]
        ):
            return "patient_selection"
        if any(
            k in t
            for k in ["pfs", "os", "orr", "endpoint", "end point", "discussions"]
        ):
            return "endpoints"

        return "other"

    def _select_diverse_top4(self, polls: List[str], persona: str = "hcp") -> List[str]:
        if not polls:
            return []

        uniq = list(dict.fromkeys([self._normalize_ws(p) for p in polls]))
        uniq = self._dedup_semantic(uniq)

        if persona == "patient":
            priority = [
                "treatment_change",
                "treatment_satisfaction",
                "symptom_burden",
                "quality_of_life",
                "doctor_communication",
                "other_patient",
            ]
        else:
            priority = [
                "practice_impact",
                "intent_to_use",
                "patient_selection",
                "endpoints",
                "other",
            ]

        chosen: List[str] = []
        seen_stems: List[str] = []

        by_bucket: Dict[str, List[str]] = {}
        for p in uniq:
            qline = p.splitlines()[0] if p else ""
            bucket = self._bucket(qline, persona=persona)
            by_bucket.setdefault(bucket, []).append(p)

        for b in priority:
            for candidate in by_bucket.get(b, []):
                stem = self._stem(candidate.splitlines()[0])

                if not seen_stems:
                    chosen.append(candidate)
                    seen_stems.append(stem)
                    break

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

        return chosen[: self.MAX_POLLS_PER_ARTICLE]

    # ==================== AWARENESS POLL (HCP) ====================

    def _build_awareness_poll(self, entities: Dict) -> str:
        drugs = entities.get("drug_names") or []
        inds = entities.get("indications") or []

        drug = drugs[0] if drugs else None
        ind = inds[0] if inds else None

        if drug and ind:
            q = f"Q: Were you aware of this {drug} development for {ind} before reading?"
        elif ind:
            q = f"Q: Were you aware of this {ind} development before reading?"
        else:
            q = "Q: Were you aware of this development before reading?"

        poll = (
            q
            + "\n- Yes, was aware"
            + "\n- No, new information"
            + "\n- Somewhat aware"
            + "\n- Will research further"
        )
        poll = self._enforce_neutrality(poll)
        poll = self._trim_to_limit(poll, self.MAX_POLL_CHARS)
        return poll

    # ==================== GROUNDING VALIDATION ====================

    @staticmethod
    def _chunk_text_for_similarity(
        text: str, chunk_size: int = 480, overlap: int = 120
    ) -> List[str]:
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
        return re.sub(r"\s+", " ", (txt or "").strip().lower())

    @staticmethod
    def _contains_phrase(q: str, phrase: str) -> bool:
        qn = PollGenerator._normalize(q)
        pn = PollGenerator._normalize(phrase)
        return pn and pn in qn

    def _entity_overlap_score(self, question_text: str, json_entities: Dict) -> float:
        q = self._normalize(question_text)
        w_drug, w_ind, w_ph = 0.5, 0.3, 0.2
        total_w = 0.0
        score = 0.0

        drugs = [
            d
            for d in (json_entities.get("drug_names") or [])
            if isinstance(d, str) and d.strip()
        ]
        if drugs:
            total_w += w_drug
            if any(self._contains_phrase(q, d) for d in drugs):
                score += w_drug

        inds = [
            i
            for i in (json_entities.get("indications") or [])
            if isinstance(i, str) and i.strip()
        ]
        if inds:
            total_w += w_ind
            present = any(self._contains_phrase(q, ind) or "tnbc" in q for ind in inds)
            if present:
                score += w_ind

        phases = [
            p
            for p in (json_entities.get("trial_phases") or [])
            if isinstance(p, str) and p.strip()
        ]
        if phases:
            total_w += w_ph
            present = any(
                self._contains_phrase(q, p)
                or self._contains_phrase(q, p.replace("Phase ", ""))
                for p in phases
            )
            if present:
                score += w_ph

        if total_w == 0.0:
            return 0.5

        return score / total_w

    def _grounding_semantic_similarity(
        self, question_text: str, grounding_text: str
    ) -> float:
        q = question_text.strip()
        if not q or not grounding_text:
            return 0.0

        q_emb = self.embedder.encode([q], normalize_embeddings=True)
        chunks = self._chunk_text_for_similarity(
            grounding_text, chunk_size=480, overlap=120
        )

        if not chunks:
            return 0.0

        c_embs = self.embedder.encode(chunks, normalize_embeddings=True)
        sims = util.cos_sim(q_emb, c_embs).cpu().numpy()[0]

        return float(np.max(sims))

    def _grounding_overall_score(
        self, question_text: str, grounding_text: str, json_entities: Dict
    ) -> Dict[str, float]:
        sem = self._grounding_semantic_similarity(question_text, grounding_text)
        ent = self._entity_overlap_score(question_text, json_entities)
        overall = (1.0 - self.entity_weight) * sem + self.entity_weight * ent

        return {
            "semantic": round(sem, 4),
            "entity": round(ent, 4),
            "overall": round(overall, 4),
        }

    # ==================== MODE DETECTION ====================

    def _detect_mode(self, article: Dict) -> str:
        """
        Decide how to handle the article based on primary_category.

        Returns:
            "hcp_polls", "patient_polls", or "meta_only"
        """
        cat = (article.get("categorization", {}).get("primary_category") or "").strip()

        hcp_categories = {
            "Clinical Trial Results",
            "FDA Approval/Regulatory",
            "Drug Development Pipeline",
            "Scientific Research",
            "Safety/Adverse Events",
        }

        patient_categories = {
            "Patient Access/Advocacy",
        }

        meta_only_categories = {
            "Partnership/Collaboration",
            "Market/Commercial",
        }

        if cat in patient_categories:
            return "patient_polls"
        if cat in meta_only_categories:
            return "meta_only"
        if cat in hcp_categories:
            return "hcp_polls"

        # Default: treat as HCP content
        return "hcp_polls"

    # ==================== MAIN POLL GENERATION ====================

    def generate_polls(self, article: Dict) -> Tuple[List[Dict], List[Dict]]:
        """
        Generate polls or metadata for an article.

        Returns:
            (poll_dicts, grounding_scores)
        """
        headline = article.get("headline", "")
        summary = article.get("summary_280", "")
        content = article.get("content", "")
        entities = article.get("entities", {})
        category = article.get("categorization", {}).get("primary_category", "")
        url = article.get("url", "")

        mode = self._detect_mode(article)
        logger.info(f"Mode={mode} for article: {headline[:60]}...")

        # ========= META-ONLY CATEGORIES =========
        if mode == "meta_only":
            logger.info("Meta-only category: returning summary/entities/url without polls")
            meta_record = {
                "poll_number": 0,
                "question": "",
                "options": [],
                "article_url": url,
                "article_headline": headline,
                "article_summary": summary,
                "category": category,
                "char_count": 0,
                "poll_type": "meta_only",
                "audience": None,
                "entities": entities,
                "grounding_score": {
                    "semantic": 0.0,
                    "entity": 0.0,
                    "overall": 0.0,
                    "threshold": self.grounding_threshold,
                    "needs_review": False,
                },
            }
            return [meta_record], []

        # ========= COMMON PREP =========

        persona = "patient" if mode == "patient_polls" else "hcp"

        grounding_text = content if content else summary
        if len(grounding_text) > 1600:
            grounding_text = grounding_text[:1600]

        facts_payload = {
            "headline": headline,
            "url": url,
            "entities": {
                "drug_names": entities.get("drug_names", [])[:6],
                "company_name": entities.get("company_name", [])[:6]
                if isinstance(entities.get("company_name"), list)
                else (
                    [entities.get("company_name")]
                    if entities.get("company_name")
                    else []
                ),
                "trial_names": entities.get("trial_names", [])[:6],
                "trial_phases": entities.get("trial_phases", [])[:6],
                "indications": entities.get("indications", [])[:6],
            },
        }
        facts_str = json.dumps(facts_payload, ensure_ascii=False)

        all_blocks: List[str] = []
        patient_highlights: Optional[Dict[str, str]] = None
        trial_summary: Optional[Dict[str, str]] = None

        # ========= PATIENT STORIES =========
        if mode == "patient_polls":
            prompt = self._build_patient_prompt(
                self.polls_per_article, facts_str, grounding_text
            )
            logger.info("Patient persona: single generation pass")
            response = self._call_claude(prompt, pass_number=1)

            patient_highlights = self._parse_patient_highlights(response)
            polls = self._parse_polls_from_response(response)
            all_blocks.extend(polls)

            if len(all_blocks) > 1:
                all_blocks = all_blocks[:1]

        # ========= HCP CATEGORIES =========
        else:
            prompt = self._build_hcp_prompt(
                self.polls_per_article, facts_str, grounding_text, category=category
            )
            for i in range(self.passes):
                logger.info(f"HCP generation pass {i+1}/{self.passes}...")
                try:
                    response = self._call_claude(prompt, pass_number=i + 1)
                    if category == "Clinical Trial Results" and i == 0:
                        trial_summary = self._parse_trial_summary(response)
                    polls = self._parse_polls_from_response(response)
                    all_blocks.extend(polls)
                except Exception as e:
                    logger.warning(f"Pass {i+1} failed: {e}")
                    continue

        # ========= BUILD POLL TEXTS =========

        if mode == "patient_polls":
            llm_polls = self._select_diverse_top4(all_blocks, persona=persona)
            final_polls_text = llm_polls[: self.MAX_POLLS_PER_ARTICLE]
        else:  # HCP polls
            llm_polls = self._select_diverse_top4(all_blocks, persona=persona)
            awareness = self._build_awareness_poll(entities)
            all_polls = [awareness] + llm_polls

            seen = set()
            unique_polls = []
            for p in all_polls:
                normalized = self._normalize_ws(p)
                if normalized not in seen:
                    seen.add(normalized)
                    unique_polls.append(p)

            final_polls_text = unique_polls[: self.MAX_POLLS_PER_ARTICLE]

        # ========= CONVERT TO DICTS + GROUNDING =========

        grounding_scores = []
        poll_dicts = []

        for i, poll in enumerate(final_polls_text, 1):
            lines = [ln for ln in poll.splitlines() if ln.strip()]
            if not lines:
                continue

            question = lines[0].replace("Q:", "").strip()
            options = [ln[2:].strip() for ln in lines[1:] if ln.startswith("- ")]

            scores = self._grounding_overall_score(
                question, grounding_text, entities
            )
            needs_review = scores["overall"] < self.grounding_threshold

            if mode == "patient_polls":
                poll_type = (
                    self._bucket(question, persona=persona)
                    if i == 1
                    else "patient_other"
                )
            else:
                if i == 1:
                    poll_type = "awareness"
                else:
                    poll_type = self._bucket(question, persona=persona)

            poll_dict = {
                "poll_number": i,
                "question": question,
                "options": options,
                "article_url": url,
                "article_headline": headline,
                "article_summary": summary,
                "category": category,
                "char_count": len(poll),
                "poll_type": poll_type,
                "audience": persona,
                "grounding_score": {
                    **scores,
                    "threshold": self.grounding_threshold,
                    "needs_review": needs_review,
                },
            }

            if mode == "patient_polls" and i == 1:
                poll_dict["patient_highlights"] = patient_highlights

            if category == "Clinical Trial Results" and i == 1:
                poll_dict["trial_summary"] = trial_summary

            poll_dicts.append(poll_dict)
            grounding_scores.append(scores)

        logger.info(f"Generated {len(poll_dicts)} records for mode={mode}")
        return poll_dicts, grounding_scores

    # ==================== BATCH & STATS ====================

    def generate_batch(
        self,
        articles: List[Dict],
        output_path: str = "data/outputs/twitter_polls.json",
    ) -> List[Dict]:
        all_polls = []
        flagged_count = 0

        for i, article in enumerate(articles, 1):
            try:
                polls, _scores = self.generate_polls(article)
                all_polls.extend(polls)

                flagged = sum(
                    1 for p in polls if p["grounding_score"]["needs_review"]
                )
                flagged_count += flagged

                if i % 10 == 0:
                    logger.info(
                        f"Progress: {i}/{len(articles)} articles, "
                        f"{len(all_polls)} records, {flagged_count} flagged"
                    )

            except Exception as e:
                logger.error(f"Failed to process article {i}: {e}")
                continue

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_polls, f, ensure_ascii=False, indent=2)

        logger.success(f"✅ Generated {len(all_polls)} records → {output_path}")
        logger.warning(
            f"⚠️  {flagged_count} polls flagged for review "
            f"(below grounding threshold {self.grounding_threshold})"
        )
        self._print_stats(all_polls)

        return all_polls

    @staticmethod
    def _print_stats(polls: List[Dict]):
        from collections import Counter

        poll_types = [p.get("poll_type") for p in polls]
        categories = [p.get("category") for p in polls]
        audiences = [p.get("audience") for p in polls]
        flagged = [
            p for p in polls if p.get("grounding_score", {}).get("needs_review", False)
        ]

        logger.info("=" * 50)
        logger.info("POLL GENERATION STATS")
        logger.info("=" * 50)
        logger.info(f"Total records: {len(polls)}")
        logger.info(
            f"Flagged for review: {len(flagged)} "
            f"({(len(flagged)/len(polls)*100 if polls else 0):.1f}%)"
        )

        avg_length = (
            sum(p.get("char_count", 0) for p in polls) / len(polls) if polls else 0
        )
        logger.info(f"Average length (where applicable): {avg_length:.0f} chars")

        avg_semantic = (
            sum(p.get("grounding_score", {}).get("semantic", 0) for p in polls)
            / len(polls)
            if polls
            else 0
        )
        avg_entity = (
            sum(p.get("grounding_score", {}).get("entity", 0) for p in polls)
            / len(polls)
            if polls
            else 0
        )
        avg_overall = (
            sum(p.get("grounding_score", {}).get("overall", 0) for p in polls)
            / len(polls)
            if polls
            else 0
        )

        logger.info("\nGrounding Scores (avg, polls only):")
        logger.info(f"  Semantic similarity: {avg_semantic:.3f}")
        logger.info(f"  Entity overlap: {avg_entity:.3f}")
        logger.info(f"  Overall: {avg_overall:.3f}")

        logger.info("\nPoll / record types:")
        for ptype, count in Counter(poll_types).most_common():
            logger.info(f"  {ptype}: {count}")

        logger.info("\nAudiences:")
        for aud, count in Counter(audiences).most_common():
            logger.info(f"  {aud}: {count}")

        logger.info("\nBy Category:")
        for cat, count in Counter(categories).most_common():
            logger.info(f"  {cat}: {count}")


def main():
    """CLI entry point for Agent 4"""
    import argparse

    parser = argparse.ArgumentParser(description="Poll Generator (Agent 4)")
    parser.add_argument("--input", required=True, help="Input JSON from Agent 3")
    parser.add_argument("--output", default="data/outputs/twitter_polls.json")
    parser.add_argument("--polls-per-article", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument(
        "--passes", type=int, default=3, help="Number of generation passes per article"
    )
    parser.add_argument("--similarity-threshold", type=float, default=0.95)
    parser.add_argument("--grounding-threshold", type=float, default=0.75)
    parser.add_argument("--entity-weight", type=float, default=0.30)

    args = parser.parse_args()

    logger.info(f"Loading articles from {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        articles = json.load(f)
    logger.info(f"Loaded {len(articles)} articles")

    generator = PollGenerator(
        polls_per_article=args.polls_per_article,
        temperature=args.temperature,
        passes=args.passes,
        similarity_threshold=args.similarity_threshold,
        grounding_threshold=args.grounding_threshold,
        entity_weight=args.entity_weight,
    )

    generator.generate_batch(articles, output_path=args.output)

    logger.success(f"✅ Complete! Output: {args.output}")
    print(f"OUTPUT_FILE={args.output}")
    return args.output


if __name__ == "__main__":
    main()
