# Breast Cancer News Intelligence Pipeline

Automated 5-agent pipeline for scraping, analyzing, and sharing breast cancer research updates with healthcare professionals via Twitter.

## 🎯 Overview

This pipeline automates the process of:
1. **Scraping** breast cancer news from medical sources
2. **Extracting** key entities (drugs, trials, biomarkers)
3. **Categorizing** content by medical relevance
4. **Generating** HCP-focused poll questions
5. **Publishing** to Twitter with quality filtering

**Platform:** Twitter (X) native polls  
**Target Audience:** Healthcare professionals (HCPs)  
**Automation:** Weekly via GitHub Actions

---

## 🏗️ Architecture

### 5-Agent Pipeline

```
Agent 1: Multi-Source Scraper
   ↓
Agent 2: Entity Extraction (Claude Sonnet 4)
   ↓
Agent 3: Content Categorization (FAISS + Claude)
   ↓
Agent 4: Poll Generation (Claude Sonnet 4.5)
   ↓
Agent 5: Twitter Publisher (Grounding-Filtered)
```

### Key Features

- ✅ **Multi-source scraping** (MedPage Today, AACR, Cancer.gov, etc.)
- ✅ **Semantic deduplication** (prevents duplicate polls)
- ✅ **Grounding score filtering** (quality control)
- ✅ **Twitter native polls** (3-tweet threads)
- ✅ **Automated weekly posting** (GitHub Actions)

---

## 📋 Prerequisites

- **Python:** 3.10-3.13
- **API Keys:**
  - Anthropic API (Claude)
  - Twitter API v2 (with OAuth 2.0)
  - Groq API (optional, for faster processing)

---

## 🚀 Installation

### 1. Clone Repository

```bash
git clone https://github.com/lasyapriyavavilala/breast-cancer-pipeline.git
cd breast-cancer-pipeline
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

Create `.env` file:

```bash
# Anthropic API (Claude)
ANTHROPIC_API_KEY=your_anthropic_key_here

# Twitter API v2
TWITTER_API_KEY=your_api_key
TWITTER_API_SECRET=your_api_secret
TWITTER_ACCESS_TOKEN=your_access_token
TWITTER_ACCESS_TOKEN_SECRET=your_access_token_secret
TWITTER_BEARER_TOKEN=your_bearer_token


```

### 5. Configure Data Sources

Create/edit these CSV files in `data/`:

**`data/pharma_urls.csv`:**
```csv
url,source_name,source_type,priority
https://www.medpagetoday.com/hematologyoncology,MedPage Today,rss,1
https://www.cancer.gov/news-events/cancer-currents-blog,Cancer.gov,html,2
```

**`data/keywords.csv`:**
```csv
keyword
breast cancer
HER2
triple negative
```

---

## 💻 Usage

### Local Testing

#### Quick Test (Use Existing Polls)
```bash
python run_pipeline.py \
  --skip-agent1 --skip-agent2 --skip-agent3 \
  --grounding-threshold 0.60 \
  --post-polls \
  --post-limit 2 \
  --no-dry-run
```

#### Full Pipeline (Scrape → Post)
```bash
python run_pipeline.py \
  --target 5 \
  --days-back 7 \
  --polls-per-article 2 \
  --grounding-threshold 0.60 \
  --post-polls \
  --post-limit 10 \
  --no-dry-run
```

#### Dry Run (No Posting)
```bash
python run_pipeline.py \
  --target 2 \
  --grounding-threshold 0.60 \
  --post-polls \
  --dry-run
```

---

## ⚙️ Configuration

### Pipeline Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `--target` | Max articles to scrape | 50 | 1-999 |
| `--days-back` | Days to look back | 7 | 1-365 |
| `--polls-per-article` | Polls per article | 3 | 1-10 |
| `--grounding-threshold` | Quality filter (0=low, 1=high) | 0.60 | 0.0-1.0 |
| `--post-limit` | Max polls to post | None | 1-999 |
| `--post-interval` | Minutes between posts | 3 | 1-15 |

### Example Configurations

**Conservative (High Quality):**
```bash
--target 3 \
--polls-per-article 2 \
--grounding-threshold 0.75 \
--post-limit 5
```

**Balanced (Recommended):**
```bash
--target 5 \
--polls-per-article 2 \
--grounding-threshold 0.60 \
--post-limit 10
```

**Aggressive (High Volume):**
```bash
--target 10 \
--polls-per-article 3 \
--grounding-threshold 0.50 \
--post-limit 20
```

---

## 🤖 GitHub Actions Automation

### Setup

1. **Add Secrets** to your repository:
   - Go to: `Settings → Secrets and variables → Actions`
   - Add:
     - `TWITTER_API_KEY`
     - `TWITTER_API_SECRET`
     - `TWITTER_ACCESS_TOKEN`
     - `TWITTER_ACCESS_TOKEN_SECRET`
     - `TWITTER_BEARER_TOKEN`
     - `ANTHROPIC_API_KEY`
     
2. **Workflow runs automatically** every Sunday at 8 AM UTC

3. **Manual trigger**: Go to `Actions → Weekly Breast Cancer Pipeline → Run workflow`

### Workflow Configuration

Edit `.github/workflows/weekly-pipeline.yml`:

```yaml
schedule:
  - cron: '0 8 * * 0'  # Sunday 8 AM UTC

run: |
  python run_pipeline.py \
    --target 5 \
    --days-back 7 \
    --polls-per-article 2 \
    --grounding-threshold 0.60 \
    --post-polls \
    --post-limit 10 \
    --post-interval 3 \
    --no-dry-run
```

**Modify these values** to change automation behavior!

---

## 📊 Pipeline Output

### Generated Files

```
data/
├── raw/
│   └── scraped_articles_20241203_185021.json    # Agent 1 output
├── processed/
│   ├── enhanced_articles.json                    # Agent 2 output
│   ├── enhanced_articles.ndjson                  # Agent 2 (streaming)
│   └── categorized_articles.json                 # Agent 3 output
├── outputs/
│   └── twitter_polls.json                        # Agent 4 output
├── embeddings/
│   ├── articles.index                            # FAISS index
│   └── metadata.pkl                              # Article metadata
└── pharma_news.db                                # Post tracking (SQLite)
```

### Twitter Output Format

**3-Tweet Thread:**

**Tweet 1:** URL + Company
```
https://article-url.com
Pfizer
```

**Tweet 2:** Summary
```
FDA approves new HER2-targeted therapy showing 40% 
improvement in progression-free survival for metastatic 
breast cancer patients...
```

**Tweet 3:** Native Poll (24 hours)
```
Were you aware of this FDA approval?

○ Yes, was aware
○ No, new information
○ Somewhat aware
○ Will research further
```

---

## 🔍 Quality Control

### Grounding Score System

Each poll is scored on:
- **Semantic similarity** (0.0-1.0): How well it matches article content
- **Entity overlap** (0.0-1.0): Mentions key drugs/trials/biomarkers
- **Overall score** (weighted average)

**Default threshold:** 0.60 (posts 70-80% of polls)

**Distribution:**
- **0.85+**: Top 5% (very high quality)
- **0.75+**: Top 15% (high quality)
- **0.60+**: Top 70% (good quality) ← **Default**
- **0.50+**: Top 90% (acceptable)

### Semantic Deduplication

Agent 4 automatically removes duplicate questions using:
- **Sentence embeddings** (all-MiniLM-L6-v2)
- **Cosine similarity** threshold: 0.95
- **Result:** No repetitive polls

---

## 📈 Monitoring

### Check Posts

- **Twitter:** [Your Twitter Profile](https://twitter.com/lasyavavilala15)
- **Database:** `data/pharma_news.db`
- **Logs:** GitHub Actions → Workflow runs

### View Stats

```bash
# Posts in last 7 days
sqlite3 data/pharma_news.db \
  "SELECT COUNT(*), AVG(grounding_score) 
   FROM posts 
   WHERE posted_at >= date('now', '-7 days')"
```

### Troubleshooting

**No polls posted:**
- Check grounding scores in `twitter_polls.json`
- Lower threshold: `--grounding-threshold 0.50`
- Check database for rate limiting

**Twitter API errors:**
- **429:** Rate limit (wait 15 min)
- **401:** Check credentials in `.env`
- **403:** App permissions issue

**Reset rate limiting:**
```bash
rm data/pharma_news.db
```

---

## 📁 Project Structure

```
breast-cancer-pipeline/
├── agents/
│   ├── agent1_scraper.py           # Multi-source scraper
│   ├── agent2_summarizer.py        # Entity extraction
│   ├── agent3_categorizer.py       # Categorization
│   ├── agent4_question_gen.py      # Poll generation
│   └── agent5_publisher.py         # Twitter publisher
├── data/
│   ├── pharma_urls.csv             # Source URLs
│   ├── keywords.csv                # Search keywords
│   └── outputs/                    # Generated files
├── .github/
│   └── workflows/
│       └── weekly-pipeline.yml     # Automation workflow
├── run_pipeline.py                 # Main orchestrator
├── requirements.txt                # Dependencies
├── .env                            # API keys (not in git)
└── README.md                       # This file
```

---

## 🛠️ Technical Details

### Technologies Used

- **LLMs:** Anthropic Claude Sonnet 4 & 4.5
- **Embeddings:** Sentence Transformers (all-MiniLM-L6-v2)
- **Vector DB:** FAISS (semantic search)
- **Database:** SQLite (post tracking)
- **Scraping:** BeautifulSoup4, Feedparser
- **Twitter API:** Tweepy (v2)
- **Automation:** GitHub Actions

### Agent Models

| Agent | Model | Purpose |
|-------|-------|---------|
| Agent 2 | Claude Sonnet 4 | Entity extraction |
| Agent 3 | Claude Sonnet 4 | Categorization |
| Agent 4 | Claude Sonnet 4.5 | Poll generation |

---

## 📊 Expected Performance

### Weekly Automation (Default Settings)

**Input:**
- Articles scraped: ~5
- Articles enhanced: ~5
- Articles categorized: ~5

**Output:**
- Polls generated: ~10
- Polls above threshold (0.60): ~7
- Polls posted to Twitter: ~7-10

**Runtime:**
- Total: ~15-20 minutes
- Scraping: ~3 min
- Processing: ~7 min
- Posting: ~10 min (with 3-min intervals)

**Cost (Anthropic API):**
- ~$0.10-0.20 per run
- ~$0.80-1.60 per month (weekly)

---

## 🔐 Security

- ✅ **Never commit `.env`** (in `.gitignore`)
- ✅ **Use GitHub Secrets** for automation
- ✅ **Rotate tokens** every 60 days
- ✅ **Monitor API usage** to detect breaches

---

## 📝 License

This project is for educational and research purposes.

---

## 👥 Contributors

- **Anjali Balram Mohanty** 
- **Jahnavi Lasyapriya Vavilala**
- **Rajasree Coomar**
- **Sruthi Keerthana Nuttakki**

---

## 🙏 Acknowledgments

- **Anthropic** - Claude API
- **Twitter** - API v2
- **Sentence Transformers** - Embeddings
- **FAISS** - Vector search

---





## 🎯 Future Enhancements

- [ ] LinkedIn company page posting (when page matures)
- [ ] Multi-language support
- [ ] Real-time notifications
- [ ] Analytics dashboard
- [ ] Custom poll templates

---

**Built with ❤️ for healthcare professionals**