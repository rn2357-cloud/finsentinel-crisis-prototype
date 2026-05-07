import os
import re
import json
import time
import random
import hashlib
import warnings
from datetime import datetime, timezone
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# Optional dependencies. The app works with fallbacks if they are unavailable.
try:
    from openai import OpenAI
    OPENAI_OK = True
except Exception:
    OPENAI_OK = False

try:
    from rank_bm25 import BM25Okapi
    BM25_OK = True
except Exception:
    BM25_OK = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

try:
    import feedparser
    FEEDPARSER_OK = True
except Exception:
    FEEDPARSER_OK = False

try:
    from datasets import load_dataset
    HF_DATASETS_OK = True
except Exception:
    HF_DATASETS_OK = False

# Section 1: Imports and Configuration —  from notebook


CONFIG = {
    # Data
    "fpb_path":          "Sentences_AllAgree.txt",
    "fpb_encoding":      "latin-1",
    "label_col":         "Sentiment",
    "text_col":          "Headline",
    "random_state":      42,
    "test_size":         0.20,
    "llm_sample_size":   200,
    "agent_sample_size": 100,

    # TF-IDF
    "tfidf_max_features": 5000,
    "tfidf_ngram_range":  (1, 2),
    "cv_folds":           5,

    # FinBERT / LoRA names preserved from notebook
    "finbert_model":      "ProsusAI/finbert",
    "distilbert_model":   "distilbert-base-uncased",
    "lora_r":             8,
    "lora_alpha":         16,
    "lora_dropout":       0.1,
    "lora_epochs":        3,
    "lora_lr":            2e-4,
    "lora_batch_size":    16,

    # LLM prompting
    "llm_model":          "gpt-4o-mini",
    "llm_temperature":    0,
    "few_shot_n":         6,
    "rag_k":              3,
    "cot_n_paths":        5,

    # RAG corpus
    "corpus_size":        500,

    # Labels
    "label_map":          {"positive": 0, "neutral": 1, "negative": 2},
    "id2label":           {0: "positive", 1: "neutral", 2: "negative"},
    "label_colors":       {"positive": "#5DCAA5", "neutral": "#EF9F27", "negative": "#F0997B"},
}

# Notebook few-shot examples preserved
FEW_SHOT_EXAMPLES = {
    "positive": [
        "Operating profit rose to EUR 13.1 mn from EUR 8.7 mn in the same period last year.",
        "Net sales of the Group increased by 9 % to EUR 48.2 mn."
    ],
    "neutral": [
        "Tikkurila Powder Coatings has some 50 employees at its four paint plants.",
        "According to Gran, the company has no plans to move all production to Russia."
    ],
    "negative": [
        "Orion Corp reported a fall in earnings hit by larger R&D expenditures.",
        "The Group operating result fell to EUR -0.3 mn."
    ],
}

# FiQA-style examples added as project extension, but kept in the same headline/sentiment format.
FIQA_STYLE_EXAMPLES = [
    {"Headline": "Shares fell after the company warned that weak demand could pressure margins.", "Sentiment": "negative"},
    {"Headline": "The bank said credit losses may increase if commercial real estate weakens further.", "Sentiment": "negative"},
    {"Headline": "The firm announced a new buyback program after stronger-than-expected earnings.", "Sentiment": "positive"},
    {"Headline": "Management said macro uncertainty remains elevated but liquidity is stable.", "Sentiment": "neutral"},
]

# RAG seed corpus: FPB examples + FiQA-style examples + financial stress examples.
DEFAULT_CORPUS = []
for label, examples in FEW_SHOT_EXAMPLES.items():
    DEFAULT_CORPUS.extend(examples)
DEFAULT_CORPUS.extend([x["Headline"] for x in FIQA_STYLE_EXAMPLES])
DEFAULT_CORPUS.extend([
    "Silicon Valley Bank collapses after deposit outflows and liquidity concerns.",
    "Regional banks face downgrade warnings after funding stress intensifies.",
    "Corporate default rates rise as tighter credit conditions pressure borrowers.",
    "Treasury yields surge as investors price in higher-for-longer interest rates.",
    "Commercial real estate losses pressure regional lenders and raise contagion fears.",
    "Major technology firms announce layoffs amid slowing revenue growth.",
    "Oil prices jump after geopolitical tensions disrupt supply routes.",
    "Market volatility spikes as investors react to inflation surprise.",
])


# Section 3: Data loading


def load_fpb(path, encoding="latin-1"):
    """Parse the @-delimited FinancialPhraseBank format."""
    records = []
    with open(path, encoding=encoding) as f:
        for line in f:
            line = line.strip()
            if not line or "@" not in line:
                continue
            sentence, label = line.rsplit("@", 1)
            records.append({
                CONFIG["text_col"]:  sentence.strip(),
                CONFIG["label_col"]: label.strip().lower()
            })
    return pd.DataFrame(records)


@st.cache_data(show_spinner=False, ttl=3600)
def load_fiqa_from_huggingface(dataset_name="TheFinAI/fiqa-sentiment-classification", split="train", max_rows=200):
    """
    Load FiQA sentiment data directly from HuggingFace.

    Default dataset: TheFinAI/fiqa-sentiment-classification
    Common columns: sentence, target, aspect, score, type.
    The app converts the selected text column into the notebook's Headline format.
    """
    if not HF_DATASETS_OK:
        return pd.DataFrame(), "Install HuggingFace datasets first: pip install datasets"

    try:
        ds = load_dataset(dataset_name, split=split)
        df = ds.to_pandas().head(max_rows).reset_index(drop=True)
        return df, None
    except Exception as e:
        return pd.DataFrame(), f"Could not load FiQA from HuggingFace: {e}"


def infer_fiqa_text_column(df):
    """Pick the most likely text column from a HuggingFace FiQA dataframe."""
    preferred = ["sentence", "text", "headline", "query", "question", "doc"]
    lower_to_original = {str(c).lower(): c for c in df.columns}
    for name in preferred:
        if name in lower_to_original:
            return lower_to_original[name]
    # fallback: first object/string-like column
    for col in df.columns:
        if df[col].dtype == "object":
            return col
    return df.columns[0] if len(df.columns) else None

# Section 7: GPT Prompting — notebook function names preserved

@st.cache_resource(show_spinner=False)
def get_openai_client():
    if not OPENAI_OK or not os.environ.get("OPENAI_API_KEY"):
        return None
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def call_llm(prompt, temperature=0, model=None):
    """
    Call GPT-4o-mini with the given prompt.
    Returns parsed JSON dict or None on failure.
    Preserved from the notebook, but made Streamlit-safe.
    """
    client = get_openai_client()
    if client is None:
        return None
    model = model or CONFIG["llm_model"]
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=400
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except json.JSONDecodeError:
        return None
    except Exception as e:
        st.warning(f"API error, using fallback instead: {e}")
        time.sleep(1)
        return None


def run_llm_evaluation(prompt_fn, sample_df, desc="LLM"):
    """
    Notebook-compatible evaluation function.
    For app use, returns structured outputs and statuses.
    """
    outputs, statuses = [], []
    for _, row in sample_df.iterrows():
        result = call_llm(prompt_fn(row[CONFIG["text_col"]]))
        if result and "sentiment" in result:
            outputs.append(result)
            statuses.append("ok")
        else:
            outputs.append(None)
            statuses.append("parse_error")
        time.sleep(0.1)
    return outputs, statuses


def format_few_shot_block():
    """Build the 6-example block for few-shot and CoT prompts."""
    lines = ["\nExamples:"]
    for label, examples in FEW_SHOT_EXAMPLES.items():
        for ex in examples:
            lines.append(f'Headline: "{ex}"')
            lines.append(f'{{"sentiment": "{label}"}}\n')
    return "\n".join(lines)

FEW_SHOT_BLOCK = format_few_shot_block()

# This is how we want the model to output 
RISK_JSON_SCHEMA = '''{
  "sentiment": "positive | neutral | negative",
  "risk_type": "liquidity_risk | credit_risk | macro_risk | market_volatility | earnings_risk | regulatory_risk | geopolitical_risk | labor_operational_risk | none",
  "severity": "low | medium | high | extreme",
  "market_impact": "regional_banking | broad_market | technology | energy | real_estate | consumer | company_specific | none",
  "confidence": 0.0,
  "reasoning": "one concise sentence"
}'''

# Original notebook prompt preserved by name; output expanded for final project.
def build_zero_shot_prompt(headline):
    return f"""You are a financial sentiment analyst.

Task: Classify the sentiment of this financial headline FROM THE PERSPECTIVE
OF A RETAIL INVESTOR. Ask: "Does this news have a clear directional signal
for my investment, or is it just factual reporting?"

Now extend the notebook's sentiment task into structured financial risk extraction.
Allowed sentiment labels: ["positive", "neutral", "negative"]
Allowed risk_type labels: ["liquidity_risk", "credit_risk", "macro_risk", "market_volatility", "earnings_risk", "regulatory_risk", "geopolitical_risk", "labor_operational_risk", "none"]
Allowed severity labels: ["low", "medium", "high", "extreme"]
Allowed time_horizon labels: ["short_term", "medium_term", "long_term", "unclear"]
Return ONLY valid JSON with no explanation or markdown.

Headline: "{headline}"
Output: {RISK_JSON_SCHEMA}"""


def build_few_shot_prompt(headline):
    return f"""You are a financial sentiment analyst.

Task: Classify the sentiment FROM THE PERSPECTIVE OF A RETAIL INVESTOR.

Label definitions:
- positive: directional improvement — profit rose, sales grew, deal finalized
- neutral: factual reporting without clear directional outcome — announcing
  meetings, reporting figures without comparison, describing operations
- negative: directional decline — profit fell, loss reported, costs rose
{FEW_SHOT_BLOCK}
Now extend this same logic into structured risk extraction.
Return ONLY valid JSON using this schema:
{RISK_JSON_SCHEMA}

Headline: "{headline}"
Output:"""


def build_cot_prompt(headline):
    return f"""You are a financial sentiment analyst.

Task: Classify the sentiment FROM THE PERSPECTIVE OF A RETAIL INVESTOR.
Think step by step before classifying.
{FEW_SHOT_BLOCK}
Now analyze step by step:
Headline: "{headline}"

Think: Does this sentence describe a directional financial outcome, or is it
factual reporting? What would a retail investor conclude from this? What risk,
severity, market impact, and time horizon are implied?

Return ONLY valid JSON:
{RISK_JSON_SCHEMA}"""

# -----------------------------------------------------------------------------
# Section 8: RAG — notebook names preserved, lightweight deployable version
# -----------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def build_bm25_index(corpus_tuple):
    corpus = list(corpus_tuple)
    if BM25_OK:
        tokenized_corpus = [text.lower().split() for text in corpus]
        return BM25Okapi(tokenized_corpus), None, None
    if SKLEARN_OK:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(corpus)
        return None, vectorizer, matrix
    return None, None, None


def bm25_retrieve(query, k=10, corpus_texts=None):
    """Return top-k headlines by BM25 score. Falls back to TF-IDF if BM25 is unavailable."""
    corpus_texts = corpus_texts or DEFAULT_CORPUS
    bm25, vectorizer, matrix = build_bm25_index(tuple(corpus_texts))
    if bm25 is not None:
        scores = bm25.get_scores(query.lower().split())
        top_k = np.argsort(scores)[::-1][:k]
        return [(corpus_texts[i], float(scores[i])) for i in top_k]
    if vectorizer is not None:
        q_vec = vectorizer.transform([query])
        sims = cosine_similarity(q_vec, matrix).flatten()
        top_k = sims.argsort()[::-1][:k]
        return [(corpus_texts[i], float(sims[i])) for i in top_k]
    q_terms = set(query.lower().split())
    scored = []
    for text in corpus_texts:
        score = len(q_terms & set(text.lower().split())) / max(1, len(q_terms))
        scored.append((text, float(score)))
    return sorted(scored, key=lambda x: x[1], reverse=True)[:k]


def hybrid_retrieve(query, k=CONFIG["rag_k"], alpha=0.6, corpus_texts=None):
    """
    Hybrid BM25 + FAISS retrieval in notebook.
    Streamlit version preserves function name and BM25 side; uses BM25/TF-IDF fallback
    instead of expensive FinBERT FAISS embeddings so it runs locally.
    """
    corpus_texts = corpus_texts or DEFAULT_CORPUS
    results = bm25_retrieve(query, k=min(20, len(corpus_texts)), corpus_texts=corpus_texts)
    top_k = [(text, score) for text, score in results if text.strip().lower() != query.strip().lower()][:k]
    return top_k


def build_rag_cot_prompt(headline, corpus_texts=None):
    """
    Inject 3 retrieved similar headlines as context before CoT classification.
    Preserved from notebook, with expanded risk JSON output.
    """
    retrieved = hybrid_retrieve(headline, k=CONFIG["rag_k"], corpus_texts=corpus_texts)
    context_block = "\n".join([f"  - {text}" for text, _ in retrieved])

    return f"""You are a financial sentiment analyst.

Similar recent financial headlines for context:
{context_block}

Use the context above to help calibrate your investor-perspective judgment.
{FEW_SHOT_BLOCK}
Now analyze step by step:
Headline: "{headline}"

Think: Does this sentence describe a directional financial outcome for an investor?
How is it similar or different from the context headlines? What risk type,
severity, market impact, and time horizon are implied?

Return ONLY valid JSON:
{RISK_JSON_SCHEMA}"""


# Section 14: Live news — notebook functions preserved

NOISY_PUBLISHERS = {"benzinga", "motley fool", "seeking alpha sponsored",
                    "globenewswire", "businesswire", "pr newswire"}


def parse_pub_date(entry):
    """
    Parse RSS published date to UTC datetime.
    Handles multiple date formats and timezone offsets.
    """
    date_str = entry.get("published", "") or entry.get("updated", "")
    if not date_str:
        return None
    for fmt in ["%a, %d %b %Y %H:%M:%S %z",
                "%a, %d %b %Y %H:%M:%S GMT",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S%z"]:
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            continue
    return None


@st.cache_data(show_spinner=False, ttl=900)
def fetch_headlines(ticker, max_results=40):
    """
    Fetch headlines from Yahoo Finance RSS with:
    - Deduplication by content hash
    - Publisher noise filtering
    - UTC timestamp normalization
    Preserved from notebook.
    """
    if not FEEDPARSER_OK:
        return []

    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    try:
        feed = feedparser.parse(url)
    except Exception:
        return []

    seen_hashes = set()
    results = []

    for entry in feed.entries[:max_results]:
        headline = entry.get("title", "").strip()
        if not headline or len(headline) < 10:
            continue

        h_hash = hashlib.md5(
            re.sub(r'\s+', ' ', headline.lower()).encode()
        ).hexdigest()
        if h_hash in seen_hashes:
            continue
        seen_hashes.add(h_hash)

        source = entry.get("source", {}).get("value", "").lower()
        if any(noise in source for noise in NOISY_PUBLISHERS):
            continue

        pub_dt = parse_pub_date(entry)
        if pub_dt is None:
            pub_dt = datetime.now(timezone.utc)

        results.append({
            "ticker":       ticker,
            "headline":     headline,
            "pub_datetime": pub_dt,
            "pub_date":     pub_dt.date(),
            "pub_hour":     pub_dt.hour,
            "source":       source,
        })

    return results


@st.cache_data(show_spinner=False, ttl=900)
def fetch_live_news(tickers, max_results=20):
    all_headlines = []
    for ticker in tickers:
        rows = fetch_headlines(ticker, max_results=max_results)
        all_headlines.extend(rows)
        time.sleep(0.1)
    if not all_headlines:
        return pd.DataFrame(columns=["ticker", "headline", "pub_datetime", "pub_date", "pub_hour", "source"])
    return pd.DataFrame(all_headlines).drop_duplicates(subset=["headline"]).reset_index(drop=True)

# -----------------------------------------------------------------------------
# New structured risk fallback layer — only used if API is unavailable/fails
# -----------------------------------------------------------------------------

RISK_TAXONOMY = {
    "liquidity_risk": ["liquidity", "deposit", "withdrawal", "funding", "cash crunch", "bank run", "outflow"],
    "credit_risk": ["default", "downgrade", "debt", "bankruptcy", "insolvency", "loan loss", "credit", "delinquency"],
    "macro_risk": ["inflation", "fed", "rates", "recession", "unemployment", "gdp", "yield", "tightening"],
    "market_volatility": ["volatility", "selloff", "rout", "crash", "tumble", "plunge", "panic", "spike"],
    "earnings_risk": ["earnings", "guidance", "profit warning", "margin", "revenue miss", "forecast cut", "miss"],
    "regulatory_risk": ["sec", "regulator", "lawsuit", "antitrust", "probe", "investigation", "fine", "compliance"],
    "geopolitical_risk": ["war", "sanction", "tariff", "conflict", "geopolitical", "border", "military"],
    "labor_operational_risk": ["layoff", "strike", "shutdown", "supply chain", "production halt", "recall"],
}

MARKET_IMPACT_TERMS = {
    "regional_banking": ["regional bank", "bank", "lender", "deposit", "svb", "jpm"],
    "broad_market": ["s&p", "nasdaq", "dow", "market", "equities", "treasury", "yield", "stocks"],
    "technology": ["tech", "ai", "chip", "semiconductor", "nvidia", "apple", "microsoft", "google", "meta", "tesla"],
    "energy": ["oil", "gas", "crude", "energy", "opec"],
    "real_estate": ["real estate", "commercial property", "mortgage", "reit", "housing"],
    "consumer": ["consumer", "retail", "sales", "demand", "spending"],
}

NEGATIVE_TERMS = ["fall", "fell", "drop", "plunge", "loss", "miss", "default", "downgrade", "bankruptcy", "layoff", "decline", "fraud", "investigation", "liquidity", "crisis", "warning", "weak", "concern", "stress"]
POSITIVE_TERMS = ["rise", "rose", "gain", "beat", "surge", "growth", "profit", "record", "upgrade", "strong", "increase", "rally", "rebound", "raises guidance"]


def _best_match(text, taxonomy, default="none"):
    lower = text.lower()
    best_label, best_hits = default, []
    for label, terms in taxonomy.items():
        hits = [term for term in terms if term in lower]
        if len(hits) > len(best_hits):
            best_label, best_hits = label, hits
    return best_label, best_hits


def fallback_structured_risk(headline):
    lower = headline.lower()
    pos = sum(1 for w in POSITIVE_TERMS if w in lower)
    neg = sum(1 for w in NEGATIVE_TERMS if w in lower)
    sentiment = "negative" if neg > pos else "positive" if pos > neg else "neutral"
    risk_type, risk_hits = _best_match(headline, RISK_TAXONOMY, default="none")
    market_impact, impact_hits = _best_match(headline, MARKET_IMPACT_TERMS, default="company_specific")

    extreme_terms = ["collapse", "crisis", "bank run", "bankruptcy", "insolvency", "contagion", "panic"]
    high_terms = ["plunge", "tumble", "downgrade", "default", "investigation", "layoffs", "warning", "selloff"]
    medium_terms = ["concern", "pressure", "risk", "weak", "miss", "decline", "slowdown", "loss"]
    if any(t in lower for t in extreme_terms):
        severity = "extreme"
    elif any(t in lower for t in high_terms):
        severity = "high"
    elif any(t in lower for t in medium_terms) or risk_type != "none":
        severity = "medium"
    else:
        severity = "low"

    if any(t in lower for t in ["today", "now", "this week", "warning", "plunge", "spike", "downgrade"]):
        horizon = "short_term"
    elif any(t in lower for t in ["quarter", "guidance", "forecast", "outlook"]):
        horizon = "medium_term"
    elif any(t in lower for t in ["structural", "long-term", "secular"]):
        horizon = "long_term"
    else:
        horizon = "unclear"

    confidence = min(0.95, 0.55 + 0.07 * (pos + neg + len(risk_hits) + len(impact_hits)))
    return {
        "sentiment": sentiment,
        "risk_type": risk_type,
        "severity": severity,
        "market_impact": market_impact,
        "time_horizon": horizon,
        "confidence": round(float(confidence), 2),
        "reasoning": "Fallback keyword logic used because an LLM response was unavailable."
    }


def normalize_structured_output(result, headline):
    if not isinstance(result, dict):
        return fallback_structured_risk(headline)
    fallback = fallback_structured_risk(headline)
    allowed = {
        "sentiment": {"positive", "neutral", "negative"},
        "severity": {"low", "medium", "high", "extreme"},
        "time_horizon": {"short_term", "medium_term", "long_term", "unclear"},
    }
    out = fallback.copy()
    out.update(result)
    for key, values in allowed.items():
        val = str(out.get(key, "")).lower().strip()
        out[key] = val if val in values else fallback[key]
    try:
        out["confidence"] = round(float(out.get("confidence", fallback["confidence"])), 2)
    except Exception:
        out["confidence"] = fallback["confidence"]
    return out


def analyze_headline_with_notebook_pipeline(headline, prompt_mode, corpus_texts):
    if prompt_mode == "Notebook zero-shot → risk JSON":
        prompt = build_zero_shot_prompt(headline)
    elif prompt_mode == "Notebook few-shot → risk JSON":
        prompt = build_few_shot_prompt(headline)
    elif prompt_mode == "Notebook CoT → risk JSON":
        prompt = build_cot_prompt(headline)
    else:
        prompt = build_rag_cot_prompt(headline, corpus_texts=corpus_texts)

    llm_result = call_llm(prompt, temperature=CONFIG["llm_temperature"])
    structured = normalize_structured_output(llm_result, headline)
    retrieved = hybrid_retrieve(headline, k=CONFIG["rag_k"], corpus_texts=corpus_texts)
    structured.update({
        "headline": headline,
        "retrieved_context": " | ".join([text for text, _ in retrieved]),
        "used_llm": bool(llm_result),
    })
    return structured


# Streamlit wrapper

st.set_page_config(page_title="FinSentinel Risk App", layout="wide")

st.title("FinSentinel: Notebook-Exact Financial Risk Intelligence")
st.caption("Uses our notebook's prompting/RAG/live-news function structure, expanded from sentiment to structured risk JSON.")

with st.sidebar:
    st.header("Notebook Controls")
    prompt_mode = st.selectbox(
        "Prompting strategy",
        [
            "zero-shot → risk JSON",
            "few-shot → risk JSON",
            "CoT → risk JSON",
            "RAG+CoT → risk JSON",
        ],
        index=3,
    )
    tickers_raw = st.text_input("Live Yahoo Finance tickers", "AAPL,MSFT,GOOGL,TSLA,NVDA,JPM,META")
    max_per_ticker = st.slider("Max headlines per ticker", 3, 30, 8)
    st.markdown("---")
    st.subheader("HuggingFace FiQA")
    hf_dataset_name = st.text_input("FiQA dataset name", "TheFinAI/fiqa-sentiment-classification")
    hf_split = st.selectbox("FiQA split", ["train", "valid", "test"], index=0)
    hf_max_rows = st.slider("Max FiQA rows", 10, 500, 100, step=10)
    st.markdown("---")
    st.write("OpenAI API detected:", "yes" if os.environ.get("OPENAI_API_KEY") and OPENAI_OK else "fallback mode")
    st.write("RSS parser detected:", "yes" if FEEDPARSER_OK else "install feedparser")
    st.write("BM25 detected:", "yes" if BM25_OK else "TF-IDF fallback")
    st.write("HF datasets detected:", "yes" if HF_DATASETS_OK else "install datasets")

input_mode = st.radio(
    "Input source",
    ["Live Yahoo Finance headlines","Paste headlines", "HuggingFace FiQA dataset", "Financial PhraseBank dataset"],
    horizontal=True,
)

if input_mode == "Live Yahoo Finance headlines":
    tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
    news_df = fetch_live_news(tickers, max_results=max_per_ticker)
    if news_df.empty:
        st.warning("Could not fetch live headlines. Paste headlines or install feedparser: pip install feedparser")
        input_df = pd.DataFrame({"Headline": DEFAULT_CORPUS[:8]})
    else:
        st.subheader("Fetched live headlines")
        st.dataframe(news_df[["ticker", "headline", "pub_date", "pub_hour", "source"]], use_container_width=True, height=250)
        input_df = news_df.rename(columns={"headline": "Headline"})[["Headline"]]
elif input_mode == "HuggingFace FiQA dataset":
    fiqa_df, fiqa_error = load_fiqa_from_huggingface(
        dataset_name=hf_dataset_name,
        split=hf_split,
        max_rows=hf_max_rows,
    )
    if fiqa_error:
        st.warning(fiqa_error)
        input_df = pd.DataFrame({"Headline": DEFAULT_CORPUS[:8]})
    elif fiqa_df.empty:
        st.warning("FiQA loaded but returned no rows. Using notebook demo examples instead.")
        input_df = pd.DataFrame({"Headline": DEFAULT_CORPUS[:8]})
    else:
        st.subheader("FiQA from HuggingFace")
        st.caption("Default source: TheFinAI/fiqa-sentiment-classification. It includes financial text plus fields such as target/aspect/score/type when available.")
        st.dataframe(fiqa_df.head(20), use_container_width=True, height=260)

        default_text_col = infer_fiqa_text_column(fiqa_df)
        default_index = list(fiqa_df.columns).index(default_text_col) if default_text_col in list(fiqa_df.columns) else 0
        text_col = infer_fiqa_text_column(fiqa_df)

        input_df = pd.DataFrame({"Headline": fiqa_df[text_col].astype(str)})

        # Preserve available FiQA metadata for display/download later.
        for optional_col in ["target", "aspect", "score", "type"]:
            if optional_col in fiqa_df.columns:
                input_df[f"fiqa_{optional_col}"] = fiqa_df[optional_col].values
elif input_mode == "Paste headlines":
    raw = st.text_area(
        "Paste one headline per line",
        "Regional banks face liquidity concerns after deposit outflows accelerate.\nCorporate default rates rise as tighter credit conditions pressure borrowers.\nNvidia shares rally after earnings beat expectations.",
        height=180,
    )
    input_df = pd.DataFrame({"Headline": [x.strip() for x in raw.splitlines() if x.strip()]})
else:
    input_df = pd.DataFrame({"Headline": DEFAULT_CORPUS[:12]})

# Notebook RAG corpus = FPB examples + FiQA-style + live/current input.
corpus_texts = list(dict.fromkeys(DEFAULT_CORPUS + input_df["Headline"].dropna().astype(str).tolist()))

st.markdown("### Current task output schema")
st.code(RISK_JSON_SCHEMA, language="json")

if st.button("Run risk analysis", type="primary", use_container_width=True):
    rows = []
    progress = st.progress(0)
    for i, headline in enumerate(input_df["Headline"].dropna().astype(str).tolist()):
        rows.append(analyze_headline_with_notebook_pipeline(headline, prompt_mode, corpus_texts))
        progress.progress((i + 1) / max(1, len(input_df)))
    results_df = pd.DataFrame(rows)
    # if the selected input came from FiQA, attach available FiQA metadata next to model outputs.
    fiqa_meta_cols = [c for c in input_df.columns if c.startswith("fiqa_")]
    if fiqa_meta_cols and len(results_df) == len(input_df):
        for c in fiqa_meta_cols:
            results_df[c] = input_df[c].values
    st.session_state["results_df"] = results_df

results_df = st.session_state.get("results_df")
if results_df is None:
    st.info("Click the button to run the pipeline.")
    st.stop()

# KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Headlines analyzed", len(results_df))
c2.metric("Negative share", f"{(results_df['sentiment'].eq('negative').mean()*100):.0f}%")
c3.metric("High/extreme severity", int(results_df["severity"].isin(["high", "extreme"]).sum()))
c4.metric("LLM used", "Yes" if results_df["used_llm"].any() else "Fallback")

st.subheader("Structured Financial Risk Outputs")
base_cols = ["headline", "sentiment", "risk_type", "severity", "market_impact", "time_horizon", "confidence", "reasoning", "retrieved_context"]
fiqa_cols = [c for c in results_df.columns if c.startswith("fiqa_")]
cols = base_cols + fiqa_cols
st.dataframe(results_df[cols], use_container_width=True, height=420)

left, right = st.columns(2)
with left:
    st.subheader("Risk Type Counts")
    st.bar_chart(results_df["risk_type"].value_counts())
with right:
    st.subheader("Market Impact Counts")
    st.bar_chart(results_df["market_impact"].value_counts())

st.subheader("Notebook Method Trace")
st.markdown(
    """
This app keeps our notebook's structure:
- `CONFIG`, `FEW_SHOT_EXAMPLES`, `FEW_SHOT_BLOCK`
- `call_llm()` and `run_llm_evaluation()`
- `build_zero_shot_prompt()`, `build_few_shot_prompt()`, `build_cot_prompt()`
- `bm25_retrieve()`, `hybrid_retrieve()`, `build_rag_cot_prompt()`
- `parse_pub_date()` and `fetch_headlines()` for Yahoo Finance RSS
- `load_fiqa_from_huggingface()` for direct FiQA dataset loading from HuggingFace

The main extension is that the original notebook sentiment JSON is expanded into multi-dimensional financial risk intelligence.
"""
)

csv = results_df.to_csv(index=False).encode("utf-8")
st.download_button("Download structured risk outputs", csv, "finsentinel_structured_risk_outputs.csv", "text/csv")
