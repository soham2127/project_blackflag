from flask import Flask, request, jsonify, send_from_directory, make_response
from flask_cors import CORS
import os
import json
import pandas as pd
import re
import joblib
from io import StringIO
import sys
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.sentiment import SentimentIntensityAnalyzer
import requests
import platform
import threading
from dotenv import load_dotenv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Load environment variables from .env (local development)
load_dotenv(override=True)
print(f"Env loaded: LLM_MODEL={os.getenv('LLM_MODEL')}, OPENROUTER_API_KEY set={bool(os.getenv('OPENROUTER_API_KEY'))}")

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from column_analyzer import analyze_columns_with_llm
from generate_predictions import generate_predictions
from keyword_drivers import compute_mean_tfidf, top_keywords, plot_bar
from aspect_sentiment_rules import process_aspects, build_summary, ASPECT_KEYWORDS
from component_failure_analysis import analyze_product_failures
from top_products_breakdown import get_product_keywords

# Download NLTK data if needed
try:
    import nltk
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    import nltk
    nltk.download('vader_lexicon')

# Download punkt_tab tokenizer if needed (used in aspect sentiment analysis)
try:
    import nltk
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    import nltk
    nltk.download('punkt_tab')

app = Flask(__name__)

CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    supports_credentials=True,
    allow_headers=["Content-Type", "Authorization"],
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"]
)


# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUTS_DIR = os.path.join(os.path.dirname(BASE_DIR), 'outputs')

# Create necessary directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def clean_data_pipeline(df, column_mapping):
    """Clean the data based on column mapping"""
    # Reverse mapping
    col_map = {v: k for k, v in column_mapping.items() if v is not None}
    
    # Rename columns
    df = df.rename(columns=col_map)

    # Collapse any duplicated columns (e.g., Review coming from both Review and text)
    if df.columns.duplicated().any():
        for name in df.columns[df.columns.duplicated()].unique():
            dup_cols = df.loc[:, name]
            # If selecting by label returns a Series, skip
            if isinstance(dup_cols, pd.Series):
                continue
            merged = dup_cols.apply(
                lambda row: next((x for x in row if pd.notna(x) and str(x).strip() != ''), ''),
                axis=1
            )
            df = df.drop(columns=[c for c in dup_cols.columns])
            df[name] = merged
    
    # Keep only required columns
    required_cols = ["ProductName", "Price", "Rate", "Review", "Summary"]
    existing_cols = [col for col in required_cols if col in df.columns]
    df = df[existing_cols].copy()
    
    # Fill missing columns
    for col in required_cols:
        if col not in df.columns:
            df[col] = ""
    
    # Normalize text
    for col in ["Review", "Summary"]:
        df[col] = df[col].fillna("").astype(str).str.strip()
    
    # Drop empty rows
    empty_mask = (df["Review"] == "") & (df["Summary"] == "")
    df = df.loc[~empty_mask].copy()
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Validate Rate
    df["Rate"] = pd.to_numeric(df["Rate"], errors="coerce")
    df = df[df["Rate"].isin([1, 2, 3, 4, 5])]
    df["Rate"] = df["Rate"].astype(int)
    
    # Clean Price
    if "Price" in df.columns:
        df["Price"] = df["Price"].astype(str).str.replace(r'[^\d]', '', regex=True)
        df["Price"] = pd.to_numeric(df["Price"], errors='coerce')
    
    # Create text column
    df["text"] = df["Summary"] + " " + df["Review"]
    
    # Clean text
    def clean_text(text):
        if not isinstance(text, str):
            text = "" if pd.isna(text) else str(text)
        text = text.lower()
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"https?://\S+|www\.\S+", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"[\n\t\r]+", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    
    for col in ["Summary", "Review", "text"]:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)
    
    # Map sentiment
    def map_sentiment(r):
        if r >= 4: return "Positive"
        if r == 3: return "Neutral"
        return "Negative"
    
    df["sentiment"] = df["Rate"].apply(map_sentiment)
    
    return df


def generate_ml_predictions(df):
    """Generate ML predictions using trained model"""
    model_path = os.path.join(MODELS_DIR, 'logistic_model.joblib')
    vectorizer_path = os.path.join(MODELS_DIR, 'tfidf_vectorizer.joblib')
    
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        X = vectorizer.transform(df['text'])
        df['predicted_sentiment'] = model.predict(X)
    else:
        print("WARNING: Models not found; skipping ML predictions and returning input DF")
        df['predicted_sentiment'] = df.get('sentiment', 'Neutral')
    return df

# ==========================================
# WRAPPER FUNCTIONS FOR SRC MODULES
# ==========================================

def extract_keywords_wrapper():
    """Extract keywords using src/keyword_drivers.py"""
    tfidf = joblib.load(os.path.join(MODELS_DIR, 'tfidf_vectorizer.joblib'))
    predictions_path = os.path.join(OUTPUTS_DIR, 'predictions.csv')
    df = pd.read_csv(predictions_path)
    
    # Filter by sentiment
    df = df[df["predicted_sentiment"].isin(["Positive", "Negative"])].copy()
    
    pos_texts = df.loc[df["predicted_sentiment"] == "Positive", "text"].dropna()
    neg_texts = df.loc[df["predicted_sentiment"] == "Negative", "text"].dropna()
    
    # Safeguard: Check if we have enough data
    MIN_REVIEWS = 5
    
    # Extract positive keywords if enough data
    if len(pos_texts) >= MIN_REVIEWS:
        feature_names_pos, scores_pos, means_pos, dfreq_pos, dfreq_pct_pos = compute_mean_tfidf(pos_texts, tfidf)
        pos_df = top_keywords(feature_names_pos, scores_pos, means_pos, dfreq_pos, dfreq_pct_pos, top_n=20)
    else:
        print(f"WARNING: Only {len(pos_texts)} positive reviews (min {MIN_REVIEWS}), skipping positive keywords")
        pos_df = pd.DataFrame(columns=['word', 'mean_tfidf', 'doc_frequency', 'doc_frequency_pct'])
    
    # Extract negative keywords if enough data
    if len(neg_texts) >= MIN_REVIEWS:
        feature_names_neg, scores_neg, means_neg, dfreq_neg, dfreq_pct_neg = compute_mean_tfidf(neg_texts, tfidf)
        neg_df = top_keywords(feature_names_neg, scores_neg, means_neg, dfreq_neg, dfreq_pct_neg, top_n=20)
    else:
        print(f"WARNING: Only {len(neg_texts)} negative reviews (min {MIN_REVIEWS}), skipping negative keywords")
        neg_df = pd.DataFrame(columns=['word', 'mean_tfidf', 'doc_frequency', 'doc_frequency_pct'])
    
    return pos_df, neg_df


def analyze_aspects_wrapper(nrows=None):
    """Analyze aspects using src/aspect_sentiment_rules.py"""
    predictions_path = os.path.join(OUTPUTS_DIR, 'predictions.csv')
    df = pd.read_csv(predictions_path)
    
    # Sample if nrows specified
    if nrows is not None and len(df) > nrows:
        df = df.sample(n=nrows, random_state=42)
    tfidf = joblib.load(os.path.join(MODELS_DIR, 'tfidf_vectorizer.joblib'))
    model = joblib.load(os.path.join(MODELS_DIR, 'logistic_model.joblib'))

    # Ensure required columns exist
    if 'text' not in df.columns:
        return pd.DataFrame(columns=['aspect', 'Positive', 'Neutral', 'Negative', 'Not Mentioned'])

    # Process aspects and build summary
    enriched_df = process_aspects(df, tfidf, model)
    summary_df = build_summary(enriched_df)

    return summary_df


def analyze_component_failures_wrapper(nrows=None):
    """Analyze component failures using src/component_failure_analysis.py"""
    predictions_path = os.path.join(OUTPUTS_DIR, 'predictions.csv')
    df = pd.read_csv(predictions_path)
    
    # Sample if nrows specified
    if nrows is not None and len(df) > nrows:
        df = df.sample(n=nrows, random_state=42)

    # Align column names expected by src/component_failure_analysis.py
    df = df.rename(columns={'predicted_sentiment': 'sentiment_pred'})

    results = []
    if 'ProductName' in df.columns:
        for product in df['ProductName'].unique():
            failures = analyze_product_failures(product, df)
            if failures:
                results.append({
                    'product': product,
                    'components': ', '.join(failures)
                })

    if not results:
        return pd.DataFrame(columns=['product', 'components'])

    return pd.DataFrame(results)


def analyze_top_products_wrapper():
    """Analyze top products using src/top_products_breakdown.py"""
    predictions_path = os.path.join(OUTPUTS_DIR, 'predictions.csv')
    df = pd.read_csv(predictions_path)
    tfidf = joblib.load(os.path.join(MODELS_DIR, 'tfidf_vectorizer.joblib'))

    # Align column names expected by src/top_products_breakdown.py
    df = df.rename(columns={'predicted_sentiment': 'sentiment_pred'})
    
    empty_cols = [
        'ProductName', 'ReviewCount', 'Positive', 'Negative',
        'Neutral', 'TopPositiveKeywords', 'TopNegativeKeywords'
    ]

    if 'ProductName' not in df.columns or df['ProductName'].isna().all():
        return pd.DataFrame(columns=empty_cols)
    
    # Get top 10 products by review count and filter invalid names
    top_products = df['ProductName'].value_counts().head(10).index.tolist()
    top_products = [p for p in top_products if pd.notna(p) and str(p).strip() and str(p).strip().lower() != 'nan']
    
    results = []
    for product in top_products:
        product_df = df[df['ProductName'] == product]
        keywords = get_product_keywords(product, product_df, tfidf)
        
        sentiment_counts = product_df['sentiment_pred'].value_counts()
        results.append({
            'ProductName': product,
            'ReviewCount': len(product_df),
            'Positive': sentiment_counts.get('Positive', 0),
            'Negative': sentiment_counts.get('Negative', 0),
            'Neutral': sentiment_counts.get('Neutral', 0),
            'TopPositiveKeywords': ', '.join(keywords.get('positive_keywords', [])),
            'TopNegativeKeywords': ', '.join(keywords.get('negative_keywords', []))
        })
    
    return pd.DataFrame(results, columns=empty_cols)


def _heuristic_column_mapping(column_names: list[str]) -> dict:
    """Fallback mapping using simple keyword heuristics when LLM is unavailable."""
    lowered = {c.lower(): c for c in column_names}
    def find(*keys):
        for key in keys:
            for lc, orig in lowered.items():
                if key in lc:
                    return orig
        return None
    return {
        'ProductName': find('productname', 'product_name', 'product title', 'product', 'title', 'name'),
        'Price': find('price', 'cost', 'amount', 'mrp'),
        'Rate': find('rating', 'rate', 'stars', 'score'),
        'Review': find('review', 'comment', 'feedback', 'description', 'text'),
        'Summary': find('summary', 'headline', 'subject', 'title')
    }


@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': 'running',
        'message': 'E-commerce Sentiment Analysis API',
        'endpoints': {
            '/analyze': 'POST - Full ML pipeline analysis',
            '/outputs/<filename>': 'GET - Download generated CSV outputs',
            '/health': 'GET - Health check'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})


@app.route('/outputs/<path:filename>', methods=['GET'])
def download_output(filename: str):
    """Serve generated CSV outputs so the frontend can fetch them."""
    return send_from_directory(OUTPUTS_DIR, filename, as_attachment=False)

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Full ML pipeline:
    1. Column analysis (identify columns)
    2. Data cleaning
    3. ML predictions
    4. Keyword extraction
    5. Aspect sentiment analysis
    6. Component failure analysis
    7. Top products breakdown
    """
    try:
        # Timeout protection for entire analyze (Linux/Unix only)
        IS_WINDOWS = platform.system() == 'Windows'
        is_main_thread = threading.current_thread() is threading.main_thread()
        if not IS_WINDOWS and is_main_thread:
            import signal
            def timeout_handler(signum, frame):
                raise TimeoutError("/analyze request exceeded 120 second timeout")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(300)  # 300 second (5 min) timeout for entire request
        
        try:
            # Check if file is uploaded
            if 'file' not in request.files:
                return jsonify({'error': 'No file uploaded'}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'Empty filename'}), 400
            
            # Save uploaded file
            input_path = os.path.join(DATA_DIR, 'uploaded.csv')
            try:
                file.save(input_path)
                print(f"File saved successfully to: {input_path}")
            except Exception as e:
                print(f"ERROR saving file: {str(e)}")
                return jsonify({'error': f'Failed to save file: {str(e)}'}), 500
            
            # We'll stream-process CSV to avoid OOM. First, read a small sample for column analysis.
            try:
                sample_df = pd.read_csv(input_path, encoding='utf-8', encoding_errors='replace', nrows=5)
                print(f"Successfully read sample rows for column analysis")
            except Exception as e:
                print(f"ERROR reading CSV sample: {str(e)}")
                return jsonify({'error': f'Failed to read CSV sample: {str(e)}'}), 500
            
            # Step 1: Column Analysis
            print("Step 1: Analyzing columns...")
            try:
                column_names = sample_df.columns.tolist()
                first_rows = sample_df.values.tolist()
                print(f"Columns: {column_names[:5]}...")
                
                try:
                    # Add timeout to prevent LLM hanging indefinitely (Linux/Unix only)
                    if not IS_WINDOWS and is_main_thread:
                        import signal
                        def timeout_handler(signum, frame):
                            raise TimeoutError("LLM analysis timeout after 30s")
                        
                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(30)  # 30 second timeout (LLM column analysis only)
                    try:
                        column_mapping = analyze_columns_with_llm(column_names, first_rows)
                    finally:
                        if not IS_WINDOWS and is_main_thread:
                            signal.alarm(0)  # Cancel alarm
                except (Exception, TimeoutError) as llm_err:
                    print(f"LLM column analysis failed ({type(llm_err).__name__}): {str(llm_err)[:100]}. Falling back to heuristic mapping.")
                    column_mapping = _heuristic_column_mapping(column_names)
                print(f"Column mapping received: {list(column_mapping.keys())}")
                
                # Save column mapping
                mapping_path = os.path.join(os.path.dirname(BASE_DIR), 'column_mapping.json')
                with open(mapping_path, 'w') as f:
                    json.dump(column_mapping, f, indent=2)
                print(f"Column mapping saved to: {mapping_path}")
            except Exception as e:
                print(f"ERROR in column analysis: {str(e)}")
                return jsonify({'error': f'Column analysis failed: {str(e)}'}), 500
            
            # Step 2 + 3: Stream cleaning + predictions to prevent OOM
            print("Step 2: Cleaning data (streaming)...")
            print("Step 3: Generating predictions (streaming)...")

            cleaned_path = os.path.join(DATA_DIR, 'uploaded_cleaned.csv')
            predictions_path = os.path.join(OUTPUTS_DIR, 'predictions.csv')

            # Truncate output files before appending
            open(cleaned_path, 'w').close()
            open(predictions_path, 'w').close()

            # Load model/vectorizer once
            model_path = os.path.join(MODELS_DIR, 'logistic_model.joblib')
            vectorizer_path = os.path.join(MODELS_DIR, 'tfidf_vectorizer.joblib')
            model = None
            vectorizer = None
            if os.path.exists(model_path) and os.path.exists(vectorizer_path):
                model = joblib.load(model_path)
                vectorizer = joblib.load(vectorizer_path)
            else:
                print("WARNING: Models not found; fallback to rating-based sentiment where available")

            # Smaller chunks reduce peak memory; configurable via env
            chunk_size = int(os.getenv('CHUNK_SIZE', '2000'))
            processed_rows = 0
            sentiment_counter = Counter()
            first_clean_header = True
            first_pred_header = True

            for chunk in pd.read_csv(input_path, encoding='utf-8', encoding_errors='replace', chunksize=chunk_size):
                cleaned_chunk = clean_data_pipeline(chunk, column_mapping)
                processed_rows += len(cleaned_chunk)

                # Append cleaned chunk
                cleaned_chunk.to_csv(cleaned_path, mode='a', index=False, header=first_clean_header)
                first_clean_header = False

                # Predictions for this chunk
                if model is not None and vectorizer is not None and not cleaned_chunk.empty:
                    X = vectorizer.transform(cleaned_chunk['text'])
                    cleaned_chunk['predicted_sentiment'] = model.predict(X)
                else:
                    # Fallback to existing sentiment column
                    cleaned_chunk['predicted_sentiment'] = cleaned_chunk.get('sentiment', 'Neutral')

                # Update summary counts
                sentiment_counter.update(cleaned_chunk['predicted_sentiment'].value_counts().to_dict())

                # Append predictions chunk
                cleaned_chunk.to_csv(predictions_path, mode='a', index=False, header=first_pred_header)
                first_pred_header = False
            
            print(f"Streaming complete. Processed rows: {processed_rows}")
            
            # For downstream steps, read predictions lazily (wrappers may load fully)
            predictions_df = None
            
            # Step 4: Extract Keywords (using src/keyword_drivers.py)
            print("Step 4: Extracting keywords...")
            positive_keywords, negative_keywords = extract_keywords_wrapper()
            
            # Save keywords
            pos_path = os.path.join(OUTPUTS_DIR, 'positive_keywords.csv')
            neg_path = os.path.join(OUTPUTS_DIR, 'negative_keywords.csv')
            positive_keywords.to_csv(pos_path, index=False)
            negative_keywords.to_csv(neg_path, index=False)
            # Generate keyword charts (optional, can be disabled to save memory/CPU)
            if os.getenv('SKIP_PLOTS', '1') != '1':
                try:
                    plot_bar(positive_keywords, "Top Positive Keywords", os.path.join(OUTPUTS_DIR, 'positive_keywords.png'))
                    plot_bar(negative_keywords, "Top Negative Keywords", os.path.join(OUTPUTS_DIR, 'negative_keywords.png'))
                except Exception as chart_err:
                    print(f"Keyword chart generation failed: {chart_err}")
            
            # Step 5: Aspect Sentiment Analysis (using src/aspect_sentiment_rules.py)
            aspect_summary = pd.DataFrame()
            failure_components = pd.DataFrame()

            disable_aspects = os.getenv('DISABLE_ASPECTS', '0') == '1'
            aspect_max_rows = int(os.getenv('ASPECT_MAX_ROWS', '3000'))

            if disable_aspects:
                print("Step 5 skipped: DISABLE_ASPECTS=1")
            elif processed_rows > aspect_max_rows:
                print(f"Step 5: Sampling {aspect_max_rows} rows for aspect analysis (total {processed_rows})")
                aspect_summary = analyze_aspects_wrapper(nrows=aspect_max_rows)
                aspect_summary_path = os.path.join(OUTPUTS_DIR, 'aspect_sentiment_summary.csv')
                aspect_summary.to_csv(aspect_summary_path, index=False)
            else:
                print("Step 5: Analyzing aspect sentiment...")
                aspect_summary = analyze_aspects_wrapper(nrows=None)
                aspect_summary_path = os.path.join(OUTPUTS_DIR, 'aspect_sentiment_summary.csv')
                aspect_summary.to_csv(aspect_summary_path, index=False)

            # Step 6: Component Failure Analysis (using src/component_failure_analysis.py)
            disable_failures = os.getenv('DISABLE_FAILURES', '0') == '1'
            if disable_failures:
                print("Step 6 skipped: DISABLE_FAILURES=1")
            elif processed_rows > aspect_max_rows:
                print(f"Step 6: Sampling {aspect_max_rows} rows for failure analysis (total {processed_rows})")
                failure_components = analyze_component_failures_wrapper(nrows=aspect_max_rows)
                failure_path = os.path.join(OUTPUTS_DIR, 'failure_components_analysis.csv')
                failure_components.to_csv(failure_path, index=False)
            else:
                print("Step 6: Analyzing component failures...")
                failure_components = analyze_component_failures_wrapper(nrows=None)
                failure_path = os.path.join(OUTPUTS_DIR, 'failure_components_analysis.csv')
                failure_components.to_csv(failure_path, index=False)
            
            # Step 7: Top Products Breakdown (using src/top_products_breakdown.py)
            print("Step 7: Analyzing top products...")
            top_products = analyze_top_products_wrapper()
            
            products_path = os.path.join(OUTPUTS_DIR, 'top_products_sentiment_breakdown.csv')
            top_products.to_csv(products_path, index=False)
            # Generate stacked bar chart for top products sentiment breakdown (optional)
            if os.getenv('SKIP_PLOTS', '1') != '1':
                try:
                    if not top_products.empty:
                        fig, ax = plt.subplots(figsize=(12, 8))
                        bottom = [0] * len(top_products)
                        colors = ["#4CAF50", "#FFC107", "#F44336"]
                        labels = ["Positive", "Neutral", "Negative"]

                        for i, sentiment in enumerate(["Positive", "Neutral", "Negative"]):
                            ax.bar(
                                top_products["ProductName"],
                                top_products[sentiment],
                                bottom=bottom,
                                label=labels[i],
                                color=colors[i]
                            )
                            bottom = [b + v for b, v in zip(bottom, top_products[sentiment])]

                        ax.set_title("Top Reviewed Products — Sentiment Breakdown")
                        ax.set_ylabel("Number of Reviews")
                        ax.set_xlabel("Product Name")
                        ax.legend()
                        plt.xticks(rotation=45, ha="right")
                        plt.tight_layout()

                        chart_path = os.path.join(OUTPUTS_DIR, 'top_products_sentiment_breakdown.png')
                        plt.savefig(chart_path, dpi=150)
                        plt.close()
                except Exception as chart_err:
                    print(f"Top products chart generation failed: {chart_err}")
            
            # Return results with correct structure
            sentiment_counts = dict(sentiment_counter)
            response = jsonify({
                'status': 'success',
                'message': 'Analysis completed successfully',
                'summary': {
                    'total_reviews': processed_rows,
                    'sentiment_summary': {
                        'positive': sentiment_counts.get('Positive', 0),
                        'negative': sentiment_counts.get('Negative', 0),
                        'neutral': sentiment_counts.get('Neutral', 0)
                    }
                },
                'data': {
                    'positive_keywords': positive_keywords.to_dict('records') if not positive_keywords.empty else [],
                    'negative_keywords': negative_keywords.to_dict('records') if not negative_keywords.empty else [],
                    'aspect_sentiment': aspect_summary.to_dict('records') if not aspect_summary.empty else [],
                    'failure_components': failure_components.to_dict('records') if not failure_components.empty else [],
                    'top_products': top_products.to_dict('records') if not top_products.empty else []
                },
                'output_files': {
                    'predictions': 'predictions.csv',
                    'positive_keywords': 'positive_keywords.csv',
                    'negative_keywords': 'negative_keywords.csv',
                    'aspect_sentiment_summary': 'aspect_sentiment_summary.csv',
                    'failure_components': 'failure_components_analysis.csv',
                    'top_products': 'top_products_sentiment_breakdown.csv'
                }
            })
            # Prevent frontend caching
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
            return response
                
        finally:
            if not IS_WINDOWS and is_main_thread:
                signal.alarm(0)  # Cancel timeout alarm
            
    except TimeoutError as te:
        print(f"TIMEOUT in /analyze: {str(te)}")
        return jsonify({'error': f'Analysis timeout: {str(te)}'}), 504
    except Exception as e:
        error_msg = str(e)
        error_type = type(e).__name__
        print(f"ERROR in /analyze endpoint: {error_type}: {error_msg}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': error_msg,
            'type': error_type,
            'details': 'Check server logs for full traceback'
        }), 500

# ==========================================
# LLM FORMULATION ENDPOINT
# ==========================================
@app.route('/formulate', methods=['POST'])
def formulate():
    """
    Generate executive summary and key insights from all analysis outputs using LLM.
    Expects JSON body: {"llm_provider": "gemini" or "openrouter", "model": "optional model name"}
    """
    try:
        data = request.get_json() or {}
        llm_provider = data.get('llm_provider', 'gemini').lower()
        model_name = data.get('model')
        
        # Load all analysis results
        predictions_path = os.path.join(OUTPUTS_DIR, 'predictions.csv')
        aspects_path = os.path.join(OUTPUTS_DIR, 'aspect_sentiment_summary.csv')
        top_products_path = os.path.join(OUTPUTS_DIR, 'top_products_sentiment_breakdown.csv')
        failures_path = os.path.join(OUTPUTS_DIR, 'failure_components_analysis.csv')
        pos_keywords_path = os.path.join(OUTPUTS_DIR, 'positive_keywords.csv')
        neg_keywords_path = os.path.join(OUTPUTS_DIR, 'negative_keywords.csv')
        
        # Check if predictions exist
        if not os.path.exists(predictions_path):
            return jsonify({'error': 'No analysis data found. Run /analyze first.'}), 400
        
        # Read predictions for overall stats
        predictions = pd.read_csv(predictions_path)
        total_reviews = len(predictions)
        sentiment_counts = predictions['predicted_sentiment'].value_counts().to_dict()
        
        # Read aspect sentiment if exists
        aspects_summary = ""
        if os.path.exists(aspects_path) and os.path.getsize(aspects_path) > 0:
            aspects_df = pd.read_csv(aspects_path)
            if not aspects_df.empty:
                aspects_summary = aspects_df.to_string(index=False)
        
        # Read top products if exists
        top_products_summary = ""
        if os.path.exists(top_products_path) and os.path.getsize(top_products_path) > 0:
            top_products_df = pd.read_csv(top_products_path)
            if not top_products_df.empty:
                top_products_summary = top_products_df.head(5).to_string(index=False)
        
        # Read failures if exists
        failures_summary = ""
        if os.path.exists(failures_path) and os.path.getsize(failures_path) > 0:
            failures_df = pd.read_csv(failures_path)
            if not failures_df.empty:
                failures_summary = failures_df.head(10).to_string(index=False)

        # Read model comparison if available
        comparison_metrics = ""
        comparison_report = ""
        metrics_file = os.path.join(OUTPUTS_DIR, 'model_comparison_metrics.json')
        report_file = os.path.join(OUTPUTS_DIR, 'model_comparison_report.txt')
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, 'r') as f:
                    comparison_metrics = f.read()
            except Exception:
                comparison_metrics = ""
        if os.path.exists(report_file):
            try:
                with open(report_file, 'r') as f:
                    comparison_report = f.read()
            except Exception:
                comparison_report = ""
        
        # Read keywords
        pos_keywords_list = []
        neg_keywords_list = []
        if os.path.exists(pos_keywords_path) and os.path.getsize(pos_keywords_path) > 0:
            pos_kw_df = pd.read_csv(pos_keywords_path)
            if 'word' in pos_kw_df.columns:
                pos_keywords_list = pos_kw_df['word'].head(10).tolist()
        if os.path.exists(neg_keywords_path) and os.path.getsize(neg_keywords_path) > 0:
            neg_kw_df = pd.read_csv(neg_keywords_path)
            if 'word' in neg_kw_df.columns:
                neg_keywords_list = neg_kw_df['word'].head(10).tolist()
        
        # Build comprehensive prompt
        prompt = f"""You are an expert data analyst. Analyze the following e-commerce product review data and provide actionable insights.

**OVERALL STATISTICS:**
- Total Reviews: {total_reviews:,}
- Sentiment Distribution:
  - Positive: {sentiment_counts.get('Positive', 0):,} ({sentiment_counts.get('Positive', 0)/total_reviews*100:.1f}%)
  - Negative: {sentiment_counts.get('Negative', 0):,} ({sentiment_counts.get('Negative', 0)/total_reviews*100:.1f}%)
  - Neutral: {sentiment_counts.get('Neutral', 0):,} ({sentiment_counts.get('Neutral', 0)/total_reviews*100:.1f}%)

**TOP POSITIVE KEYWORDS:**
{', '.join(pos_keywords_list) if pos_keywords_list else 'N/A'}

**TOP NEGATIVE KEYWORDS:**
{', '.join(neg_keywords_list) if neg_keywords_list else 'N/A'}

**ASPECT SENTIMENT ANALYSIS:**
{aspects_summary if aspects_summary else 'N/A'}

**TOP PRODUCTS BY REVIEW COUNT:**
{top_products_summary if top_products_summary else 'N/A'}

**COMPONENT FAILURE ANALYSIS:**
{failures_summary if failures_summary else 'N/A'}

**MODEL COMPARISON (VADER vs TF-IDF + Logistic Regression):**
Metrics:
{comparison_metrics if comparison_metrics else 'N/A'}

Report:
{comparison_report if comparison_report else 'N/A'}

---

Provide a concise, bullet-only executive summary. Keep wording brief.

1. **EXECUTIVE SUMMARY** — 2 short bullets
2. **KEY INSIGHTS** — 5-7 bullets
3. **STRENGTHS** — 3-5 bullets
4. **AREAS FOR IMPROVEMENT** — 3-5 bullets
5. **RECOMMENDATIONS** — 3-5 bullets

Format: Markdown headers + bullets only. No long paragraphs. Be specific and data-driven."""
        
        # Call appropriate LLM
        formatted_response = ""
        
        if llm_provider == 'gemini':
            google_api_key = os.getenv('GOOGLE_API_KEY')
            if not google_api_key:
                return jsonify({'error': 'GOOGLE_API_KEY not configured'}), 400
            
            model = model_name or 'gemini-2.0-flash-exp'
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={google_api_key}"
            
            response = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.7, "maxOutputTokens": 2000}
                },
                timeout=60
            )
            
            response.raise_for_status()
            response_data = response.json()
            formatted_response = response_data['candidates'][0]['content']['parts'][0]['text']
        
        elif llm_provider == 'openrouter':
            openrouter_keys = [
                os.getenv('OPENROUTER_API_KEY'),
                os.getenv('OPENROUTER_KEY_1'),
                os.getenv('OPENROUTER_KEY_2')
            ]
            model = model_name or os.getenv('LLM_MODEL', 'openai/gpt-4o')

            last_error = None
            for openrouter_key in (k for k in openrouter_keys if k):
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {openrouter_key}",
                        "HTTP-Referer": "http://localhost",
                        "X-Title": "Project BlackFlag"
                    },
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.7,
                        "max_tokens": 2000
                    },
                    timeout=60
                )

                if response.status_code == 200:
                    formatted_response = response.json()['choices'][0]['message']['content']
                    break

                last_error = response

            if not formatted_response:
                if last_error is not None:
                    last_error.raise_for_status()
                return jsonify({'error': 'No OpenRouter API keys configured'}), 400
        
        else:
            return jsonify({'error': f'Invalid llm_provider: {llm_provider}. Use "gemini" or "openrouter"'}), 400
        
        response = jsonify({
            'status': 'success',
            'llm_provider': llm_provider,
            'total_reviews': total_reviews,
            'sentiment_summary': sentiment_counts,
            'executive_summary': formatted_response
        })
        # Prevent frontend caching
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
    
    except requests.exceptions.HTTPError as e:
        return jsonify({
            'error': f'{e.response.status_code} {e.response.reason}',
            'details': str(e),
            'type': 'HTTPError'
        }), 500
    except Exception as e:
        return jsonify({
            'error': str(e),
            'type': type(e).__name__
        }), 500

# ==========================================
# MODEL COMPARISON ENDPOINT
# ==========================================
@app.route('/compare', methods=['GET', 'POST'])
def compare_models():
    """
    Compare VADER vs TF-IDF + Logistic Regression models
    Returns accuracy metrics and LLM-formatted analysis
    """
    try:
        # Always run fresh comparison to avoid stale cached metrics
        result = run_model_comparison()
        # Prevent frontend caching with proper response headers
        response = make_response(result)
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
    
    except Exception as e:
        print(f"Error in compare_models: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'type': type(e).__name__
        }), 500

def run_model_comparison():
    """Run VADER vs Logistic Regression comparison"""
    try:
        # Load the most recent cleaned data from the current /analyze run
        data_path = os.path.join(DATA_DIR, 'uploaded_cleaned.csv')
        
        if not os.path.exists(data_path):
            return jsonify({'error': 'No cleaned data found. Run /analyze first.'}), 400
        
        # Read CSV with UTF-8 and replace invalid bytes
        df = pd.read_csv(data_path, encoding='utf-8', encoding_errors='replace')
        
        # Log dataset fingerprint to verify it's changing
        total_rows = len(df)
        sentiment_dist = df['sentiment'].value_counts().to_dict() if 'sentiment' in df.columns else {}
        print(f"Comparison data loaded: {total_rows} rows, sentiment dist: {sentiment_dist}")
        
        # Prepare data
        TEXT_COLUMN = "text"
        LABEL_COLUMN = "sentiment"
        
        df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str)
        
        # Map sentiment labels
        sentiment_map = {"positive": 1, "negative": 0, "neutral": 2}
        if LABEL_COLUMN in df.columns:
            df['sentiment_numeric'] = df[LABEL_COLUMN].map(lambda x: sentiment_map.get(str(x).lower(), -1))
        else:
            return jsonify({'error': f'{LABEL_COLUMN} column not found'}), 400
        
        X = df[TEXT_COLUMN]
        y = df['sentiment_numeric']
        
        # Safeguard: Check minimum samples
        if len(df) < 20:
            return jsonify({
                'error': 'Insufficient data for comparison',
                'details': f'Need at least 20 reviews, got {len(df)}'
            }), 400
        
        # Split data - use timestamp-based seed so each dataset gets a fair random split
        import time
        random_seed = int(time.time()) % 10000  # Use timestamp for varying splits
        
        class_counts = y.value_counts()
        stratify = None
        # Only stratify if we have at least 2 samples per class
        if (class_counts >= 2).all():
            stratify = y
        else:
            print(f"WARNING: Skipping stratification due to class imbalance: {class_counts.to_dict()}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_seed, stratify=stratify
        )
        
        print(f"Train/test split: {len(X_train)} train, {len(X_test)} test (seed={random_seed})")
        
        # Evaluate VADER
        sia = SentimentIntensityAnalyzer()
        vader_predictions = []
        
        for text in X_test:
            scores = sia.polarity_scores(text)
            compound = scores['compound']
            
            if compound >= 0.05:
                label = sentiment_map['positive']
            elif compound <= -0.05:
                label = sentiment_map['negative']
            else:
                label = sentiment_map['neutral']
            
            vader_predictions.append(label)
        
        vader_predictions = np.array(vader_predictions)
        
        vader_metrics = {
            'accuracy': float(accuracy_score(y_test, vader_predictions)),
            'precision': float(precision_score(y_test, vader_predictions, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_test, vader_predictions, average='weighted', zero_division=0)),
            'f1': float(f1_score(y_test, vader_predictions, average='weighted', zero_division=0))
        }
        
        # Evaluate Logistic Regression
        try:
            tfidf = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
            model = joblib.load(os.path.join(MODELS_DIR, "logistic_model.joblib"))
            
            X_test_tfidf = tfidf.transform(X_test)
            lr_predictions = model.predict(X_test_tfidf)
            
            # Convert string predictions to numeric if necessary
            if isinstance(lr_predictions[0], str):
                lr_predictions_numeric = np.array([sentiment_map.get(str(p).lower(), -1) for p in lr_predictions])
            else:
                lr_predictions_numeric = lr_predictions
            
            lr_metrics = {
                'accuracy': float(accuracy_score(y_test, lr_predictions_numeric)),
                'precision': float(precision_score(y_test, lr_predictions_numeric, average='weighted', zero_division=0)),
                'recall': float(recall_score(y_test, lr_predictions_numeric, average='weighted', zero_division=0)),
                'f1': float(f1_score(y_test, lr_predictions_numeric, average='weighted', zero_division=0))
            }
        except FileNotFoundError:
            return jsonify({'error': 'Model files not found'}), 400
        
        # Calculate agreement
        agreement = (vader_predictions == lr_predictions_numeric).sum() / len(y_test) * 100
        
        comparison = {
            'vader': vader_metrics,
            'logistic_regression': lr_metrics,
            'comparison': {
                'agreement_percent': float(agreement),
                'test_size': int(len(y_test))
            }
        }
        
        # Generate report
        report = generate_comparison_report(vader_metrics, lr_metrics, agreement, len(y_test))
        
        # Save results
        metrics_file = os.path.join(OUTPUTS_DIR, 'model_comparison_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        report_file = os.path.join(OUTPUTS_DIR, 'model_comparison_report.txt')
        with open(report_file, 'w') as f:
            f.write(report)
        
        import time
        return jsonify({
            'status': 'success',
            'metrics': comparison,
            'report': report,
            'cached': False,
            'timestamp': int(time.time())  # Force cache busting
        })
    
    except Exception as e:
        print(f"Error in run_model_comparison: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def generate_comparison_report(vader_metrics, lr_metrics, agreement, test_size):
    """Generate comparison report using LLM or default template"""
    # Select LLM model via env, default to GPT-5 for all clients
    llm_model = os.getenv('LLM_MODEL', 'openai/gpt-5')
    
    prompt = f"""
Based on the following model comparison results, generate a comprehensive analysis report:

VADER Sentiment Analysis Performance:
- Accuracy: {vader_metrics['accuracy']:.4f}
- Precision: {vader_metrics['precision']:.4f}
- Recall: {vader_metrics['recall']:.4f}
- F1 Score: {vader_metrics['f1']:.4f}

TF-IDF + Logistic Regression Performance:
- Accuracy: {lr_metrics['accuracy']:.4f}
- Precision: {lr_metrics['precision']:.4f}
- Recall: {lr_metrics['recall']:.4f}
- F1 Score: {lr_metrics['f1']:.4f}

Model Agreement Rate: {agreement:.2f}%
Test Set Size: {test_size} samples

Please provide:
1. Key findings and model performance comparison
2. Strengths and weaknesses of each approach
3. Recommendations for which model to use in production
4. Suggestions for improvement

Format as a clear, professional report.
"""
    
    # Try OpenRouter
    openrouter_keys = [
        os.getenv('OPENROUTER_KEY_1'),
        os.getenv('OPENROUTER_KEY_2')
    ]
    
    for key in openrouter_keys:
        if not key:
            continue
        
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {key}",
                    "HTTP-Referer": "http://localhost",
                    "X-Title": "Project BlackFlag"
                },
                json={
                    "model": llm_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 1500
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
        except Exception as e:
            print(f"OpenRouter error: {e}")
    
    # Fallback to default report
    return f"""
SENTIMENT ANALYSIS MODEL COMPARISON REPORT
{'='*60}

EXECUTIVE SUMMARY
{'-'*60}
This analysis compares two sentiment analysis approaches:
1. VADER (Valence Aware Dictionary and sEntiment Reasoner)
2. TF-IDF + Logistic Regression

PERFORMANCE METRICS
{'-'*60}

VADER Sentiment Analysis:
  Accuracy:  {vader_metrics['accuracy']:.4f} ({vader_metrics['accuracy']*100:.2f}%)
  Precision: {vader_metrics['precision']:.4f}
  Recall:    {vader_metrics['recall']:.4f}
  F1 Score:  {vader_metrics['f1']:.4f}

TF-IDF + Logistic Regression:
  Accuracy:  {lr_metrics['accuracy']:.4f} ({lr_metrics['accuracy']*100:.2f}%)
  Precision: {lr_metrics['precision']:.4f}
  Recall:    {lr_metrics['recall']:.4f}
  F1 Score:  {lr_metrics['f1']:.4f}

MODEL COMPARISON
{'-'*60}
Agreement Rate: {agreement:.2f}%
Test Set Size: {test_size} samples

WINNER: {'TF-IDF + Logistic Regression' if lr_metrics['accuracy'] > vader_metrics['accuracy'] else 'VADER' if vader_metrics['accuracy'] > lr_metrics['accuracy'] else 'TIE'}
Accuracy Difference: {abs(lr_metrics['accuracy'] - vader_metrics['accuracy']):.4f}

RECOMMENDATIONS
{'-'*60}
Use {'TF-IDF + Logistic Regression' if lr_metrics['accuracy'] > vader_metrics['accuracy'] else 'VADER'} for production deployment based on accuracy metrics.
"""

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
