import pandas as pd
pd.set_option('future.no_silent_downcasting', True) 

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import sys 
import warnings

# Suppress specific pandas warnings
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

# Utility Functions 

def clean_genre(genres):
    """Cleans and formats genres for TF-IDF vectorization."""
    return ' '.join([g.strip().lower() for g in str(genres).split(',')])

def categorize_type(anime_type):
    """
    Groups OVA and ONA types into 'non-tv'.
    Groups 'special' and 'tv special' into 'special'.
    """
    anime_type = str(anime_type).lower().strip()
    
    if anime_type in ['ova', 'ona']:
        return 'non-tv'
    if anime_type in ['special', 'tv special']:
        return 'special'
        
    return anime_type

def get_base_title(title):
    """Extracts the base title for series deduplication."""
    title = str(title) 
    lower_title = title.lower()
    
    title = title.split('(')[0]
    if 'movie' in lower_title:
        title = title[:lower_title.find('movie')]
    if ':' in title:
        title = title.split(":")[0]

    suffixes_to_remove = [
        ' season', ' part', ' cour', ' final', ' episode', ' ii', ' iii', ' iv',
        ' 1st', ' 2nd', ' 3rd', ' 4th', ' ova', ' '
    ]
    
    for suffix in suffixes_to_remove:
        if title.lower().endswith(suffix):
            title = title[:title.lower().rfind(suffix)].strip()
            
    return title.strip()

# Data Loading and Preparation 

# Load the main anime dataset
try:
    anime_df = pd.read_csv('AnimeList.csv', low_memory=False) 
    print("Main Anime dataset loaded successfully")
except FileNotFoundError:
    print("ERROR: AnimeList.csv not found.")
    sys.exit()

# USER DATA LOADING 
user_df = pd.DataFrame()

print("\n--- Loading local 'user_watched.csv' ---")
try:
    user_df = pd.read_csv('user_watched.csv')
    print(f"Local 'user_watched.csv' loaded successfully.")
except FileNotFoundError:
    print("CRITICAL ERROR: 'user_watched.csv' not found. Cannot proceed without user data.")
    sys.exit()

# Validation and Cleaning
if user_df.empty or 'anime_title' not in user_df.columns:
    print("CRITICAL ERROR: User data loaded but is empty or missing 'anime_title' column. Cannot proceed.")
    sys.exit()
    
if 'user_rating' not in user_df.columns:
    user_df['user_rating'] = 5.0
else:
    user_df['user_rating'] = pd.to_numeric(user_df['user_rating'], errors='coerce').fillna(5.0)
    user_df['user_rating'] = np.clip(user_df['user_rating'], 1.0, 10.0)

# Clean and filter the main anime data
anime_df = anime_df.dropna(subset=['Genres']).drop_duplicates(subset=['title']).copy()

# Type Grouping and Filtering 
TYPE_COLUMN = 'Type' 
if TYPE_COLUMN not in anime_df.columns:
    print(f"\nCRITICAL ERROR: Column '{TYPE_COLUMN}' not found in AnimeList.csv.")
    sys.exit()

anime_df[TYPE_COLUMN] = anime_df[TYPE_COLUMN].astype(str).str.strip().str.lower().fillna('unknown')
initial_count = len(anime_df)
anime_df = anime_df[anime_df[TYPE_COLUMN] != 'music'].copy()
print(f"Removed {initial_count - len(anime_df)} 'Music' entries from the dataset.")

anime_df['type_group'] = anime_df[TYPE_COLUMN].apply(categorize_type)

# Recency Calculation 
current_year = datetime.now().year
anime_df['Released_Year'] = pd.to_numeric(anime_df['Released_Year'], errors='coerce')
min_year_data = anime_df['Released_Year'].min() if not anime_df['Released_Year'].empty else current_year - 20
max_year_data = anime_df['Released_Year'].max() if not anime_df['Released_Year'].empty else current_year
year_span = max_year_data - min_year_data

if year_span == 0:
    anime_df['recency_score'] = 0.0
else:
    anime_df['recency_score'] = (anime_df['Released_Year'] - min_year_data) / year_span
    
anime_df['recency_score'] = anime_df['recency_score'].fillna(0.0) 
print("Recency score calculated using 'Released_Year' column.")

# Title Matching 
watched_titles = user_df['anime_title'].tolist()

watched_mask = anime_df['title'].isin(watched_titles)
watched_df = anime_df[watched_mask].copy()

if watched_df.empty:
    print("\nCRITICAL ERROR: No watched anime titles matched. Please ensure titles in 'user_watched.csv' exactly match titles in 'AnimeList.csv' (including case).")
    sys.exit()

# Merge User Ratings (1-10) into watched_df
watched_df = pd.merge(
    watched_df,
    user_df[['anime_title', 'user_rating']],
    left_on='title', 
    right_on='anime_title', 
    how='left'
).drop_duplicates(subset=['title']).drop(columns=['anime_title']) 

print(f"Total anime watched by user (found in main list): {len(watched_df)}")

# User Weight Normalization
user_rating_median = watched_df['user_rating'].median()
print(f"User's overall median rating (bias baseline): {user_rating_median:.2f}")

MIN_RATING = 1.0
MAX_RATING = 10.0

# Calculate Normalized Rating (Deviation from Median)
watched_df['normalized_rating'] = watched_df['user_rating'] - user_rating_median

# Scale the normalized rating to a weight factor [0.1 to 1.0]
watched_df['user_weight'] = (watched_df['normalized_rating'] / (MAX_RATING - MIN_RATING)) + 0.5
watched_df['user_weight'] = np.clip(watched_df['user_weight'], 0.1, 1.0)

# Add small bonus for very highly rated items (top 10% of ratings)
high_rating_threshold = watched_df['user_rating'].quantile(0.9)
watched_df['user_weight'] += (watched_df['user_rating'] > high_rating_threshold) * 0.1
watched_df['user_weight'] = np.clip(watched_df['user_weight'], 0.1, 1.0) 

print("Individual anime ratings normalized by user's median to create a fairer 'user_weight'.")

# Genre Preprocessing 
anime_df['Genres'] = anime_df['Genres'].fillna('')
anime_df['clean_genres'] = anime_df['Genres'].apply(clean_genre)

# Vectorization and Weighted Profile Creation

# Genre Vectorization
tfidf_genre = TfidfVectorizer(stop_words=None) 
tfidf_matrix_genre = tfidf_genre.fit_transform(anime_df['clean_genres']) 

# User Profile Creation: Weighted Average of watched anime vectors
watched_indices = anime_df[anime_df['title'].isin(watched_df['title'])]['title'].index
watched_vectors_genre = tfidf_matrix_genre[watched_indices].toarray()

# Get the new median-normalized weights
weights_array = watched_df['user_weight'].values.reshape(-1, 1)

# Calculate the weighted average vector for the user profile
weighted_sum_genre = np.sum(watched_vectors_genre * weights_array, axis=0)
sum_of_weights = np.sum(weights_array)
user_profile_genre = (weighted_sum_genre / sum_of_weights).reshape(1, -1)

# Scoring: Cosine Similarity between user profile and all anime
unwatched_mask = ~watched_mask
unwatched_vectors_genre = tfidf_matrix_genre[unwatched_mask.values].toarray() 
similarity_scores_genre = cosine_similarity(user_profile_genre, unwatched_vectors_genre).flatten()

# Hybrid Scoring and Final Ranking 

# Feature normalization on full dataset
anime_df['Score'] = pd.to_numeric(anime_df['Score'], errors='coerce')
median_score = anime_df['Score'].median() 
anime_df['Score'] = anime_df['Score'].fillna(median_score)

anime_df['Members'] = pd.to_numeric(anime_df['Members'], errors='coerce')
median_members = anime_df['Members'].median() 
anime_df['Members'] = anime_df['Members'].fillna(median_members)

anime_df['normalized_score'] = anime_df['Score'] / anime_df['Score'].max()
anime_df['normalized_members'] = anime_df['Members'] / anime_df['Members'].max()

# Create the final candidate DataFrame by masking the feature-rich anime_df
final_candidate_df = anime_df[unwatched_mask].copy().reset_index(drop=True)

# Add the genre similarity scores
final_candidate_df.loc[:, 'similarity_score_genre'] = similarity_scores_genre 

# Define Weights
WEIGHTS = {
    'similarity_score_genre': 0.80,      
    'normalized_score': 0.10,          
    'normalized_members': 0.05,          
    'recency_score': 0.05,             
}

# Calculate the FINAL weighted score
final_candidate_df.loc[:, 'final_score'] = (
    final_candidate_df['similarity_score_genre'] * WEIGHTS['similarity_score_genre'] +
    final_candidate_df['normalized_score'] * WEIGHTS['normalized_score'] +
    final_candidate_df['normalized_members'] * WEIGHTS['normalized_members'] +
    final_candidate_df['recency_score'] * WEIGHTS['recency_score'] 
)

# User Input, Perturbation, and Filtering 

year_input = input(f"\nWhat is the earliest release year to consider (e.g., 2010 or type 'ANY' to skip)? ").strip()
min_release_year = None
try:
    if year_input.upper() != 'ANY':
        min_release_year = int(year_input)
        if min_release_year > current_year:
             print("WARNING: Year too high. Resetting minimum year filter.")
             min_release_year = None
except ValueError:
    print("WARNING: Invalid year input. Skipping minimum year filter.")

target_genre = input("Which genre are you in the mood for? (Type 'ANY' to skip) ").strip().lower()

exclude_genres_input = input("Are there any genres you wish to EXCLUDE? (Type 'ANY' to skip)").strip().lower()

target_type = input("Are you looking for 'TV', 'Movie', 'Special', 'Non-TV' (for OVA/ONA), or type 'ANY' to skip? ").strip().lower()

# Apply Random Perturbation and Clip
RANDOM_PERTURBATION_STRENGTH = 0.2 
random_noise = np.random.uniform(
    low=-RANDOM_PERTURBATION_STRENGTH, 
    high=RANDOM_PERTURBATION_STRENGTH, 
    size=len(final_candidate_df['final_score'])
)
final_candidate_df.loc[:, 'final_score'] += random_noise
print(f"--- Applying Random Perturbation (+/- {RANDOM_PERTURBATION_STRENGTH}) for variety. ---")

final_candidate_df.loc[:, 'final_score'] = np.clip(final_candidate_df['final_score'], a_min=0.0, a_max=1.0)

final_candidates = final_candidate_df 

# MIN YEAR FILTERING  
if min_release_year is not None:
    filtered_df = final_candidates[
        (final_candidates['Released_Year'] >= min_release_year)
    ].copy()

    if not filtered_df.empty:
        final_candidates = filtered_df
        print(f"--- Candidates filtered strictly to start from year {min_release_year} onwards. ---")
    else:
        print(f"WARNING: Strict minimum year filter resulted in zero candidates. Reverting to previous candidates.")


if target_type != 'any':
    if target_type not in ['tv', 'movie', 'special', 'non-tv']:
          print(f"\nWARNING: Invalid type '{target_type}'. Skipping filter.")
    else:
        filtered_df = final_candidates[
            final_candidates['type_group'] == target_type
        ].copy()
        
        if not filtered_df.empty:
            final_candidates = filtered_df
            print(f"--- Candidates filtered down to '{target_type.upper()}' titles. ---")
        else:
            print(f"WARNING: No '{target_type.upper()}' candidates found among top matches. Returning top general matches.")
            
# GENRE EXCLUSION FILTERING 
if exclude_genres_input != 'any':
    genres_to_exclude = [g.strip() for g in exclude_genres_input.split(',')]
    
    exclusion_mask = pd.Series([True] * len(final_candidates), index=final_candidates.index)
    
    for genre in genres_to_exclude:
        contains_genre = final_candidates['Genres'].str.lower().str.contains(genre, na=False)
        exclusion_mask = exclusion_mask & (~contains_genre) 
        
    if exclusion_mask.any():
        final_candidates = final_candidates[exclusion_mask].copy()
        print(f"--- Candidates filtered to EXCLUDE: {', '.join(genres_to_exclude).capitalize()} ---")
    else:
        print(f"WARNING: Exclusion filter resulted in zero candidates. Skipping exclusion.")

# Rank the candidates based on the filtered list
ranked_candidates = final_candidates.sort_values(by='final_score', ascending=False).reset_index(drop=True)

# Series Deduplication and Final Output 

ranked_candidates.loc[:, 'base_title'] = ranked_candidates['title'].apply(get_base_title)

final_ranked_candidates = ranked_candidates.drop_duplicates(
    subset=['base_title'],
    keep='first'
).reset_index(drop=True)

ranked_candidates = final_ranked_candidates

print(f"\nSeries Deduplication complete. Candidates reduced to {len(ranked_candidates)} unique franchises.")

final_recommendation = ranked_candidates 

if target_genre != 'any':
    filtered_df = ranked_candidates[
        ranked_candidates['Genres'].str.lower().str.contains(target_genre, na=False)
    ]
    if not filtered_df.empty:
        final_recommendation = filtered_df
        print(f"--- Recommendations narrowed to INCLUDE: '{target_genre.capitalize()}' ---")
    else:
        print(f"WARNING: No candidates found for '{target_genre.capitalize()}' among top matches. Returning top general matches.")

NUM_RECOMMENDATIONS = 30

# Output Recommendations 

top_results = final_recommendation.head(NUM_RECOMMENDATIONS)

print("\n {'=' * 50}\n")
print(f"Your Top {NUM_RECOMMENDATIONS} Anime Recommendations")
print("\n {'=' * 50}")


if top_results.empty:
    print("Sorry, no matching recommendations could be found.")
else:
    rank_counter = 1 
    for _, row in top_results.iterrows():
        display_genres = row['Genres'].replace(',', ' | ')  
        
        print(f"\n{rank_counter}. {row['title']}")
        print(f"   Score: {row['final_score']:.4f}")
        print(f"   Genres: {display_genres}")
        print(f"   Format: {row['type_group'].upper()}") 
        print(f"   Year: {int(row['Released_Year']) if not pd.isna(row['Released_Year']) else 'N/A'}")
        
        source_url = row['source_url'] if not pd.isna(row['source_url']) else 'URL N/A'
        print(f"   MAL Link: {source_url}")
                     
        rank_counter += 1


print("\n {'=' * 50}")

