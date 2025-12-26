# Anime-Recommender

This is a Python tool that generates personalized anime recommendawtions using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization and eighted cosine similarity. This tool is able to judge a user's taste and preferences by analysing the genres and scores of anime that they have watched.

it uses a Hybrid Content Based Filtering approach by combing four metrics in order to build the user's taste profile.

1. Genre Profile: Uses TF-IDF to turn genres into vectors, then calculating the cosine similarity between your profile and the list of unwatched anime.
2. Popularity: Factors in the number of members for a particular anime in Mymal_anime (MAL).
3. Quality: Factors in the MAL community users' ratings in order to raise the overall quality of recommendations.
4. Recency: Gives some weighting onto newer anime.

---

###### Prerequisties

This script was made in Python 3.13.

``` pip install pandas numpy scikit-learn```

Two CSV files are used in the same directory:
```1. mal_anime.csv```
This is taken from a Kaggle dataset by Syahrual Apriansyah that contains animes and their attributes found on MAL. The Link is found at the end of the README.MD. 
The columns needed for functionality are:
```- title```
```- Genres```
```- Type```
```- Score```
```- Members```
```- Released_Year```

```2. user_watched.csv```
This is written up by the user and requires:
- ```anime_title```: Must match the title in ```mal_anime.csv```
- ```user_rating```: PErsonal score from ```1.0``` to ```10.0```

---

##### Usage

1. Run script ```python anime_recommender.py```
2. When prompted, the program will ask you specific filters that narrows down recommendations:
   - Minimum Year
   - Genre
   - Exclusion of Genres
   - Type

List of compatible genres: 
```Action```
```Romance```
```Comedy```
```Slice of Life```
```Sci-Fi```
```Sports```
```Drama```
```Mystery```
```Suspense```
```Adventure```
```Fantasy```
```Ecchi```
```Horror```
```Supernatural```
```Avant Garde```
```Hentai```
```Boys Love```
```Girls Love```
```Award Winning```

The formats ```"OVA"``` and ```"ONA"``` are grouped under ```"Non-TV"```.
Similarly, both ```tv special``` and ```special``` are grouped under ```"Special"```.

When excluding genres, separate each by a comma. 
e.g. ```Action, Romance```

3. The program will then output a list of anime as well as its similairty score genres, format, year released and a url link to its MAL. 

---
##### Customization

The user is able to adjust the specific weights of how the tool judges each metric by changing the ```WEIGHTS``` dictionary in the code: 

```
WEIGHTS = {
    'similarity_score_genre': 0.80, # How much your genre taste matters
    'normalized_score': 0.10,      # Importance of high MAL scores
    'normalized_members': 0.05,    # Importance of popularity
    'recency_score': 0.05,         # Priority for newer anime
}
```
The user can also adjust how random the recommendations could be using ```RANDOM_PERMUATATION_STRENGTH``` variable. The value of ```0.2``` worked best in testing:
```
RANDOM_PERTURBATION_STRENGTH = 0.2
random_noise = np.random.uniform(
    low=-RANDOM_PERTURBATION_STRENGTH,
    high=RANDOM_PERTURBATION_STRENGTH,
    size=len(final_candidate_df['final_score'])
)
```

The user can change the number of recommendations by adjusting the variable ```NUM_RECOMMENDATIONS```, which is by deafult 30.

```NUM_RECOMMENDATIONS = 30```

---

##### Troubleshooting and Limitations

I could not get the JikanAPI to load properly in the script so the user will have to manually add the anime they have watched into the csv.
The anime's title in the dataset matches the first name that shows up on the MAL page. 

<img width="404" height="462" alt="image" src="https://github.com/user-attachments/assets/e6fe09ca-1301-4659-8530-f56730b194d8" />

(In this case, the anime's title would be ```Shingeki no Kyojin```)

In the ```user_watched.csv```, the format is ```anime_title,user_rating``` as seen in the sample csv in the respository. If this format is not used then that anime will be skipped. 

---

##### Credits

The dataset was taken from Kaggle by user Syahrual Apriansyah: https://www.kaggle.com/datasets/syahrulapriansyah2/mymal_anime-2025.
The file was too large to be uploaded to GitHub.
