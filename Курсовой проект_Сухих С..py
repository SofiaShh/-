# 1. Проанализировать собственный ДФ

import re
import nltk
import pandas as pd

from sklearn.model_selection import train_test_split # используется для разбиения данных (которые содержатся в переменных x,y) на обучающую и тестовую выборки.
from sklearn.feature_extraction.text import TfidfVectorizer # используется для преобразования текстовых данных в численные.
from sklearn.naive_bayes import MultinomialNB # используется для обучения модели на основе обучающей выборки и предсказания значений для тестовой выборки.
from sklearn.metrics import accuracy_score #  используется для оценки качества обученной модели.
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as sklearn_stopwords

from nltk import sent_tokenize, word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')

nltk_stopwords = stopwords.words("english")

# Прочитать ДФ
df = pd.read_csv("netflix_titles.csv")  # читаем файл
print(df.to_string()[:3])  # приводим в удобочитаемый вид

# 1. Какие жанры наиболее часто встречаются в кол-ве 2 и более?

genre_df = (
    df["listed_in"].str.split(", ", expand=True).stack().reset_index(level=1, drop=True)
)  # Разделяем столбец "listed_in" на отдельные строки для каждого жанра
genre_df.name = "genre"  # Присваиваем имя "genre" новому ДФ
# print(genre_df[:5])

genre_counts = (genre_df.value_counts())  # Подсчитываем количество раз, которое каждый жанр встречается в исходном ДФ

common_genres = genre_counts[genre_counts >= 2].index.tolist()  # Находим жанры, которые встречаются хотя бы два раза

# Выводим результаты
print("Genres that appear at least twice:")  # заголовок
for genre in common_genres:
    print(f"{genre}: {genre_counts[genre]} times")  # Выводим жанр и количество его вхождений

# 2. Найти все фильмы в жанре ужасы за 2021 год (получить срез ДФ)

horror_movies = df[df["listed_in"].str.contains("Horror Movies")]  # Находим все фильмы в жанре "Horror"
movies_2021 = df[df["release_year"] == 2021]  # Находим все фильмы, выпущенные в 2022 году
horror_movies_2021 = horror_movies[horror_movies["release_year"] == 2021]  # Находим все фильмы в жанре "Horror", выпущенные в 2022 году

# Вывод результатов
print("Horror movies released in 2021:")  # заголовок
print(horror_movies_2021)  # Выводим ДФ

# 3. Для каждой страны найти самый популярный жанр и возрастной рейтинг

df["genre"] = df["listed_in"].str.split(", ")  # делим столбец "listed_in" в серию с жанрами
df["country"] = df["country"].str.split(", ")  # делим столбец "country" в серию со странами

# Преобразование ДФ
# explode - разделяет строки в столбце "genre" на отдельные строки для каждого жанра
# т.е. было:
# Title     Genre    Country
# Movie 1   [Genre 1, Genre 2]   [Country 1]
# Movie 2   [Genre 1, Genre 2]   [Country 1, Country 2]
# Movie 3   [Genre 3]   [Country 3]
# стало
# Title     Genre    Country
# Movie 1   Genre 1   Country 1
# Movie 1   Genre 2   Country 1
# Movie 2   Genre 1   Country 1
# Movie 2   Genre 2   Country 1
# Movie 2   Genre 1   Country 2
# Movie 2   Genre 2   Country 2
# Movie 3   Genre 3   Country 3
genre_country_df = df.explode("genre")
genre_country_df = genre_country_df.explode("country")

genre_popularity = (genre_country_df.groupby(["country", "genre", "rating"]).size().reset_index(name="count"))  # Подсчитываем кол-во раз, которое каждый жанр встречается в каждой стране

# Находим самый популярный жанр в каждой стране
# loc - вернёт строки по индексу(индексам)
# idxmax - вернёт индекс с максимальным значением
# Поэтому мы в loc передаем список индексов строк сгруппированных по стране и максимальному значению count;
# группировка по стране, т.к. она тут первична
most_popular_genre_by_country = genre_popularity.loc[genre_popularity.groupby("country")["count"].idxmax()]

most_popular_genre_by_country = (
    most_popular_genre_by_country[  # Оставляем те строки, где country не пусто
        most_popular_genre_by_country["country"] != ""
    ]
)

most_popular_genre_by_country.drop(columns=["count"], inplace=True)  # Удаляем count


# Переименовываем столбцы (для удобства)
most_popular_genre_by_country.rename(
    columns={"country": "Country", "genre": "Popular Genre", "rating": "Rating"},
    inplace=True,
)

# Выводим результаты
print("Popular Genre and Rating by Country:")  # заголовок
print(most_popular_genre_by_country)  # Выводим ДФ

# 4. В каком месяце в году за весь период выходит больше всего фильмов?
# В каком месяце выходит больше всего продуктов с детским рейтингом?

# Разделим столбец "date_added" на месяц и год
df["month"] = df["date_added"].str.extract(r"(\w+)")
df["year"] = df["date_added"].str.extract(r"(\d{4})")

# Подсчитываем кол-во фильмов по месяцам и годам
month_year_df = df.groupby(["month", "year"]).size().reset_index(name="count")

# Находим месяц с максимальным кол-вом фильмов
max_films_month = month_year_df.groupby("month")["count"].sum().idxmax()
max_films_month_count = month_year_df.groupby("month")["count"].sum().max()
print(f"Month with the most releases: {max_films_month} with {max_films_month_count} films")

# Как и раньше, только добавляем разбивку по рейтингу
month_year_rating_df = (df.groupby(["month", "year", "rating"]).size().reset_index(name="count"))
RATING_TARGET = "G"

# Оставляем фильмы, где rating == RATING_TARGET
month_year_rating_df = month_year_rating_df[month_year_rating_df["rating"] == RATING_TARGET]

# Находим месяц с максимальным кол-вом фильмов
max_films_month_rating = month_year_rating_df.groupby(["month"])["count"].max().idxmax()
max_films_month_rating_count = (month_year_rating_df.groupby(["month"])["count"].max().max())
print(f"Month with the most releases for {RATING_TARGET} rated films: {max_films_month_rating} with {max_films_month_rating_count} films")

# КУРСОВОЙ ПРОЕКТ

# Будем по описанию
X = df["description"]
# определять жанры
y = df["genre"]

# Стеммер и анализатор
stemmer = SnowballStemmer("english")
analyzer = TfidfVectorizer().build_analyzer()


# Функция для токенизации и стемминга
def tokenize_and_stem(text):
    # токенизация
    tokens = [word for sent in sent_tokenize(text) for word in word_tokenize(sent)]
    # фильтруем только буквенные слова
    filtered_tokens = []
    for token in tokens:
        if re.search("[a-zA-Z]", token):
            filtered_tokens.append(token)

    # стемминг
    stems = (
        stemmer.stem(t)
        for t in filtered_tokens
        if t not in nltk_stopwords and t not in sklearn_stopwords
    )

    return stems


# Создаем векторизатор
vectorizer = TfidfVectorizer(analyzer=tokenize_and_stem)
# Применяем векторизатор
X = vectorizer.fit_transform(X)

# Все жанры по порядку
all_genres = sorted(list(set([genre for genres in y for genre in genres])))


# Функция перевода жанров в бинарный вид
# Каждое описание будет соответствовать числу, у которого биты соответствуют жанрам
# 1 - такой жанр есть, 0 - нет
# Получается, каждое описание будет соответствовать бинарному числу типа 1001101000
def genres_to_binary(genres):
    bin_val = 0

    for idx, genre in enumerate(all_genres):
        if genre in genres:
            bin_val |= 1 << idx

    return bin_val


# Функция перевода бинарного значения в жанры
def binary_to_genres(bin_val):
    genres = []

    for idx, genre in enumerate(all_genres):
        if bin_val & (1 << idx):
            genres.append(genre)

    return genres


# Кодируем жанры в бинарный вид
y = y.map(lambda x: genres_to_binary(x))

# Разбиваем на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Обучаем модель
mnb = MultinomialNB()
mnb.fit(X_train, y_train)

# Предсказываем
pred = mnb.predict(X_test)

# Оцениваем
score = accuracy_score(y_test, pred)
print(f"Score: {score:.2f}") # :.2f - округлить плавающее значение до 2х точек после запятой

# Случайная выборка
random_from_df = df.sample(n=10)
pred = mnb.predict(vectorizer.transform(random_from_df["description"]))

# Переводим бинарные значения в жанры
random_from_df["predicted-genre"] = list(map(binary_to_genres, pred))
# Вывод
print(random_from_df[["genre", "predicted-genre", "description"]])
