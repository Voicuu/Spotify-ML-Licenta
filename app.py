from imports import *
from data_loading_and_cleaning import load_and_clean_data
from data_transformation import transform_data
from modeling import run_models

st.set_page_config(
    layout="wide", page_title="Licenta Voicu Andrei-Ciprian", page_icon="🎵"
)

st.title("Predicția Popularității pe Spotify")

# Sidebar
st.sidebar.image("images/logo.png", width=280)
st.sidebar.header("📚 Despre")
st.sidebar.info(
    """
Acest tool folosește învățarea automată pentru a prezice popularitatea melodiilor de pe Spotify. 
Pentru mai multe informații și codul sursă, vizitați repo-ul meu de [GitHub](https://github.com/Voicuu/Spotify-ML-Licenta).
"""
)

st.markdown(
    """
### Salut 👋

Pornește într-o călătorie muzicală pentru a descoperi potențialul ascuns al melodiilor de pe Spotify.
Algoritmii noștri avansati de învățare automată analizează esența caracteristicilor muzicale pentru a dezvălui probabilitatea ca o melodie să ajungă în topuri.

### Cum funcționează? 🤔
1. **Selectează Artiștii**: Alege dintr-o varietate de muzicieni talentați din bara laterală.
2. **Prezicerea Hitului**: Cu doar un clic, începe procesul de prezicere a popularității.
3. **Rezultate**: Explorează elementele care dau unei melodii popularitatea sa pe Spotify.
""",
    unsafe_allow_html=True,
)

model_folder = "trained_models/"


def train_and_save_models(df, model_folder):
    # Transform the data
    (
        X_train_preprocessed,
        X_test_preprocessed,
        y_train,
        y_test,
        _,
        column_transformer,
    ) = transform_data(df, [])

    # Run the models
    run_models(X_train_preprocessed, X_test_preprocessed, y_train, y_test)

    # Save the column transformer
    with open(os.path.join(model_folder, "column_transformer.pkl"), "wb") as f:
        pickle.dump(column_transformer, f)


def load_model_and_transformer(model_folder):
    best_model_name_path = os.path.join(model_folder, "best_model_name.txt")

    # Check if the best_model_name file exists, otherwise train the models
    if not os.path.isfile(best_model_name_path):
        df = load_and_clean_data()
        train_and_save_models(df, model_folder)

    if os.path.isfile(best_model_name_path):
        with open(best_model_name_path, "r") as f:
            best_model_name = f.read().strip()
        best_model_path = os.path.join(model_folder, f"{best_model_name}_model.pkl")
        column_transformer_path = os.path.join(model_folder, "column_transformer.pkl")

        if os.path.isfile(best_model_path) and os.path.isfile(column_transformer_path):
            with open(best_model_path, "rb") as f:
                best_model = pickle.load(f)
            with open(column_transformer_path, "rb") as f:
                column_transformer = pickle.load(f)
            return best_model, column_transformer
    return None, None


def parse_artists(artist_string):
    # Check if artist_string is a string
    if isinstance(artist_string, str):
        try:
            # list
            if artist_string.startswith("["):
                artist_list = ast.literal_eval(artist_string)
            else:
                # single artist name
                artist_list = [artist_string.strip('"')]

            # Clean the artist's name
            cleaned_artist_list = [artist.strip().strip("'") for artist in artist_list]

            return ", ".join(cleaned_artist_list)
        except (ValueError, SyntaxError) as e:
            # If there's an error in conversion, return the original string
            return artist_string
    else:
        # If artist_string is not a string, just return it as is
        return artist_string


def predict_and_save_results(artists_list, model, column_transformer, model_folder):
    df = load_and_clean_data()

    # Transform the data
    _, _, _, _, cases_with_mode, _ = transform_data(df, artists_list)

    if cases_with_mode.empty:
        return pd.DataFrame()

    # the 'popularity' column is not dropped, as it's needed for comparison
    prediction_features = cases_with_mode.drop(
        ["explicit", "id", "release_date"],
        axis=1,
        errors="ignore",
    )

    # Transform features by column transformer
    prediction_features_transformed = column_transformer.transform(prediction_features)

    # Make predictions
    cases_with_mode["predicted_popularity"] = model.predict(
        prediction_features_transformed
    )

    # Save the predictions
    cases_with_mode_path = os.path.join(model_folder, "cases_with_mode.csv")
    cases_with_mode.to_csv(cases_with_mode_path, index=False)

    # Return the DataFrame with actual and predicted popularity for comparison
    return cases_with_mode[
        ["artists", "name", "popularity", "popularity_level", "predicted_popularity"]
    ]


def display_prediction_table(predictions_df):
    def highlight(row):
        correct = "background-color: #8BC34A;"
        incorrect = "background-color: #E57373;"
        default = ""

        if row["popularity_level"] == row["predicted_popularity"]:
            return [
                default,
                default,
                default,
                correct,
                correct,
            ]
        else:
            return [
                default,
                default,
                default,
                incorrect,
                incorrect,
            ]

    # Display the predictions with highlighting
    st.dataframe(predictions_df.style.apply(highlight, axis=1).hide(axis="index"))


def on_form_submit():
    with st.spinner("Prezicere în curs..."):
        df = load_and_clean_data()
        best_model, column_transformer = load_model_and_transformer(model_folder)

        if user_artists and best_model and column_transformer:
            all_individual_artist_predictions = []

            for artist in user_artists:
                artist_data = df[df["artists"].apply(lambda x: artist in x)]
                if artist_data.empty:
                    st.error(
                        f"Artistul '{artist}' nu a fost găsit în setul de date. Te rog să alegi un alt artist."
                    )
                    continue

                individual_artist_predictions = predict_and_save_results(
                    [artist], best_model, column_transformer, model_folder
                )
                if (
                    individual_artist_predictions is not None
                    and not individual_artist_predictions.empty
                ):
                    all_individual_artist_predictions.append(
                        individual_artist_predictions
                    )

            if all_individual_artist_predictions:
                st.success("Prezicerea a fost realizată cu succes! 🎉")

                st.subheader("Rezultatele prezicerii:")
                st.text(
                    "Mai jos sunt afișate predicțiile pentru popularitatea melodiilor artiștilor aleși: "
                )

                updated_cases_with_mode = pd.concat(
                    all_individual_artist_predictions, ignore_index=True
                )
                display_prediction_table(updated_cases_with_mode)
            else:
                st.error(
                    "Nu s-a putut realiza prezicerea popularității pentru niciun artist. Te rog să alegi alt artist și să încerci din nou."
                )
        else:
            st.error("Vă rugăm să alegeți cel puțin un artist și să încercați din nou.")


with st.form(key="user_input_form"):
    df = load_and_clean_data()

    all_artists = list(set(chain.from_iterable(df["artists"])))
    user_artists = st.multiselect(
        "🎤 Selectează artiștii",
        options=all_artists,
        help="Selectează unul sau mai mulți artiști pentru a prezice popularitatea melodiilor lor pe Spotify.",
        placeholder="Alege artiștii",
    )
    submit_button = st.form_submit_button("🔍 Prezice popularitatea")

if submit_button:
    on_form_submit()
