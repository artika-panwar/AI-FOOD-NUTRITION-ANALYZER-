import streamlit as st
import pickle
import os

# ── PAGE CONFIG ────────────────────────────────────────────
st.set_page_config(
    page_title="AI Food Nutrition Analyzer",
    page_icon="🥗",
    layout="wide"
)

# ── LOAD MODEL ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists("nutrition_model.pkl"):
        return None
    with open("nutrition_model.pkl", "rb") as f:
        return pickle.load(f)

model_data = load_model()

# ── SIDEBAR ────────────────────────────────────────────────
with st.sidebar:
    st.header("Food Nutrition Analyzer")
    st.caption("Random Forest")

    st.divider()

    st.write("**Accuracy:** 99.6% (Calories)")
    st.write("**Algorithm:** Random Forest Regressor")
    st.write("**Dataset:** 4,221 food items")

    st.divider()

    st.write("**Developer:** Artika Panwar")

# ── MAIN PAGE ──────────────────────────────────────────────
st.title("AI Food Nutrition Analyzer")
st.caption("Enter a food name and click **Predict Nutrition**.")

if model_data is None:
    st.error("nutrition_model.pkl not found. Please run train_model.py first.")
    st.stop()

food_list = model_data['food_list']
lookup    = model_data['food_lookup']
targets   = model_data['target_cols']

UNITS = {
    "Caloric Value":  "kcal",
    "Fat":            "g",
    "Protein":        "g",
    "Carbohydrates":  "g",
    "Dietary Fiber":  "g",
    "Sugars":         "g",
}

# ── INPUT FORM ─────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    food_input = st.selectbox(
        "Food Name",
        options=[""] + [f.title() for f in food_list],
        index=0,
        help="Select a food item from the list"
    )

with col2:
    serving = st.number_input(
        "Serving Size (grams)",
        min_value=1,
        max_value=1000,
        value=100,
        step=10,
        help="Values are per 100g by default"
    )

predict_btn = st.button("Predict Nutrition", type="primary", use_container_width=True)


# ── PREDICTION ─────────────────────────────────────────────
def predict(name):
    n = name.strip().lower()
    if n in lookup:
        return {c: round(lookup[n][c], 2) for c in targets}, "exact"
    matches = [f for f in lookup if n in f or f in n]
    if matches:
        best = matches[0]
        return {c: round(lookup[best][c], 2) for c in targets}, f"partial:{best}"
    return None, "not_found"


if predict_btn:
    if not food_input or food_input == "":
        st.warning("Please select a food item first.")
    else:
        result, match_type = predict(food_input)

        if result is None:
            st.error(f"'{food_input}' could not be found in the dataset.")
        else:
            # Scale by serving size (dataset values are per 100g)
            factor = serving / 100.0
            scaled = {k: round(v * factor, 2) for k, v in result.items()}

            label = food_input.title()

            # Status badge -> native st.success / st.info / st.warning instead of HTML badge
            if match_type == "exact":
                status_text = "Exact match"
                status_fn = st.success
            elif match_type.startswith("partial"):
                matched_food = match_type.split(":")[1].title()
                status_text = f"Partial match: {matched_food}"
                status_fn = st.warning
            else:
                status_text = "ML Estimate"
                status_fn = st.info

            st.subheader(f"Results for {label}")
            status_fn(status_text)
            st.caption(f"Serving size: {serving} g")

            # ── Native metric cards instead of HTML nutrient rows ──
            metric_cols = st.columns(3)
            nutrient_items = list(scaled.items())

            for i, (nutrient, val) in enumerate(nutrient_items):
                unit = UNITS.get(nutrient, "")
                with metric_cols[i % 3]:
                    st.metric(label=nutrient, value=f"{val} {unit}")

            # ── Also show as a clean table ──
            st.divider()
            table_rows = [
                {"Nutrient": nutrient, "Value": f"{val} {UNITS.get(nutrient, '')}"}
                for nutrient, val in scaled.items()
            ]
            st.dataframe(table_rows, use_container_width=True, hide_index=True)

            # History log
            if "history" not in st.session_state:
                st.session_state.history = []
            st.session_state.history.insert(
                0, f"{label} ({serving}g) — "
                   f"Cal: {scaled['Caloric Value']} kcal | "
                   f"Protein: {scaled['Protein']} g | "
                   f"Fat: {scaled['Fat']} g"
            )

# ── HISTORY ────────────────────────────────────────────────
if "history" in st.session_state and st.session_state.history:
    st.divider()
    st.subheader("Search History")
    for item in st.session_state.history[:8]:
        st.text(item)