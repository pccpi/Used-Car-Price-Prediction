from __future__ import annotations

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import joblib


# =========================
# Настройки проекта
# =========================
ICE_MAE_EUR = 2862
EV_MAE_EUR = 4910

LANGS = {"Русский": "ru", "Română": "ro", "English": "en"}

T = {
    # Заголовок в 2 строки
    "title_l1": {"ru": "Калькулятор стоимости авто", "ro": "Calculator preț auto", "en": "Used Car Price"},
    "title_l2": {"ru": "с пробегом", "ro": "rulat", "en": "Calculator"},
    "subtitle": {
        "ru": "Оценка по данным рынка Германии. Это приблизительная оценка, а не гарантированная цена.",
        "ro": "Estimare pe baza pieței din Germania. Este o estimare, nu un preț garantat.",
        "en": "Estimate based on the German market. This is an estimate, not a guaranteed price.",
    },
    "lang": {"ru": "Язык", "ro": "Limba", "en": "Language"},
    "is_ev": {"ru": "Электромобиль (EV)", "ro": "Mașină electrică (EV)", "en": "Electric vehicle (EV)"},
    "brand": {"ru": "Марка", "ro": "Marcă", "en": "Brand"},
    "model": {"ru": "Модель", "ro": "Model", "en": "Model"},
    "color": {"ru": "Цвет", "ro": "Culoare", "en": "Color"},
    "year": {"ru": "Год выпуска", "ro": "An fabricație", "en": "Year"},
    "mileage": {"ru": "Пробег, км", "ro": "Kilometraj, km", "en": "Mileage, km"},
    "power": {"ru": "Мощность", "ro": "Putere", "en": "Power"},
    "power_unit": {"ru": "Единицы мощности", "ro": "Unități putere", "en": "Power unit"},
    "transmission": {"ru": "Коробка передач", "ro": "Transmisie", "en": "Transmission"},
    "fuel_type": {"ru": "Тип топлива", "ro": "Tip combustibil", "en": "Fuel type"},
    "consumption": {"ru": "Расход топлива, л/100км", "ro": "Consum, l/100km", "en": "Consumption, l/100km"},
    "calc": {"ru": "Рассчитать цену", "ro": "Calculează prețul", "en": "Calculate price"},
    "features_sent": {"ru": "Показать признаки, отправленные в модель", "ro": "Afișează caracteristicile trimise în model", "en": "Show features sent to the model"},
    "price_est": {"ru": "Оценка цены", "ro": "Estimare preț", "en": "Estimated price"},
    "range": {"ru": "Ориентировочный коридор (±MAE)", "ro": "Interval orientativ (±MAE)", "en": "Approx. range (±MAE)"},
    "range_note": {
        "ru": "Это не доверительный интервал, а подсказка по средней ошибке модели.",
        "ro": "Nu este un interval de încredere, ci o indicație bazată pe eroarea medie.",
        "en": "Not a confidence interval; it’s a hint based on the model’s average error.",
    },
    "age_clip_warn": {
        "ru": "Модель обучена на авто не старше **{max_age} лет** (примерно не раньше **{min_year} года**). "
              "Для введённого года точность может быть ниже — возраст будет ограничен.",
        "ro": "Modelul a fost antrenat pe mașini de maximum **{max_age} ani** (aprox. nu mai devreme de **{min_year}**). "
              "Pentru anul introdus, precizia poate fi mai mică — vârsta va fi limitată.",
        "en": "The model was trained on cars up to **{max_age} years old** (roughly not earlier than **{min_year}**). "
              "For your input, accuracy may be lower — age will be clipped.",
    },
    "age_penalty_note": {
        "ru": "Дополнительно применён мягкий штраф за «лишние» годы: −3% за каждый год сверх обучающего диапазона.",
        "ro": "S-a aplicat un „penalty” ușor pentru anii în plus: −3% pentru fiecare an peste intervalul de antrenare.",
        "en": "A mild penalty was applied for extra years: −3% for each year beyond the training range.",
    },
    "future_year_warn": {
        "ru": "Год выпуска в будущем. Возраст будет принят как 0.",
        "ro": "Anul este în viitor. Vârsta va fi considerată 0.",
        "en": "Year is in the future. Age will be set to 0.",
    },
    "need_files": {
        "ru": "Положи рядом с app.py файлы **ice_bundle.joblib**, **ev_bundle.joblib** "
              "и (желательно) **cars_ice.csv**, **cars_ev.csv** для зависимых списков «марка → модель».",
        "ro": "Pune lângă app.py fișierele **ice_bundle.joblib**, **ev_bundle.joblib** "
              "și (opțional) **cars_ice.csv**, **cars_ev.csv** pentru liste dependente „marcă → model”.",
        "en": "Place next to app.py: **ice_bundle.joblib**, **ev_bundle.joblib** "
              "and (optionally) **cars_ice.csv**, **cars_ev.csv** for dependent lists “brand → model”.",
    },
    "mode_ice": {"ru": "ДВС", "ro": "ICE", "en": "ICE"},
    "mode_ev": {"ru": "Электро (EV)", "ro": "Electric (EV)", "en": "EV"},
    "years_suffix": {"ru": "лет", "ro": "ani", "en": "y"},
    "max_age_label": {"ru": "Макс. возраст (обучение)", "ro": "Vârsta max. (antrenare)", "en": "Max age (train)"},
    "approx_from_year": {"ru": "≈ не раньше {y}", "ro": "≈ nu mai devreme de {y}", "en": "≈ ≥ {y}"},
}


# =========================
# Локализация значений (для UI)
# =========================
TRANSLATE_VALUE = {
    "fuel_type": {
        "Petrol": {"ru": "Бензин", "ro": "Benzină", "en": "Petrol"},
        "Diesel": {"ru": "Дизель", "ro": "Motorină", "en": "Diesel"},
        "Electric": {"ru": "Электро", "ro": "Electric", "en": "Electric"},
        "Hybrid": {"ru": "Гибрид", "ro": "Hibrid", "en": "Hybrid"},
        "LPG": {"ru": "Газ (LPG)", "ro": "GPL", "en": "LPG"},
        "CNG": {"ru": "Метан (CNG)", "ro": "GNC", "en": "CNG"},

        # Ethanol / E85
        "Ethanol": {"ru": "Этанол (E85)", "ro": "Etanol (E85)", "en": "Ethanol (E85)"},
        "E85": {"ru": "Этанол (E85)", "ro": "Etanol (E85)", "en": "Ethanol (E85)"},
        "E-85": {"ru": "Этанол (E85)", "ro": "Etanol (E85)", "en": "Ethanol (E85)"},
        "Ethanol (E85)": {"ru": "Этанол (E85)", "ro": "Etanol (E85)", "en": "Ethanol (E85)"},

        # Diesel hybrid
        "Diesel hybrid": {"ru": "Дизель-гибрид", "ro": "Hibrid diesel", "en": "Diesel hybrid"},
        "Diesel Hybrid": {"ru": "Дизель-гибрид", "ro": "Hibrid diesel", "en": "Diesel hybrid"},

        # Hydrogen
        "Hydrogen": {"ru": "Водород", "ro": "Hidrogen", "en": "Hydrogen"},
        "Hydrogen fuel cell": {"ru": "Водород (топливный элемент)", "ro": "Hidrogen (celulă)", "en": "Hydrogen fuel cell"},
    },
    "transmission_type": {
        "Manual": {"ru": "Механика", "ro": "Manuală", "en": "Manual"},
        "Automatic": {"ru": "Автомат", "ro": "Automată", "en": "Automatic"},
        "Semi-automatic": {"ru": "Полуавтомат", "ro": "Semi-automată", "en": "Semi-automatic"},
        "Unknown": {"ru": "Не указано", "ro": "Necunoscut", "en": "Unknown"},
    },
    "color": {
        "black": {"ru": "Чёрный", "ro": "Negru", "en": "Black"},
        "white": {"ru": "Белый", "ro": "Alb", "en": "White"},
        "silver": {"ru": "Серебристый", "ro": "Argintiu", "en": "Silver"},
        "grey": {"ru": "Серый", "ro": "Gri", "en": "Grey"},
        "blue": {"ru": "Синий", "ro": "Albastru", "en": "Blue"},
        "red": {"ru": "Красный", "ro": "Roșu", "en": "Red"},
        "green": {"ru": "Зелёный", "ro": "Verde", "en": "Green"},
        "orange": {"ru": "Оранжевый", "ro": "Portocaliu", "en": "Orange"},
        "brown": {"ru": "Коричневый", "ro": "Maro", "en": "Brown"},
        "yellow": {"ru": "Жёлтый", "ro": "Galben", "en": "Yellow"},
        "beige": {"ru": "Бежевый", "ro": "Bej", "en": "Beige"},
        "violet": {"ru": "Фиолетовый", "ro": "Violet", "en": "Violet"},
        "gold": {"ru": "Золотой", "ro": "Auriu", "en": "Gold"},
        "bronze": {"ru": "Бронзовый", "ro": "Bronz", "en": "Bronze"},
    },
}

POWER_UNITS_UI = {
    "ru": [("kW", "kW"), ("PS", "л.с. (PS)")],
    "ro": [("kW", "kW"), ("PS", "CP (PS)")],
    "en": [("kW", "kW"), ("PS", "hp (PS)")],
}


# =========================
# Утилиты
# =========================
def tr(key: str, lang: str) -> str:
    return T.get(key, {}).get(lang, key)


def pretty_brand(s: str) -> str:
    s = str(s).strip()
    if not s:
        return s
    if s.isupper() and len(s) <= 4:
        return s
    return s[:1].upper() + s[1:]


def pretty_title(s: str) -> str:
    s = str(s).strip()
    if not s:
        return s
    return s[:1].upper() + s[1:]


def is_unknown(v: str) -> bool:
    return str(v).strip().lower() in {"unknown", "nan", "none", ""}


def filter_unknown(items: list[str]) -> list[str]:
    return [x for x in items if not is_unknown(x)]


def translate_value(field: str, value: str, lang: str) -> str:
    v = str(value).strip()
    if not v:
        return v

    if field == "color":
        key = v.lower()
        return TRANSLATE_VALUE.get(field, {}).get(key, {}).get(lang, pretty_title(v))

    if field == "fuel_type":
        # нормализация: сравниваем "мягко", чтобы не плодить варианты
        v_norm = v.strip().lower().replace("_", " ").replace("-", " ")
        for key, langs in TRANSLATE_VALUE["fuel_type"].items():
            k_norm = str(key).lower().replace("_", " ").replace("-", " ")
            if v_norm == k_norm:
                return langs.get(lang, pretty_title(v))
        return pretty_title(v)

    if field == "transmission_type":
        if is_unknown(v):
            return TRANSLATE_VALUE["transmission_type"]["Unknown"].get(lang, "Unknown")
        return TRANSLATE_VALUE.get(field, {}).get(v, {}).get(lang, pretty_title(v))

    return pretty_title(v)


def sort_by_ui(items: list[str], field: str, lang: str) -> list[str]:
    return sorted(items, key=lambda x: translate_value(field, x, lang).lower())


def to_kw(value: float, unit: str) -> float:
    if unit == "kW":
        return float(value)
    return float(value) / 1.35962  # PS -> kW


# =========================
# Файлы и модели (Codespaces-friendly)
# =========================
APP_DIR = Path(__file__).resolve().parent


def _find_file(filename: str) -> Path:
    candidates = [
        APP_DIR / filename,
        APP_DIR / "models" / filename,
        APP_DIR / "artifacts" / filename,
        APP_DIR / "model" / filename,
    ]
    for p in candidates:
        if p.exists():
            return p

    # fallback: "ice_bundle (1).joblib"
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    for folder in [APP_DIR, APP_DIR / "models", APP_DIR / "artifacts", APP_DIR / "model"]:
        if folder.exists():
            matches = sorted(folder.glob(f"{stem}*{suffix}"), key=lambda x: x.stat().st_mtime, reverse=True)
            if matches:
                return matches[0]

    raise FileNotFoundError(f"Не найден файл {filename}. Ищу в {APP_DIR} и подпапках models/artifacts/model.")


@st.cache_resource
def load_bundles():
    ice_bundle = joblib.load(_find_file("ice_bundle.joblib"))
    ev_bundle = joblib.load(_find_file("ev_bundle.joblib"))
    return ice_bundle, ev_bundle


@st.cache_data
def load_reference_frames():
    out = {"ice": None, "ev": None}
    ice_csv = APP_DIR / "cars_ice.csv"
    ev_csv = APP_DIR / "cars_ev.csv"
    if ice_csv.exists():
        out["ice"] = pd.read_csv(ice_csv)
    if ev_csv.exists():
        out["ev"] = pd.read_csv(ev_csv)
    return out


def build_refs(df_ref: pd.DataFrame | None):
    brands, colors, transmissions, fuel_types = [], [], [], []
    models_by_brand: dict[str, list[str]] = {}

    if df_ref is None:
        return brands, models_by_brand, colors, transmissions, fuel_types

    df = df_ref.copy()
    for col in ["brand", "model", "color", "transmission_type", "fuel_type"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    if "brand" in df.columns:
        brands = sorted(df["brand"].dropna().unique().tolist(), key=lambda x: x.lower())

    if "brand" in df.columns and "model" in df.columns:
        tmp = df[["brand", "model"]].dropna()
        for b, grp in tmp.groupby("brand"):
            models_by_brand[str(b)] = sorted(grp["model"].unique().tolist(), key=lambda x: x.lower())

    if "color" in df.columns:
        colors = sorted(df["color"].dropna().unique().tolist(), key=lambda x: str(x).lower())
    if "transmission_type" in df.columns:
        transmissions = sorted(df["transmission_type"].dropna().unique().tolist(), key=lambda x: str(x).lower())
    if "fuel_type" in df.columns:
        fuel_types = sorted(df["fuel_type"].dropna().unique().tolist(), key=lambda x: str(x).lower())

    return brands, models_by_brand, colors, transmissions, fuel_types


def predict_with_bundle(bundle: dict, features: dict) -> float:
    feature_cols = bundle["feature_cols"]
    use_log = bool(bundle.get("use_log_target", False))
    model = bundle["model"]

    X = pd.DataFrame([{k: features.get(k, np.nan) for k in feature_cols}])
    y_pred = float(model.predict(X)[0])
    return float(np.expm1(y_pred)) if use_log else y_pred


# =========================
# UI
# =========================
st.set_page_config(page_title="Used Car Price", page_icon="🚗", layout="centered")

with st.sidebar:
    lang_name = st.selectbox("Language / Limba / Язык", list(LANGS.keys()), index=0)
lang = LANGS[lang_name]

# Заголовок в 2 строки
st.markdown(
    f"<h2 style='margin-bottom:0'>{tr('title_l1', lang)}<br>{tr('title_l2', lang)}</h2>",
    unsafe_allow_html=True,
)
st.caption(tr("subtitle", lang))

try:
    ice_bundle, ev_bundle = load_bundles()
except Exception:
    st.error(tr("need_files", lang))
    st.stop()

refs = load_reference_frames()
current_year = datetime.now().year

colA, colB = st.columns([1, 1])
with colA:
    is_ev = st.toggle(tr("is_ev", lang), value=False, key="is_ev_toggle")
with colB:
    max_age = int((ev_bundle if is_ev else ice_bundle).get("max_train_age", 0) or 0)
    if max_age:
        min_year_allowed = current_year - max_age
        st.write("")
        st.write(
            f"**{tr('max_age_label', lang)}**: {max_age} {tr('years_suffix', lang)} "
            f"({tr('approx_from_year', lang).format(y=min_year_allowed)})"
        )

st.divider()

mode_key = "ev" if is_ev else "ice"
df_ref = refs.get(mode_key)
if df_ref is None:
    st.info(tr("need_files", lang))

brands, models_by_brand, colors, transmissions, fuel_types = build_refs(df_ref)

# Полировка: сортировка по алфавиту ВЫБРАННОГО языка (для UI)
if colors:
    colors = sort_by_ui(colors, "color", lang)

# Коробка: Unknown скрываем в UI, но сохраняем как fallback
transmissions = filter_unknown(transmissions)
if transmissions:
    transmissions = sort_by_ui(transmissions, "transmission_type", lang)

if fuel_types:
    fuel_types = sort_by_ui(fuel_types, "fuel_type", lang)

# =========================
# Все поля в одном красивом блоке,
# но марка/модель реагируют сразу (вне формы)
# =========================
with st.container(border=True):
    # 1) Марка/модель
    top1, top2 = st.columns(2)

    with top1:
        if brands:
            brand = st.selectbox(tr("brand", lang), brands, format_func=pretty_brand, key=f"brand_{mode_key}")
        else:
            brand = st.text_input(tr("brand", lang), value=("tesla" if is_ev else "volkswagen"), key=f"brand_text_{mode_key}")

    with top2:
        if models_by_brand and brand in models_by_brand and models_by_brand[brand]:
            model_name = st.selectbox(
                tr("model", lang),
                models_by_brand[brand],
                format_func=pretty_title,
                key=f"model_{mode_key}_{brand}",
            )
        elif df_ref is not None and "model" in df_ref.columns:
            all_models = sorted(df_ref["model"].dropna().unique().tolist(), key=lambda x: x.lower())
            model_name = st.selectbox(tr("model", lang), all_models, format_func=pretty_title, key=f"model_all_{mode_key}")
        else:
            model_name = st.text_input(tr("model", lang), value=("Model 3" if is_ev else "Golf"), key=f"model_text_{mode_key}")

    st.write("")

    # 2) Остальные поля — в форме
    with st.form("car_form", clear_on_submit=False):
        c1, c2 = st.columns(2)

        with c1:
            if colors:
                color = st.selectbox(
                    tr("color", lang),
                    colors,
                    format_func=lambda x: translate_value("color", x, lang),
                    key=f"color_{mode_key}",
                )
            else:
                color = st.text_input(tr("color", lang), value="black", key=f"color_text_{mode_key}")

            # Коробка передач: ICE/EV
            if is_ev:
                ev_trans_opts = transmissions[:] if transmissions else ["Automatic", "Manual", "Semi-automatic"]
                ev_trans_opts = filter_unknown(ev_trans_opts)

                if len(ev_trans_opts) == 1:
                    transmission = ev_trans_opts[0]
                    st.selectbox(
                        tr("transmission", lang),
                        ev_trans_opts,
                        index=0,
                        format_func=lambda x: translate_value("transmission_type", x, lang),
                        disabled=True,
                        key=f"trans_{mode_key}",
                    )
                else:
                    idx = ev_trans_opts.index("Automatic") if "Automatic" in ev_trans_opts else 0
                    transmission = st.selectbox(
                        tr("transmission", lang),
                        ev_trans_opts,
                        index=idx,
                        format_func=lambda x: translate_value("transmission_type", x, lang),
                        key=f"trans_{mode_key}",
                    )
            else:
                ice_trans_opts = transmissions[:] if transmissions else ["Manual", "Automatic", "Semi-automatic"]
                ice_trans_opts = filter_unknown(ice_trans_opts)
                idx = ice_trans_opts.index("Manual") if "Manual" in ice_trans_opts else 0
                transmission = st.selectbox(
                    tr("transmission", lang),
                    ice_trans_opts,
                    index=idx,
                    format_func=lambda x: translate_value("transmission_type", x, lang),
                    key=f"trans_{mode_key}",
                )

        with c2:
            year = st.number_input(
                tr("year", lang),
                min_value=1950,
                max_value=current_year + 1,
                value=min(2018, current_year),
                step=1,
                key=f"year_{mode_key}",
            )
            mileage = st.number_input(
                tr("mileage", lang),
                min_value=0,
                max_value=1_000_000,
                value=95_000,
                step=1000,
                key=f"mileage_{mode_key}",
            )

            unit_pairs = POWER_UNITS_UI[lang]
            power_unit_label = st.selectbox(
                tr("power_unit", lang),
                [lbl for _, lbl in unit_pairs],
                index=0,
                key=f"punit_{mode_key}",
            )
            power_unit = next(code for code, lbl in unit_pairs if lbl == power_unit_label)

            power_val = st.number_input(
                tr("power", lang),
                min_value=1.0,
                max_value=2000.0,
                value=110.0 if is_ev else 85.0,
                step=1.0,
                key=f"pval_{mode_key}",
            )

        c3, _ = st.columns(2)
        with c3:
            # Топливо: EV фиксируем Electric (и дизейблим), ICE — выбор, но всё локализовано
            if is_ev:
                fuel_type = "Electric"
                st.selectbox(
                    tr("fuel_type", lang),
                    ["Electric"],
                    index=0,
                    format_func=lambda x: translate_value("fuel_type", x, lang),
                    disabled=True,
                    key=f"fuel_{mode_key}",
                )
            else:
                fuel_options = fuel_types if fuel_types else ["Petrol", "Diesel", "Hybrid", "LPG", "CNG", "Ethanol", "Hydrogen", "Diesel hybrid"]
                fuel_type = st.selectbox(
                    tr("fuel_type", lang),
                    fuel_options,
                    index=0,
                    format_func=lambda x: translate_value("fuel_type", x, lang),
                    key=f"fuel_{mode_key}",
                )

        fuel_consumption = None
        if not is_ev:
            fuel_consumption = st.number_input(
                tr("consumption", lang),
                min_value=0.1,
                max_value=40.0,
                value=6.8,
                step=0.1,
                key="cons_ice",
            )

        submitted = st.form_submit_button(tr("calc", lang))

# =========================
# Предсказание
# =========================
if submitted:
    bundle = ev_bundle if is_ev else ice_bundle
    mae = EV_MAE_EUR if is_ev else ICE_MAE_EUR
    label = tr("mode_ev", lang) if is_ev else tr("mode_ice", lang)

    power_kw = to_kw(power_val, power_unit)

    # возраст по текущему году
    car_age = int(current_year - int(year))
    if car_age < 0:
        st.warning(tr("future_year_warn", lang))
        car_age = 0

    # клиппинг + мягкий штраф (вариант B)
    max_train_age = bundle.get("max_train_age", None)
    age_used = car_age
    clipped = False
    extra_years = 0

    if max_train_age is not None:
        max_train_age = int(max_train_age)
        if car_age > max_train_age:
            clipped = True
            extra_years = car_age - max_train_age
            age_used = max_train_age
            min_year_allowed = current_year - max_train_age
            st.warning(tr("age_clip_warn", lang).format(max_age=max_train_age, min_year=min_year_allowed))

    # Фолбэк по коробке: UI скрывает Unknown, но модель его понимает
    if is_unknown(transmission):
        transmission = "Unknown"

    features = {
        "brand": str(brand).strip(),          # не переводим
        "model": str(model_name).strip(),     # не переводим
        "color": str(color).strip(),          # из датасета (обычно lower)
        "car_age": int(age_used),
        "mileage_in_km": float(mileage),
        "power_kw": float(power_kw),
        "transmission_type": str(transmission).strip(),
        "fuel_type": str(fuel_type).strip(),
    }
    if not is_ev:
        features["fuel_consumption_l_100km"] = float(fuel_consumption)

    try:
        price = predict_with_bundle(bundle, features)
    except Exception as e:
        st.error("Ошибка предсказания. Проверь, что bundle.joblib соответствует feature_cols и версиям пакетов.")
        st.exception(e)
        st.stop()

    # Мягкий штраф за «лишние» годы
    penalty_applied = False
    if extra_years > 0:
        price *= (0.97 ** extra_years)
        penalty_applied = True

    price_round = int(round(price))
    low = int(max(0, round(price - mae)))
    high = int(round(price + mae))

    st.success(f"{tr('price_est', lang)} ({label}): **{price_round:,} €**".replace(",", " "))
    st.caption(
        f"{tr('range', lang)}: **{low:,} – {high:,} €**".replace(",", " ")
        + f"\n\n{tr('range_note', lang)}"
    )
    if penalty_applied:
        st.caption(tr("age_penalty_note", lang))

    with st.expander(tr("features_sent", lang)):
        debug = dict(features)
        debug["car_age_raw"] = int(car_age)
        debug["car_age_used"] = int(age_used)
        debug["age_was_clipped"] = bool(clipped)
        debug["age_extra_years"] = int(extra_years)
        st.json(debug)
