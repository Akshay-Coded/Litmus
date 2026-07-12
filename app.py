import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw

from src.config.tox21_model_config import NR_TASKS, SR_TASKS
from src.inference.esol_applicability import (
    confidence_verdict,
    load_train_reference,
    nearest_train_molecules,
)
from src.inference.esol_predictor import load_ensemble, predict_smiles
from src.inference.tox21_applicability import (
    is_structurally_novel,
)
from src.inference.tox21_applicability import (
    load_train_reference as load_tox21_train_reference,
)
from src.inference.tox21_applicability import (
    nearest_train_molecules as tox21_nearest_train_molecules,
)
from src.inference.tox21_predictor import (
    load_assay_reliability,
    load_model,
    predict_smiles as predict_tox21_smiles,
)

st.set_page_config(
    page_title="Litmus",
    page_icon="\U0001f9ea",
    layout="wide",
)

# Streamlit left-aligns headings/images/metrics by default -- center them,
# but ONLY inside each tab's result-panel container (key="..._result_panel"
# -> CSS class "st-key-..._result_panel"), so the About tab keeps its
# normal left-aligned reading layout.
st.markdown(
    """
    <style>
    .st-key-sol_result_panel div[data-testid="stHeading"],
    .st-key-sol_result_panel div[data-testid="stMarkdown"] h1,
    .st-key-sol_result_panel div[data-testid="stMarkdown"] h2,
    .st-key-sol_result_panel div[data-testid="stMarkdown"] h3,
    .st-key-sol_result_panel div[data-testid="stMarkdown"] h4,
    .st-key-sol_result_panel div[data-testid="stMarkdown"] h5,
    .st-key-sol_result_panel div[data-testid="stMarkdown"] h6,
    .st-key-tox_result_panel div[data-testid="stHeading"],
    .st-key-tox_result_panel div[data-testid="stMarkdown"] h1,
    .st-key-tox_result_panel div[data-testid="stMarkdown"] h2,
    .st-key-tox_result_panel div[data-testid="stMarkdown"] h3,
    .st-key-tox_result_panel div[data-testid="stMarkdown"] h4,
    .st-key-tox_result_panel div[data-testid="stMarkdown"] h5,
    .st-key-tox_result_panel div[data-testid="stMarkdown"] h6 {
        text-align: center;
    }
    .st-key-sol_result_panel div[data-testid="stMetric"],
    .st-key-sol_result_panel div[data-testid="stCaptionContainer"],
    .st-key-sol_result_panel div[data-testid="stAlert"],
    .st-key-tox_result_panel div[data-testid="stMetric"],
    .st-key-tox_result_panel div[data-testid="stCaptionContainer"],
    .st-key-tox_result_panel div[data-testid="stAlert"] {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

SOLUBILITY_EXAMPLES = {
    "Caffeine (in-domain)": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Aspirin (borderline)": "CC(=O)Oc1ccccc1C(=O)O",
    "Halogenated polycycle (out-of-domain)": "ClC1(C(=O)C2(Cl)C3(Cl)C14Cl)C5(Cl)C2(Cl)C3(Cl)C(Cl)(Cl)C45Cl",
}

TOX21_EXAMPLES = {
    "Triarylmethane dye (promiscuous toxicant)": "CN(C)c1ccc(C(=C2C=CC(=[N+](C)C)C=C2)c2ccc(N(C)C)cc2)cc1",
    "Glucose (benign)": "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
    "Halogenated polycycle (out-of-domain)": "ClC1(C(=O)C2(Cl)C3(Cl)C14Cl)C5(Cl)C2(Cl)C3(Cl)C(Cl)(Cl)C45Cl",
}

VERDICT_STYLE = {
    "green": ("\U0001f7e2", "In-domain", st.success),
    "amber": ("\U0001f7e1", "Borderline", st.warning),
    "red": ("\U0001f534", "Out-of-domain", st.error),
}

RELIABILITY_STYLE = {
    "high": ("\U0001f7e2", "reliable"),
    "medium": ("\U0001f7e1", "moderate"),
    "low": ("\U0001f534", "low-confidence"),
}

ACTIVE_THRESHOLD = 0.5


def show_centered_image(pil_image, width):
    """Centers a fixed-width image using Streamlit's own column layout
    instead of CSS -- flanking spacer columns leave no room for the image
    to drift left, unlike relying on a guessed internal CSS class name.
    """
    _, center, _ = st.columns([1, 2, 1])
    with center:
        st.image(pil_image, width=width)


@st.cache_resource
def get_solubility_models():
    return load_ensemble()


@st.cache_resource
def get_solubility_train_reference():
    return load_train_reference()


@st.cache_resource
def get_tox21_model():
    return load_model()


@st.cache_resource
def get_tox21_reliability():
    return load_assay_reliability()


@st.cache_resource
def get_tox21_train_reference():
    return load_tox21_train_reference()


def render_assay_row(task, prob, reliability):
    emoji, label = RELIABILITY_STYLE[reliability["reliability"]]
    name_col, bar_col, badge_col = st.columns([2, 3, 1.6])
    name_col.markdown(f"**{task}**")
    bar_col.progress(prob, text=f"P(active) = {prob:.2f}")
    badge_col.markdown(f"{emoji} {label}")


def render_solubility_tab():
    st.title("\U0001f9ea Solubility Predictor")
    st.caption(
        "Predicts aqueous solubility from a SMILES string and tells you when not to trust it."
    )

    if "sol_smiles_input" not in st.session_state:
        st.session_state.sol_smiles_input = SOLUBILITY_EXAMPLES["Caffeine (in-domain)"]

    example_cols = st.columns(len(SOLUBILITY_EXAMPLES))
    for col, (name, example_smiles) in zip(example_cols, SOLUBILITY_EXAMPLES.items()):
        if col.button(name, use_container_width=True, key=f"sol_ex_{name}"):
            st.session_state.sol_smiles_input = example_smiles

    smiles = st.text_input("SMILES string", key="sol_smiles_input")
    mol = Chem.MolFromSmiles(smiles) if smiles else None

    if smiles and mol is None:
        st.error(f"Couldn't parse '{smiles}' as a valid SMILES string.")
    elif mol is not None:
        models, weights = get_solubility_models()
        train_ref = get_solubility_train_reference()

        result = predict_smiles(smiles, models=models, weights=weights)
        neighbors, max_similarity = nearest_train_molecules(smiles, train_ref, k=3)
        level, reason = confidence_verdict(max_similarity, result["uncertainty"])
        emoji, label, verdict_box = VERDICT_STYLE[level]

        with st.container(key="sol_result_panel"):
            col_structure, col_prediction, col_confidence = st.columns([1, 1, 1.3])

            CARD_HEIGHT = 420

            with col_structure, st.container(border=True, height=CARD_HEIGHT):
                st.markdown("##### Structure")
                show_centered_image(Draw.MolToImage(mol, size=(260, 260)), width=260)

            with col_prediction, st.container(border=True, height=CARD_HEIGHT):
                st.markdown("##### Predicted solubility")
                st.metric(
                    label="log solubility (mol/L)",
                    value=f"{result['prediction']:.2f} ± {result['uncertainty']:.2f}",
                )
                st.caption("point estimate ± spread across the 5-model ensemble")
                with st.expander("Individual model predictions"):
                    for i, p in enumerate(result["per_model_predictions"], 1):
                        st.text(f"model {i}: {p:.2f}")

            with col_confidence, st.container(border=True, height=CARD_HEIGHT):
                st.markdown("##### Confidence")
                verdict_box(f"**{emoji} {label}**\n\n{reason}")

            st.divider()
            st.subheader("Nearest training molecules")
            st.caption(
                "The most structurally similar molecules the model actually saw during training."
            )
            neighbor_cols = st.columns(len(neighbors))
            for col, neighbor in zip(neighbor_cols, neighbors):
                neighbor_mol = Chem.MolFromSmiles(neighbor["smiles"])
                with col:
                    show_centered_image(Draw.MolToImage(neighbor_mol, size=(200, 200)), width=200)
                    st.caption(f"similarity: {neighbor['similarity']:.2f}")
                    st.caption(f"true log solubility: {neighbor['true_solubility']:.2f}")

    st.divider()
    st.markdown("#### Model & evaluation")
    st.write(
        "GINE graph neural network, ensemble of 5 seeds. Evaluated on a scaffold split (test "
        "molecules share no core ring system with training) -- a harder, more honest measure "
        "than the random-split numbers usually quoted for ESOL."
    )
    metric_cols = st.columns(3)
    metric_cols[0].metric("Test RMSE", "0.935", help="log mol/L, scaffold-split test set")
    metric_cols[1].metric("Test R²", "0.807", help="variance explained on unseen scaffolds")
    metric_cols[2].metric(
        "Test molecules", "111", help="held out, no scaffold overlap with training"
    )
    st.caption(
        "Mean absolute error roughly doubles (about 0.92 vs 0.65 log units) for molecules with "
        "less than 0.3 Tanimoto similarity to anything in training. Halogenated compounds and "
        "large polycyclic ring systems account for most of the bad predictions. The confidence "
        "badge combines model disagreement with structural novelty."
    )


def render_toxicity_tab():
    st.title("☠️ Toxicity Predictor (Tox21)")
    st.caption(
        "Predicts activity across 12 Tox21 assays from a SMILES string, and tells you which of "
        "those 12 predictions to actually trust."
    )
    st.info(
        "Probabilities below are **relative activity scores, not calibrated likelihoods** -- no "
        "temperature scaling or other calibration has been applied. A value of 0.70 means "
        "\"more likely active than 0.30\", not \"70% chance of toxicity\".",
        icon="ℹ️",
    )

    if "tox_smiles_input" not in st.session_state:
        st.session_state.tox_smiles_input = TOX21_EXAMPLES["Glucose (benign)"]

    example_cols = st.columns(len(TOX21_EXAMPLES))
    for col, (name, example_smiles) in zip(example_cols, TOX21_EXAMPLES.items()):
        if col.button(name, use_container_width=True, key=f"tox_ex_{name}"):
            st.session_state.tox_smiles_input = example_smiles

    smiles = st.text_input("SMILES string", key="tox_smiles_input")
    mol = Chem.MolFromSmiles(smiles) if smiles else None

    if smiles and mol is None:
        st.error(f"Couldn't parse '{smiles}' as a valid SMILES string.")
    elif mol is not None:
        model = get_tox21_model()
        reliability = get_tox21_reliability()
        train_ref = get_tox21_train_reference()

        result = predict_tox21_smiles(smiles, model=model)
        probs = result["probabilities"]
        neighbors, max_similarity = tox21_nearest_train_molecules(smiles, train_ref, k=3)
        novel = is_structurally_novel(max_similarity)

        with st.container(key="tox_result_panel"):
            col_structure, col_summary = st.columns([1, 2])

            with col_structure, st.container(border=True, height=300):
                st.markdown("##### Structure")
                show_centered_image(Draw.MolToImage(mol, size=(220, 220)), width=220)

            with col_summary, st.container(border=True, height=300):
                st.markdown("##### Structural novelty")
                if novel:
                    st.error(
                        f"\U0001f534 Structurally novel (similarity {max_similarity:.2f} to the "
                        f"nearest training molecule, below the 0.3 threshold) -- treat **all 12** "
                        f"predictions below as low-confidence guesses, not just the low-reliability ones."
                    )
                else:
                    st.success(
                        f"\U0001f7e2 Structurally similar to training chemistry "
                        f"(similarity {max_similarity:.2f})."
                    )
                st.caption(
                    "Unlike solubility, scaffold similarity is only a weak predictor of error here "
                    "(see About tab) -- it's a coarse novelty flag, not a fine-grained confidence score."
                )
                n_flagged = sum(1 for p in probs.values() if p >= ACTIVE_THRESHOLD)
                st.metric("Assays flagged active (of 12)", n_flagged)

            st.divider()
            st.subheader("Flagged assays")
            flagged = sorted(
                [(t, p) for t, p in probs.items() if p >= ACTIVE_THRESHOLD],
                key=lambda tp: -tp[1],
            )
            if not flagged:
                st.success("Not predicted active on any of the 12 assays.")
            else:
                for task, prob in flagged:
                    render_assay_row(task, prob, reliability[task])

            with st.expander(f"Show all 12 assays, grouped by panel ({len(probs) - len(flagged)} predicted inactive)"):
                st.markdown("###### NR (nuclear receptor) panel")
                for task in NR_TASKS:
                    render_assay_row(task, probs[task], reliability[task])
                st.markdown("###### SR (stress response) panel")
                for task in SR_TASKS:
                    render_assay_row(task, probs[task], reliability[task])
                st.caption(
                    "EDA found the NR/SR split isn't as clean as the biological grouping suggests -- "
                    "some SR assays (e.g. SR-MMP, SR-p53) correlate with several NR assays about as "
                    "strongly as with each other. Treat the grouping as organizational, not a claim "
                    "that panels are independent."
                )

            st.divider()
            st.subheader("Nearest training molecules")
            st.caption("The most structurally similar molecules the model actually saw during training.")
            neighbor_cols = st.columns(len(neighbors))
            for col, neighbor in zip(neighbor_cols, neighbors):
                neighbor_mol = Chem.MolFromSmiles(neighbor["smiles"])
                with col:
                    show_centered_image(Draw.MolToImage(neighbor_mol, size=(180, 180)), width=180)
                    st.caption(f"similarity: {neighbor['similarity']:.2f}")
                    active_on = [t for t, v in neighbor["labels"].items() if v == 1.0]
                    st.caption(
                        f"active on: {', '.join(active_on)}" if active_on else "active on: none tested"
                    )

    st.divider()
    st.markdown("#### Model & evaluation")
    st.write(
        "Same GINE backbone as the solubility model, with one shared trunk feeding 12 "
        "classification heads instead of one regression output. Missing labels (most assays are "
        "only sparsely tested) are masked out of the loss rather than treated as negative. "
        "Evaluated on a scaffold split, one seed trained so far (no ensemble yet)."
    )
    metric_cols = st.columns(3)
    metric_cols[0].metric("Mean test AUC", "0.765", help="averaged across all 12 assays")
    metric_cols[1].metric("Mean test AUPRC", "0.361", help="more informative than AUC given 3-16% positive rates")
    metric_cols[2].metric("Test molecules", "781", help="held out, no scaffold overlap with training")

    st.markdown("##### Per-assay reliability -- why some predictions are worth more than others")
    st.write(
        "Every assay's test-fold AUC has a different amount of statistical support behind it, "
        "driven by how many positives that assay actually had. NR-ER-LBD's AUC (0.80) is built on "
        "just 28 test positives, giving it a standard error of about 0.05 (95% CI roughly "
        "[0.70, 0.90]) -- versus NR-AhR's 92 positives and SE of about 0.03. The reliability badge "
        "on every assay row above (\U0001f7e2 high / \U0001f7e1 moderate / \U0001f534 low) is this "
        "standard error, computed directly from each assay's test-set support, not a guess."
    )
    st.caption(
        "Scaffold similarity to training data is a much weaker confidence signal here than for "
        "solubility -- bucketing test molecules by similarity showed almost no relationship with "
        "per-molecule prediction error (Brier score ~0.19-0.21 across low/mid/high similarity "
        "terciles), except for a real but modest jump for the most novel molecules (<0.3 "
        "similarity). So this tab leans on per-assay reliability and the predicted probability's "
        "own distance from 0.5 as the primary confidence signals, with structural novelty as a "
        "coarse secondary flag -- not a fine-grained badge the way it is for solubility."
    )


def render_about_tab():
    st.title("ℹ️ About Litmus")
    st.write(
        "Litmus is two molecular property predictors -- aqueous solubility (regression) and "
        "Tox21 toxicity (12-assay multi-task classification) -- sharing one methodology, not one "
        "model. Each tab is a separate model trained on its own data; the tabs are kept visually "
        "distinct on purpose so it's clear these are two different predictions, not one number reused."
    )

    st.markdown("##### Shared methodology")
    st.write(
        "Both models are GINE (Graph Isomorphism Network with Edge features) message-passing "
        "networks that read the molecular graph directly -- atoms and bonds -- instead of "
        "precomputed chemical descriptors. Bond features (bond type, conjugation, ring membership) "
        "matter for both solubility and reactivity-driven toxicity mechanisms."
    )
    st.write(
        "Both are evaluated on a **scaffold split** rather than a random one. A random split can "
        "put near-identical molecules in both train and test, inflating scores by rewarding "
        "memorization. A scaffold split forces test molecules to share no core ring system with "
        "any training molecule, so every reported test score reflects real generalization to "
        "unseen chemistry. Numbers here are lower than commonly-quoted random-split figures for "
        "both ESOL and Tox21 -- that's the expected cost of measuring the harder, more honest task."
    )
    st.write(
        "Both pipelines were built the same way: clean the raw data with SMILES validation and "
        "deduplication preserved (never fill missing labels with a placeholder value), run an EDA "
        "pass that locks the loss/metric/split decisions in data rather than convention, then train "
        "and evaluate once against a held-out test set."
    )

    st.markdown("##### Where the two properties diverge")
    st.write(
        "Solubility is one continuous target with a five-model ensemble, so its confidence signal "
        "is ensemble disagreement plus scaffold similarity -- both are strong, well-correlated "
        "signals here. Toxicity is 12 sparse, imbalanced binary labels with only one trained seed "
        "so far, so its confidence signal is different in kind: a per-assay reliability tier "
        "computed from each assay's test-set statistical support (via the Hanley-McNeil AUC "
        "standard error), plus the predicted probability's own distance from 0.5. Scaffold "
        "similarity, which drives most of the solubility tab's confidence badge, turned out to be "
        "a much weaker signal for toxicity -- so the toxicity tab doesn't pretend otherwise."
    )

    st.caption(
        "Source: notebooks/esol_eda.ipynb and notebooks/esol_error_analysis.ipynb (solubility), "
        "notebooks/tox21_eda.ipynb (toxicity)."
    )


st.title("\U0001f9ea Litmus")
st.caption("Molecular property prediction, one honest model per property.")

tab_sol, tab_tox, tab_about = st.tabs(["\U0001f9ea Solubility", "☠️ Toxicity", "ℹ️ About"])

with tab_sol:
    render_solubility_tab()

with tab_tox:
    render_toxicity_tab()

with tab_about:
    render_about_tab()
