from flask import Flask, render_template, request, session, make_response
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, rdMolDescriptors, Draw
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
import csv
import io
import json
import base64
import os
import sys
import time

app = Flask(__name__)

# -------------------------------------------------
# Deployment-friendly configuration
# -------------------------------------------------
app.secret_key = os.environ.get("SECRET_KEY", "molscore_pk_dev_secret_key")
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB upload limit
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

# If you deploy under HTTPS later, enable this:
# app.config["SESSION_COOKIE_SECURE"] = True

# -----------------------------
# Optional SA score import
# -----------------------------
SA_SCORER_AVAILABLE = False
sascorer = None

try:
    import sascorer  # works if already available in environment
    SA_SCORER_AVAILABLE = True
except ImportError:
    try:
        rdkit_contrib = os.path.join(sys.prefix, "share", "RDKit", "Contrib", "SA_Score")
        if os.path.isdir(rdkit_contrib) and rdkit_contrib not in sys.path:
            sys.path.append(rdkit_contrib)
        import sascorer
        SA_SCORER_AVAILABLE = True
    except Exception:
        SA_SCORER_AVAILABLE = False
        sascorer = None

# -----------------------------
# Filter catalogs
# -----------------------------
pains_params = FilterCatalogParams()
pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
PAINS_CATALOG = FilterCatalog(pains_params)

brenk_params = FilterCatalogParams()
brenk_params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
BRENK_CATALOG = FilterCatalog(brenk_params)


def clamp(value, low=0.0, high=6.0):
    return max(low, min(high, value))


def scale_range(value, low, high):
    if high == low:
        return 3.0
    scaled = ((value - low) / (high - low)) * 6.0
    return clamp(scaled)


def scale_inverse(value, low, high):
    return 6.0 - scale_range(value, low, high)


def get_filter_matches(mol, catalog):
    matches = catalog.GetMatches(mol)
    entries = []
    for match in matches:
        try:
            entries.append(match.GetDescription())
        except Exception:
            entries.append("Alert")
    return entries


def summarize_alerts(alert_list):
    if not alert_list:
        return "0 alert"
    unique_alerts = sorted(set(alert_list))
    if len(unique_alerts) == 1:
        return f"1 alert: {unique_alerts[0]}"
    return f"{len(unique_alerts)} alerts: " + "; ".join(unique_alerts[:5])


def calculate_sa_score(mol):
    if not SA_SCORER_AVAILABLE or sascorer is None:
        return "Not available"
    try:
        return round(float(sascorer.calculateScore(mol)), 2)
    except Exception:
        return "Not available"


def classify_rank(ranking_score):
    if ranking_score >= 4.5:
        return "Excellent"
    elif ranking_score >= 3.5:
        return "Good"
    elif ranking_score >= 2.5:
        return "Moderate"
    return "Poor"


def calculate_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)

    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    rot_bonds = Lipinski.NumRotatableBonds(mol)
    heavy_atoms = Lipinski.HeavyAtomCount(mol)
    ring_count = Lipinski.RingCount(mol)
    aromatic_rings = Lipinski.NumAromaticRings(mol)
    frac_csp3 = Lipinski.FractionCSP3(mol)
    mol_refractivity = Crippen.MolMR(mol)
    formal_charge = Chem.GetFormalCharge(mol)
    atom_count = mol.GetNumAtoms()
    heteroatoms = rdMolDescriptors.CalcNumHeteroatoms(mol)
    valence_electrons = Descriptors.NumValenceElectrons(mol)

    lipinski_violations = sum([
        mw > 500,
        logp > 5,
        hbd > 5,
        hba > 10
    ])

    veber_pass = (rot_bonds <= 10 and tpsa <= 140)

    # Descriptor/rule-based assessments
    bbb_assessment = "Favorable" if (mw <= 450 and 1 <= logp <= 5 and tpsa <= 90 and hbd <= 3) else "Unfavorable"
    absorption_assessment = "Favorable" if (tpsa <= 140 and rot_bonds <= 10) else "Less favorable"
    permeability_assessment = "Favorable" if (mw <= 500 and logp <= 5 and tpsa <= 120) else "Limited"

    solubility_risk = "High risk" if logp > 5 else "Manageable"
    flexibility_risk = "High" if rot_bonds > 10 else "Acceptable"
    polarity_class = "Low" if tpsa < 75 else ("Medium" if tpsa <= 140 else "High")
    size_class = "Small" if mw < 300 else ("Medium" if mw <= 500 else "Large")
    hydrophobicity_class = "Low" if logp < 1 else ("Moderate" if logp <= 5 else "High")

    ghose_like = "Pass" if (160 <= mw <= 480 and -0.4 <= logp <= 5.6 and 20 <= atom_count <= 70) else "Fail"
    rule_of_3 = "Pass" if (mw <= 300 and logp <= 3 and hbd <= 3 and hba <= 3 and rot_bonds <= 3) else "Fail"
    lead_like = "Pass" if (mw <= 450 and logp <= 4 and rot_bonds <= 10) else "Fail"

    # Descriptor-based ranking
    mw_score = scale_inverse(mw, 200, 500)
    logp_score = scale_inverse(abs(logp - 2.5), 0, 5)
    tpsa_score = scale_inverse(tpsa, 20, 140)
    flex_score = scale_inverse(rot_bonds, 0, 10)

    ranking_score = round((mw_score + logp_score + tpsa_score + flex_score) / 4, 3)
    rank_class = classify_rank(ranking_score)

    pains_matches = get_filter_matches(mol, PAINS_CATALOG)
    brenk_matches = get_filter_matches(mol, BRENK_CATALOG)
    sa_score = calculate_sa_score(mol)

    return {
        "canonical_smiles": canonical_smiles,
        "mw": round(mw, 2),
        "logp": round(logp, 2),
        "hbd": hbd,
        "hba": hba,
        "tpsa": round(tpsa, 2),
        "rot_bonds": rot_bonds,
        "heavy_atoms": heavy_atoms,
        "ring_count": ring_count,
        "aromatic_rings": aromatic_rings,
        "frac_csp3": round(frac_csp3, 3),
        "mol_refractivity": round(mol_refractivity, 2),
        "formal_charge": formal_charge,
        "atom_count": atom_count,
        "heteroatoms": heteroatoms,
        "valence_electrons": valence_electrons,
        "lipinski_violations": lipinski_violations,
        "veber_pass": "Pass" if veber_pass else "Fail",
        "bbb_assessment": bbb_assessment,
        "absorption_assessment": absorption_assessment,
        "permeability_assessment": permeability_assessment,
        "solubility_risk": solubility_risk,
        "flexibility_risk": flexibility_risk,
        "polarity_class": polarity_class,
        "size_class": size_class,
        "hydrophobicity_class": hydrophobicity_class,
        "ghose_like": ghose_like,
        "rule_of_3": rule_of_3,
        "lead_like": lead_like,
        "ranking_score": ranking_score,
        "rank_class": rank_class,
        "pains_alerts_count": len(set(pains_matches)),
        "pains_alerts_text": summarize_alerts(pains_matches),
        "brenk_alerts_count": len(set(brenk_matches)),
        "brenk_alerts_text": summarize_alerts(brenk_matches),
        "synthetic_accessibility": sa_score
    }


def parse_smiles_from_csv(file_storage):
    smiles_list = []
    if not file_storage or file_storage.filename == "":
        return smiles_list

    try:
        content = file_storage.read().decode("utf-8", errors="ignore")
        file_storage.seek(0)

        reader = csv.DictReader(io.StringIO(content))
        if reader.fieldnames is None:
            return smiles_list

        fieldnames_lower = [f.lower().strip() for f in reader.fieldnames]
        smiles_col = None

        for idx, col in enumerate(fieldnames_lower):
            if col == "smiles":
                smiles_col = reader.fieldnames[idx]
                break

        if smiles_col is None:
            return []

        for row in reader:
            val = row.get(smiles_col, "")
            if val and val.strip():
                smiles_list.append(val.strip())

        return smiles_list

    except Exception:
        return []


def collect_smiles_from_request(req):
    input_method = req.form.get("input_method", "batch")
    smiles_list = []

    if input_method == "single":
        single_smiles = req.form.get("single_smiles", "").strip()
        if single_smiles:
            smiles_list.append(single_smiles)

    elif input_method == "batch":
        batch_smiles = req.form.get("batch_smiles", "")
        smiles_list.extend([s.strip() for s in batch_smiles.splitlines() if s.strip()])

    elif input_method == "draw":
        drawn_smiles = req.form.get("drawn_smiles", "").strip()
        if drawn_smiles:
            smiles_list.append(drawn_smiles)

    elif input_method == "upload":
        uploaded = req.files.get("file_upload")
        smiles_list = parse_smiles_from_csv(uploaded)

    return input_method, smiles_list


def mol_image_base64(smiles, size=(360, 260)):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    img = Draw.MolToImage(mol, size=size)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def build_radar_axes(row):
    lipo = clamp(6.0 - abs(row["logp"] - 2.5) * 1.2)
    size = clamp(6.0 - abs(row["mw"] - 350) / 40.0)
    polar = scale_inverse(row["tpsa"], 20, 150)
    insol = clamp(6.0 - max(row["logp"], 0) * 0.8)
    insatu = scale_range(row["frac_csp3"], 0.0, 1.0)
    flex = scale_inverse(row["rot_bonds"], 0, 12)

    return [
        {"label": "LIPO", "value": round(lipo, 2)},
        {"label": "SIZE", "value": round(size, 2)},
        {"label": "POLAR", "value": round(polar, 2)},
        {"label": "INSOLU", "value": round(insol, 2)},
        {"label": "INSATU", "value": round(insatu, 2)},
        {"label": "FLEX", "value": round(flex, 2)}
    ]


def build_sections(row):
    return {
        "Physicochemical Properties": [
            ("Canonical SMILES", row["canonical_smiles"]),
            ("Formula surrogate", f"Atoms: {row['atom_count']} | Heavy atoms: {row['heavy_atoms']}"),
            ("Molecular weight", f"{row['mw']} g/mol"),
            ("TPSA", f"{row['tpsa']} Å²"),
            ("H-bond acceptors", row["hba"]),
            ("H-bond donors", row["hbd"]),
            ("Rotatable bonds", row["rot_bonds"]),
            ("Ring count", row["ring_count"]),
            ("Aromatic rings", row["aromatic_rings"]),
            ("Fraction Csp3", row["frac_csp3"]),
            ("Molar refractivity", row["mol_refractivity"])
        ],
        "Lipophilicity and Size": [
            ("LogP", row["logp"]),
            ("Hydrophobicity class", row["hydrophobicity_class"]),
            ("Size class", row["size_class"]),
            ("Formal charge", row["formal_charge"]),
            ("Heteroatoms", row["heteroatoms"]),
            ("Valence electrons", row["valence_electrons"])
        ],
        "PK-Oriented Assessments": [
            ("BBB assessment", row["bbb_assessment"]),
            ("Absorption assessment", row["absorption_assessment"]),
            ("Permeability assessment", row["permeability_assessment"]),
            ("Solubility risk", row["solubility_risk"]),
            ("Flexibility risk", row["flexibility_risk"]),
            ("Polarity class", row["polarity_class"])
        ],
        "Drug-Likeness Rules": [
            ("Lipinski violations", row["lipinski_violations"]),
            ("Veber", row["veber_pass"]),
            ("Ghose", row["ghose_like"]),
            ("Rule of 3", row["rule_of_3"]),
            ("Lead-like", row["lead_like"])
        ],
        "Medicinal Chemistry": [
            ("PAINS", row["pains_alerts_text"]),
            ("Brenk", row["brenk_alerts_text"]),
            ("Synthetic accessibility", row["synthetic_accessibility"])
        ],
        "MolScore-PK Ranking": [
            ("Descriptor-based ranking score", row["ranking_score"]),
            ("Ranking class", row["rank_class"]),
            ("Status", row["status"]),
            ("Interpretation", "Calculated descriptors with rule-based assessment for early-stage molecular profiling")
        ]
    }


def build_download_rows(results):
    csv_rows = []
    for row in results:
        csv_rows.append({
            "id": row["id"],
            "smiles": row["smiles"],
            "canonical_smiles": row.get("canonical_smiles", ""),
            "status": row["status"],
            "mw": row.get("mw", ""),
            "logp": row.get("logp", ""),
            "hbd": row.get("hbd", ""),
            "hba": row.get("hba", ""),
            "tpsa": row.get("tpsa", ""),
            "rot_bonds": row.get("rot_bonds", ""),
            "heavy_atoms": row.get("heavy_atoms", ""),
            "ring_count": row.get("ring_count", ""),
            "aromatic_rings": row.get("aromatic_rings", ""),
            "frac_csp3": row.get("frac_csp3", ""),
            "mol_refractivity": row.get("mol_refractivity", ""),
            "bbb_assessment": row.get("bbb_assessment", ""),
            "absorption_assessment": row.get("absorption_assessment", ""),
            "permeability_assessment": row.get("permeability_assessment", ""),
            "solubility_risk": row.get("solubility_risk", ""),
            "flexibility_risk": row.get("flexibility_risk", ""),
            "polarity_class": row.get("polarity_class", ""),
            "size_class": row.get("size_class", ""),
            "hydrophobicity_class": row.get("hydrophobicity_class", ""),
            "lipinski_violations": row.get("lipinski_violations", ""),
            "veber_pass": row.get("veber_pass", ""),
            "ghose_like": row.get("ghose_like", ""),
            "rule_of_3": row.get("rule_of_3", ""),
            "lead_like": row.get("lead_like", ""),
            "pains_alerts_count": row.get("pains_alerts_count", ""),
            "pains_alerts_text": row.get("pains_alerts_text", ""),
            "brenk_alerts_count": row.get("brenk_alerts_count", ""),
            "brenk_alerts_text": row.get("brenk_alerts_text", ""),
            "synthetic_accessibility": row.get("synthetic_accessibility", ""),
            "ranking_score": row.get("ranking_score", ""),
            "rank_class": row.get("rank_class", "")
        })
    return csv_rows


@app.after_request
def add_no_cache_headers(response):
    """
    Avoid browser caching for dynamic pages during updates and deployment testing.
    """
    if request.path in ["/", "/prediction", "/documentation", "/predict", "/download_csv"]:
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/prediction", methods=["GET"])
def prediction():
    return render_template("prediction.html", selected_method="draw")


@app.route("/documentation")
def documentation():
    return render_template("documentation.html")


@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()
    input_method, smiles_list = collect_smiles_from_request(request)

    if not smiles_list:
        return render_template(
            "prediction.html",
            error="No valid molecular input detected. Please enter SMILES, draw a molecule, or upload a CSV with a 'smiles' column.",
            selected_method=input_method
        )

    if len(smiles_list) > 50:
        return render_template(
            "prediction.html",
            error="Maximum 50 molecules are allowed per run.",
            selected_method=input_method
        )

    results = []

    for i, smiles in enumerate(smiles_list, start=1):
        props = calculate_properties(smiles)

        if props is None:
            results.append({
                "id": i,
                "smiles": smiles,
                "status": "Invalid SMILES",
                "image": None,
                "radar_axes": [],
                "sections": {}
            })
        else:
            row = {
                "id": i,
                "smiles": smiles,
                "status": "Valid"
            }
            row.update(props)
            row["image"] = mol_image_base64(smiles)
            row["radar_axes"] = build_radar_axes(row)
            row["sections"] = build_sections(row)
            results.append(row)

    try:
        session["last_results"] = json.dumps(build_download_rows(results))
    except Exception:
        session["last_results"] = "[]"

    elapsed = round(time.time() - start_time, 2)
    return render_template("results.html", results=results, runtime=elapsed)


@app.route("/download_csv", methods=["GET"])
def download_csv():
    raw_results = session.get("last_results")
    if not raw_results:
        return render_template(
            "prediction.html",
            error="No results are available for download. Please run a prediction first.",
            selected_method="draw"
        )

    try:
        results = json.loads(raw_results)
    except Exception:
        results = []

    output = io.StringIO()
    writer = csv.writer(output)

    if results:
        headers = list(results[0].keys())
        writer.writerow(headers)
        for row in results:
            writer.writerow([row.get(h, "") for h in headers])

    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = "attachment; filename=molscore_pk_results.csv"
    response.headers["Content-Type"] = "text/csv; charset=utf-8"
    return response


@app.errorhandler(413)
def file_too_large(error):
    return render_template(
        "prediction.html",
        error="Uploaded file is too large. Please upload a smaller CSV file.",
        selected_method="upload"
    ), 413


@app.errorhandler(404)
def page_not_found(error):
    return render_template(
        "prediction.html",
        error="Requested page was not found.",
        selected_method="draw"
    ), 404


@app.errorhandler(500)
def internal_server_error(error):
    return render_template(
        "prediction.html",
        error="An internal server error occurred. Please try again.",
        selected_method="draw"
    ), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)