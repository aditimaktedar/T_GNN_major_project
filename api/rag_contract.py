def build_rag_query(drug_a, drug_b):

    return (
        "Drug-drug interaction evidence for "
        + drug_a
        + " and "
        + drug_b
    )


def build_response(
    prediction,
    evidence
):

    return {
        "model_prediction": prediction,
        "evidence": evidence,
        "note": (
            "T-GNN prediction and RAG evidence "
            "are reported separately."
        )
    }\n