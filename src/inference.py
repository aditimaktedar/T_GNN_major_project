import torch


SEVERITY_CLASSES = [
    "Minor",
    "Moderate",
    "Major"
]


def load_checkpoint(
    model,
    checkpoint_path,
    device=None
):

    if device is None:

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    if (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):
        state_dict = checkpoint["state_dict"]

    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.to(device)
    model.eval()

    return model


@torch.no_grad()
def predict_pair(
    model,
    drug_a_idx,
    drug_b_idx,
    edge_index
):

    device = next(
        model.parameters()
    ).device

    drug_a = torch.tensor(
        [drug_a_idx],
        dtype=torch.long,
        device=device
    )

    drug_b = torch.tensor(
        [drug_b_idx],
        dtype=torch.long,
        device=device
    )

    edge_index = edge_index.to(device)

    output = model(
        drug_a,
        drug_b,
        edge_index
    )

    presence_probability = float(
        torch.sigmoid(
            output["presence_logits"]
        ).squeeze().item()
    )

    severity_probability = torch.softmax(
        output["severity_logits"],
        dim=-1
    )[0]

    severity_index = int(
        torch.argmax(
            severity_probability
        ).item()
    )

    return {
        "presence_probability":
            presence_probability,

        "presence_label":
            (
                "Yes"
                if presence_probability >= 0.5
                else "No"
            ),

        "severity_label":
            SEVERITY_CLASSES[
                severity_index
            ],

        "severity_probabilities": {
            label: float(
                severity_probability[i].item()
            )
            for i, label in enumerate(
                SEVERITY_CLASSES
            )
        }
    }\n