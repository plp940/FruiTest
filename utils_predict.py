import numpy as np

def interpret_predictions_with_rules(pred_vector, class_names, threshold=0.8):
    """
    Takes the softmax prediction vector and applies the 5-rule system.
    Returns the predicted category (Fresh/Rotten) and an appropriate message.
    """
    sorted_indices = np.argsort(pred_vector)[::-1]  # indices of predictions sorted by confidence
    top1_index = sorted_indices[0]
    top2_index = sorted_indices[1]
    top1_conf = pred_vector[top1_index]
    top2_conf = pred_vector[top2_index]

    top1_label = class_names[top1_index]
    top2_label = class_names[top2_index]

    # Rule 1: Extract category only (Fresh or Rotten)
    def get_category(label):
        return "Fresh" if "Fresh" in label else "Rotten"

    top1_cat = get_category(top1_label)
    top2_cat = get_category(top2_label)

    output_message = ""

    if top1_conf >= threshold:
        output_message += f"✅ Prediction: **{top1_cat}** ({top1_conf*100:.2f}%)\n\n"
        output_message += "🟢 Looks safe to eat!" if top1_cat == "Fresh" else "🔴 Not safe to eat."
    else:
        if top1_cat != top2_cat:
            output_message += f"⚠️ Mixed Prediction:\n- {top1_cat}: {top1_conf*100:.2f}%\n- {top2_cat}: {top2_conf*100:.2f}%\n\n"
            if top1_cat == "Rotten":
                output_message += "🔴 Better avoid eating this fruit."
            else:
                output_message += "🟡 Seems to be fresh, but not confident enough."
        else:
            output_message += f"⚠️ Low confidence prediction but consistent:\n- {top1_cat}: {top1_conf*100:.2f}%\n\n"
            if top1_cat == "Rotten":
                output_message += "🔴 Caution: It may not be safe to eat."
            else:
                output_message += "🟡 Seems fresh, but prediction confidence is low."

    return output_message
