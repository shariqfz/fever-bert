import six

def check_predicted_evidence_format(instance):
    if 'predicted_evidence' in instance.keys() and len(instance['predicted_evidence']):
        assert all(isinstance(prediction, list)
                   for prediction in instance["predicted_evidence"]), \
            "Predicted evidence must be a list of (page,line) lists"

        assert all(len(prediction) == 2
                   for prediction in instance["predicted_evidence"]), \
            "Predicted evidence must be a list of (page,line) lists"

        assert all(isinstance(prediction[0], six.string_types)
                    for prediction in instance["predicted_evidence"]), \
            "Predicted evidence must be a list of (page<string>,line<int>) lists"

        assert all(isinstance(prediction[1], int)
                   for prediction in instance["predicted_evidence"]), \
            "Predicted evidence must be a list of (page<string>,line<int>) lists"

def is_correct_label_cm(instance, confusion_matrix):
    pred_label = instance["predicted_label"].upper()
    true_label = instance["label"].upper()

    if true_label == pred_label :
        confusion_matrix[true_label]["TP"] += 1
    else:
        confusion_matrix[true_label]["FN"] += 1
        confusion_matrix[pred_label]["FP"] += 1

    return true_label == pred_label

def is_correct_label(instance):
    return instance["label"].upper() == instance["predicted_label"].upper()


def is_strictly_correct(instance, max_evidence=None):
    #Strict evidence matching is only for NEI class
    check_predicted_evidence_format(instance)

    if instance["label"].upper() != "NOT ENOUGH INFO" and is_correct_label(instance):
        assert 'predicted_evidence' in instance, "Predicted evidence must be provided for strict scoring"

        if max_evidence is not None:
            instance["predicted_evidence"] = instance["predicted_evidence"][:max_evidence]


        for evience_group in instance["evidence"]:
            #Filter out the annotation ids. We just want the evidence page and line number
            actual_sentences = [[e[2], e[3]] for e in evience_group]
            #Only return true if an entire group of actual sentences is in the predicted sentences
            if all([actual_sent in instance["predicted_evidence"] for actual_sent in actual_sentences]):
                return True

    #If the class is NEI, we don't score the evidence retrieval component
    elif instance["label"].upper() == "NOT ENOUGH INFO" and is_correct_label(instance):
        return True

    return False


def evidence_macro_precision(instance, max_evidence=None):
    this_precision = 0.0
    this_precision_hits = 0.0

    if instance["label"].upper() != "NOT ENOUGH INFO":
        all_evi = [[e[2], e[3]] for eg in instance["evidence"] for e in eg if e[3] is not None]

        predicted_evidence = instance["predicted_evidence"] if max_evidence is None else \
                                                                        instance["predicted_evidence"][:max_evidence]

        for prediction in predicted_evidence:
            if prediction in all_evi:
                this_precision += 1.0
            this_precision_hits += 1.0

        return (this_precision / this_precision_hits) if this_precision_hits > 0 else 1.0, 1.0

    return 0.0, 0.0

def evidence_macro_recall(instance, max_evidence=None):
    # We only want to score F1/Precision/Recall of recalled evidence for NEI claims
    if instance["label"].upper() != "NOT ENOUGH INFO":
        # If there's no evidence to predict, return 1
        if len(instance["evidence"]) == 0 or all([len(eg) == 0 for eg in instance]):
           return 1.0, 1.0

        predicted_evidence = instance["predicted_evidence"] if max_evidence is None else \
                                                                        instance["predicted_evidence"][:max_evidence]

        for evidence_group in instance["evidence"]:
            evidence = [[e[2], e[3]] for e in evidence_group]
            if all([item in predicted_evidence for item in evidence]):
                # We only want to score complete groups of evidence. Incomplete groups are worthless.
                return 1.0, 1.0
        return 0.0, 1.0
    return 0.0, 0.0

def evidence_micro_recall(instance, max_evidence=None):
    this_recall = 0.0
    this_recall_hits = 0.0

    if instance["label"].upper() != "NOT ENOUGH INFO":
        if len(instance["evidence"]) == 0 or all([len(eg) == 0 for eg in instance]):
           return 1.0, 1.0


        predicted_evidence = instance["predicted_evidence"] if max_evidence is None else \
                                                                        instance["predicted_evidence"][:max_evidence]

        for evidence_group in instance["evidence"]:
            evidence = [[e[2], e[3]] for e in evidence_group]
            if all([item in predicted_evidence for item in evidence]):
                this_recall += 1.0
            this_recall_hits += 1.0

    return this_recall, this_recall_hits


# Micro is not used. This code is just included to demostrate our model of macro/micro
def evidence_micro_precision(instance):
    this_precision = 0
    this_precision_hits = 0

    # We only want to score Macro F1/Precision/Recall of recalled evidence for NEI claims
    if instance["label"].upper() != "NOT ENOUGH INFO":
        all_evi = [[e[2], e[3]] for eg in instance["evidence"] for e in eg if e[3] is not None]

        for prediction in instance["predicted_evidence"]:
            if prediction in all_evi:
                this_precision += 1.0
            this_precision_hits += 1.0

    return this_precision, this_precision_hits

def fever_score_micro_f1(predictions,actual=None, max_evidence=5):
    correct = 0
    strict = 0

    micro_precision = 0
    micro_precision_hits = 0

    micro_recall = 0
    micro_recall_hits = 0

    class_map = {"REFUTES": 0, "SUPPORTS": 1, "NOT ENOUGH INFO": 2}

    confusion_matrix = {"REFUTES": {"TP": 0, "FP": 0, "FN": 0},
                        "SUPPORTS": {"TP": 0, "FP": 0, "FN": 0},
                        "NOT ENOUGH INFO": {"TP": 0, "FP": 0, "FN": 0}}
    
    recall_at_k = {1: 0, 10: 0, 100: 0}

    for idx,instance in enumerate(predictions):
        assert 'predicted_evidence' in instance.keys(), 'evidence must be provided for the prediction'

        #If it's a blind test set, we need to copy in the values from the actual data
        if 'evidence' not in instance or 'label' not in instance:
            assert actual is not None, 'in blind evaluation mode, actual data must be provided'
            assert len(actual) == len(predictions), 'actual data and predicted data length must match'
            assert 'evidence' in actual[idx].keys(), 'evidence must be provided for the actual evidence'
            instance['evidence'] = actual[idx]['evidence']
            instance['label'] = actual[idx]['label']

        assert 'evidence' in instance.keys(), 'gold evidence must be provided'

        if is_correct_label_cm(instance, confusion_matrix):
            correct += 1.0

            if is_strictly_correct(instance, max_evidence):
                strict+=1.0

        micro_prec = evidence_micro_precision(instance)
        micro_precision += micro_prec[0]
        micro_precision_hits += micro_prec[1]

        micro_rec = evidence_micro_recall(instance, max_evidence)
        micro_recall += micro_rec[0]
        micro_recall_hits += micro_rec[1]

        for k in recall_at_k.keys():
            recalled, total_true =  micro_rec[0], micro_rec[1]
            rc = recalled / total_true if total_true > 0 else 1.0
            recall_at_k[k] += (rc if total_true < k else recalled / k)

    true_positives = sum([confusion_matrix[class_]["TP"] for class_ in class_map.keys() ])
    false_positives = sum([confusion_matrix[class_]["FP"] for class_ in class_map.keys() ])
    false_negatives = sum([confusion_matrix[class_]["FN"] for class_ in class_map.keys() ])
    
    denominator = true_positives +  (false_positives + false_negatives) / 2
    micro_f1 = true_positives / denominator if denominator > 0 else 1.0

    total = len(predictions)

    strict_score = strict / total
    acc_score = correct / total
    for k in recall_at_k.keys():
        recall_at_k[k] /= total if total > 0 else 1.0

    pr = (micro_precision / total) if total > 0 else 1.0
    rec = (micro_recall / total) if total > 0 else 0.0

    # f1 = 2.0 * pr * rec / (pr + rec)

    return strict_score, acc_score, pr, recall_at_k, micro_f1


def fever_score(predictions,actual=None, max_evidence=5):
    correct = 0
    strict = 0

    macro_precision = 0
    macro_precision_hits = 0

    macro_recall = 0
    macro_recall_hits = 0

    for idx,instance in enumerate(predictions):
        assert 'predicted_evidence' in instance.keys(), 'evidence must be provided for the prediction'

        #If it's a blind test set, we need to copy in the values from the actual data
        if 'evidence' not in instance or 'label' not in instance:
            assert actual is not None, 'in blind evaluation mode, actual data must be provided'
            assert len(actual) == len(predictions), 'actual data and predicted data length must match'
            assert 'evidence' in actual[idx].keys(), 'evidence must be provided for the actual evidence'
            instance['evidence'] = actual[idx]['evidence']
            instance['label'] = actual[idx]['label']

        assert 'evidence' in instance.keys(), 'gold evidence must be provided'

        if is_correct_label(instance):
            correct += 1.0

            if is_strictly_correct(instance, max_evidence):
                strict+=1.0

        macro_prec = evidence_macro_precision(instance, max_evidence)
        macro_precision += macro_prec[0]
        macro_precision_hits += macro_prec[1]

        macro_rec = evidence_macro_recall(instance, max_evidence)
        macro_recall += macro_rec[0]
        macro_recall_hits += macro_rec[1]

    total = len(predictions)

    strict_score = strict / total
    acc_score = correct / total

    pr = (macro_precision / macro_precision_hits) if macro_precision_hits > 0 else 1.0
    rec = (macro_recall / macro_recall_hits) if macro_recall_hits > 0 else 0.0

    f1 = 2.0 * pr * rec / (pr + rec)

    return strict_score, acc_score, pr, rec, f1


