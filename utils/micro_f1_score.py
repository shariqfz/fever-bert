def micro_f1(predictions):
  class_map = {"REFUTES": 0, "SUPPORTS": 1, "NOT ENOUGH INFO": 2}

  confusion_matrix = {"REFUTES": {"TP": 0, "FP": 0, "FN": 0},
                      "SUPPORTS": {"TP": 0, "FP": 0, "FN": 0},
                      "NOT ENOUGH INFO": {"TP": 0, "FP": 0, "FN": 0}}
                      
  for idx,instance in enumerate(predictions):
    pred_label = instance["predicted_label"].upper()
    true_label = instance["label"].upper()

    if true_label == pred_label :
        confusion_matrix[true_label]["TP"] += 1
    else:
        confusion_matrix[true_label]["FN"] += 1
        confusion_matrix[pred_label]["FP"] += 1

  true_positives = sum([confusion_matrix[class_]["TP"] for class_ in class_map.keys() ])
  false_positives = sum([confusion_matrix[class_]["FP"] for class_ in class_map.keys() ])
  false_negatives = sum([confusion_matrix[class_]["FN"] for class_ in class_map.keys() ])

  # micro f1 = true_positives / (true_positives +  (1/2)*(false_positives + false_negatives))
  # reference: https://stephenallwright.com/micro-vs-macro-f1-score/
  
  denominator = true_positives +  (false_positives + false_negatives) / 2
  micro_f1 = true_positives / denominator if denominator > 0 else 1.0         

return micro_f1