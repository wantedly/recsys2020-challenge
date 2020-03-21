from sklearn.metrics import precision_recall_curve, auc, log_loss


def compute_prauc(pred, gt):
  prec, recall, thresh = precision_recall_curve(gt, pred)
  prauc = auc(recall, prec)
  return prauc


def calc_metrics(y_true, y_pred):
    """ Calculate metrics excluding unpredicted values, such as undersampling
    """
    not_pred_idx = y_pred != 0
    y_true = y_true[not_pred_idx]
    y_pred = y_pred[not_pred_idx]

    ce = log_loss(y_true, y_pred)
    prauc = compute_prauc(y_pred, y_true)

    result = {
        "ce": ce,
        "prauc": prauc
    }
    return result
