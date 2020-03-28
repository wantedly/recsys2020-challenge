from sklearn.metrics import precision_recall_curve, auc, log_loss


def compute_prauc(pred, gt):
    """ https://gist.github.com/alykhantejani/497889c55bdd28be1c3cbc08bcb8fd46#file-compute_metrics-py
    """
    prec, recall, thresh = precision_recall_curve(gt, pred)
    prauc = auc(recall, prec)
    return prauc


def calculate_ctr(gt):
    """ https://gist.github.com/alykhantejani/497889c55bdd28be1c3cbc08bcb8fd46#file-compute_metrics-py
    """
    positive = len([x for x in gt if x == 1])
    ctr = positive/float(len(gt))
    return ctr


def compute_rce(pred, gt):
    """ https://gist.github.com/alykhantejani/497889c55bdd28be1c3cbc08bcb8fd46#file-compute_metrics-py
    """
    cross_entropy = log_loss(gt, pred)
    data_ctr = calculate_ctr(gt)
    strawman_cross_entropy = log_loss(gt, [data_ctr for _ in range(len(gt))])
    return (1.0 - cross_entropy/strawman_cross_entropy)*100.0


def calc_metrics(y_true, y_pred):
    """ Calculate metrics excluding unpredicted values
    """
    not_pred_idx = y_pred != 0
    y_true = y_true[not_pred_idx]
    y_pred = y_pred[not_pred_idx]

    ce = log_loss(y_true, y_pred)
    rce = compute_rce(y_pred, y_true)
    prauc = compute_prauc(y_pred, y_true)

    result = {
        "ce": ce,
        "rce": rce,
        "prauc": prauc
    }
    return result
