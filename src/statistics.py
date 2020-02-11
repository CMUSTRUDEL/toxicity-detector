from sklearn.metrics import roc_auc_score

def calculate_statistics(predicted,actual):
    """ Calculate statistics on how a predicted distribution diverges from actual
        Predicted is a 0-1 array, and so is actual
        len(predicted) = len(actual) """

    num_samples = len(predicted)

    accuracy = 0
    tp = fp = tn = fn = 0

    for i in range(num_samples):
        if predicted[i] == actual[i]:
            accuracy+=1
            if predicted[i] == 1:
                tp+=1
            else:
                tn+=1
        else:
            if predicted[i] == 1:
                fp+=1
            else:
                fn+=1

    accuracy/=num_samples
    precision = 1
    if(tp+fp!=0):
        precision = tp/(tp+fp)

    recall = 0
    if(tp+fn!=0):
        recall = tp/(tp+fn)

    if precision+recall != 0:
        f1 = 2*precision*recall/(precision+recall)
    else:
        f1 = 0

    f5 = 0
    if precision+recall != 0:
        f5 = (1+0.25) * precision*recall/(0.25*precision+recall)

    if tp+fn != 0 and tn+fp != 0:
        balanced_accuracy = ((tp/(tp+fn)) + (tn/(tn+fp))) /2

    try:
        auc = roc_auc_score(actual,predicted)
    except:
        auc = 0

    return {'precision':precision,'recall':recall,'accuracy':accuracy,'auc':auc,'f1':f1,'f_0.5':f5}
