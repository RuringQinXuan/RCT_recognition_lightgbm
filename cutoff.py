

# cut_off = 0.5
# cut_off = 0.01
def get_y_pred_labels(y_preds,select_type):
    if select_type == 'sensitive':
        cut_off = 0.01
    elif select_type == 'balanced':
        cut_off = 0.5
    else:
        print('select_type should be sensitive or balanced')
    y_pred_labels = []
    for y_pred in y_preds:
        if y_pred > cut_off:
            y_label = 1
        else:
            y_label= 0
        y_pred_labels.append(y_label)
    return y_pred_labels
        