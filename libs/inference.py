def trim_data(data, trim_labels=[], trim_percent = 0.5):
    data = list(data)

    trim_quantities = {}
    for label in trim_labels:
        trim_quantities[label] = int(len([1 for _, _label in data if _label == label]) * trim_percent)

    keep_data = []
    for index, _ in enumerate(data):
        data[index] = list(data[index])
        label = data[index][1]
        if label in trim_labels:
            if trim_quantities[label] == 0:
                keep_data.append(tuple([data[index][0], label]))
            else:
                trim_quantities[label] = trim_quantities[label] - 1
        if label not in trim_labels:
            keep_data.append(tuple([data[index][0], label]))

    return tuple(keep_data)