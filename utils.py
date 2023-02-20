def resize_word_list(word_list, max_len=5):
    """Resize a list of words into a fixed length, truncating or padding as necessary"""
    resized_list = []
    for word in word_list:
        # Truncate or pad word to fixed length
        if len(word) > max_len:
            resized_word = word[:max_len]
        else:
            resized_word = word + '0'*(max_len-len(word))
        resized_list.append(resized_word)
    return resized_list

def NormalizeData(data, input_columns, target_columns):
    means = []
    stds = []
    for column in input_columns:
        means.append(data[column].mean())
        std = data[column].std()
        if std == 0:
            std = 1
        stds.append(std)
    for column in target_columns:
        means.append(data[column].mean())
        std = data[column].std()
        if std == 0:
            std = 1
        stds.append(std)
    data[input_columns] = (data[input_columns] - means[:len(input_columns)]) / stds[:len(input_columns)]
    data[target_columns] = (data[target_columns] - means[len(input_columns):]) / stds[len(input_columns):]
    return data, means, stds

def NormalizeDataCustom(data, input_columns, target_columns, means, stds):
    data[input_columns] = (data[input_columns] - means[:len(input_columns)]) / stds[:len(input_columns)]
    data[target_columns] = (data[target_columns] - means[len(input_columns):]) / stds[len(input_columns):]
    return data

def NormalizeDataWithMeansStds(data, input_columns, target_columns, means, stds):
    data[input_columns] = (data[input_columns] - means[:len(input_columns)]) / stds[:len(input_columns)]
    data[target_columns] = (data[target_columns] - means[len(input_columns):]) / stds[len(input_columns):]
    return data

def DenormalizeData(column, mean, std):
    return column * std + mean