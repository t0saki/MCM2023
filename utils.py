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