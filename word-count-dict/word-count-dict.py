def word_count_dict(sentences: list[list[str]]) -> dict[str, int]:
    """Compute word frequency dictionary given a list of tokenized
    sentences.

    Parameters
    ----------
    sentences: list[list[str]]

    Returns
    -------
    dict[str, int]
        Global word frequency across all sentences.
    """
    freq_dict = {}
    # Iterate over sentences + tokens:
    for sent in sentences:
        for w in sent:
            # Track frequency of token:
            if w in freq_dict:
                freq_dict[w] += 1
            else:
                freq_dict[w] = 1

    return freq_dict
