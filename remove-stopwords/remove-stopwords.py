def remove_stopwords(tokens: list[str], stopwords: list[str]) -> list[str]:
    """Given a list of tokens and a list of stopwords, removes all tokens
    appearing in stopwords.

    Parameters
    ----------
    tokens: list[str]
        List of tokens to filter.
    stopwords: list[str]
        List of stopwords to remove from tokens.

    Returns
    -------
    list[str]
        Tokens with stopwords removed and order preserved.
    """
    return [tk for tk in tokens if tk not in stopwords]
