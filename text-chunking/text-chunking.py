def text_chunking(
    tokens: list[str],
    chunk_size: int,
    overlap: int,
) -> list[list[str]]:
    """Given a list of tokens, splits them into fixed-size chunks with
    optional overlap.

    Parameters
    ----------
    tokens: list[str]
        A list of tokens to be split into chunks.
    chunk_size: int
        The size of each chunk (number of tokens).
    overlap: int
        The number of tokens to overlap between consecutive chunks.

    Returns
    -------
    list[list[str]]
        A list of chunks, where each chunk is a list of tokens.
    """
    # [CHECK] Input validation:
    if chunk_size <= 0:
        raise ValueError("The parameter 'chunk_size' must be a positive integer.")
    if overlap < 0:
        raise ValueError("The parameter 'overlap' must be a non-negative integer.")
    if overlap >= chunk_size:
        raise ValueError(
            "The parameter 'overlap' must be less than 'chunk_size' to avoid "
            "infinite loops."
        )

    chunked_results = []
    # Calc step size btw chunk start positions:
    step = chunk_size - overlap
    for i in range(0, len(tokens), step):
        chunk = tokens[i : i + chunk_size]

        if len(chunk) == chunk_size or i == 0:
            chunked_results.append(chunk)

    return chunked_results
