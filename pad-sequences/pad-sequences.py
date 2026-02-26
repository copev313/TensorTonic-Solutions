import numpy as np


def pad_sequences(
    seqs: list[list[int]],
    pad_value: int = 0,
    max_len: int | None = None,
) -> np.ndarray:
    """Given a sequence of id tokens, pads them with a special value
    to match the length of the longest sequence.

    Parameters
    ----------
    seqs: list[list[int]]
        A list of sequences, where each sequence is a list of token ids.
    pad_value: int, optional (default=0)
        The value to use for padding shorter sequences.
    max_len: int | None, optional (default=None)
        The maximum length to pad the sequences to. If None, it will be
        determined by the longest sequence in `seqs`.

    Returns
    -------
    np.ndarray
        NumPy array with shape (N, L), where N = len(seqs) and
        L = max_len if provided else max(len(seq) for seq in seqs) or 0.
    """
    # [CASE] No max_len provided, determine it from the longest sequence:
    if max_len is None:
        max_len = max((len(seq) for seq in seqs), default=0)

    # Fill 2D array with pad_value, then copy sequences into it:
    padded_seqs_arr = np.full((len(seqs), max_len), pad_value, dtype=int)
    # Load seqs into corresponding row:
    for i, seq in enumerate(seqs):
        seq_length = min(len(seq), max_len)
        padded_seqs_arr[i, :seq_length] = seq[:seq_length]

    return padded_seqs_arr
