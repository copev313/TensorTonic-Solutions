from typing import List, Dict


class WordPieceTokenizer:
    """
    WordPiece tokenizer for BERT.
    """

    def __init__(
        self, 
        vocab: Dict[str, int], 
        unk_token: str = "[UNK]", 
        max_word_len: int = 100,
    ):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_word_len = max_word_len

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into WordPiece tokens."""
        tokens = []
        for word in text.lower().split():
            word_tokens = self._tokenize_word(word)
            tokens.extend(word_tokens)
        return tokens

    def _tokenize_word(self, word: str) -> List[str]:
        """Tokenize a single word into subwords.

        Split a single lowercased word into a list of subword tokens 
        using greedy longest-match-first against self.vocab.
        """
        # [CASE] Word exceeds maximum length:
        if len(word) > self.max_word_len:
            return [self.unk_token]

        subwords = []
        start = 0
        # Continue until the entire word is tokenized:
        while start < len(word):
            end = len(word)
            matched_subword = None

            # Find the longest subword in vocab that matches the current substring:
            while start < end:
                subword = word[start:end]
                # Add "##" prefix if not at the start of the word:
                if start > 0:
                    subword = "##" + subword

                # Check for a match in the vocabulary:
                if subword in self.vocab:
                    matched_subword = subword
                    break

                # If no match, shorten the substring by one character:
                end -= 1

            # [CASE] No matching subword found in vocab:
            if matched_subword is None:
                return [self.unk_token]

            # Add the matched subword to the list :
            subwords.append(matched_subword)
            # Move the start index forward:
            start = end

        return subwords