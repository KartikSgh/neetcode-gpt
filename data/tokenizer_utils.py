from typing import List, Dict

class Solution:
    def tokenize(self, s, d):
        t = []
        while s!='':
            flag = True
            for i in range(len(s), 1, -1):
                if s[:i] in d:
                    flag = False
                    t.append(s[:i])
                    s = s[i:]
            if flag:
                t.append(s[0])
                s = s[1:]
        return t

    def tokenize_numbers(self, l: List[int], d: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.
        ans = []
        for s in l:
            ans.append(self.tokenize(str(s), d))
        return ans

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.
        c=len(self.tokenize(text, vocab))
        return c

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        c=len(self.tokenize(text, vocab))
        return round(c/len(text.split()), 4)
