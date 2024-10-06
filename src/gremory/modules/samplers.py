import torch
from transformers.generation.logits_process import LogitsProcessor, LogitsWarper


# Code: https://github.com/oobabooga/text-generation-webui/blob/main/modules/sampler_hijack.py
class DRYLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        multiplier: float,
        base: float,
        allowed_length: int,
        sequence_breakers: list[int],
        _range: int,
    ):
        self.multiplier = multiplier
        self.base = base
        self.allowed_length = allowed_length
        self.sequence_breakers = sequence_breakers
        self._range = _range

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if self._range > 0:
            input_ids = input_ids[:, -self._range :]

        for input_ids_row, scores_row in zip(input_ids, scores):
            # Use normal Python data types for improved performance
            input_ids = input_ids_row.tolist()

            last_token = input_ids[-1]
            if last_token in self.sequence_breakers:
                continue

            # Exclude the last token as it always matches.
            match_indices = []
            for idx, val in enumerate(input_ids[:-1]):
                if val == last_token:
                    match_indices.append(idx)

            # Stores the maximum matching sequence length
            # for each token immediately following the sequence in the input.
            match_lengths = {}

            for i in match_indices:
                next_token = input_ids[i + 1]

                if next_token in self.sequence_breakers:
                    continue

                # We have already found that `last_token` matches at this index,
                # so the match is at least of length 1.
                match_length = 1

                # Extend the match backwards (at most to 50 to prevent exponent overflow at penalty calculation) (this cap also improves performance on worst case)
                while match_length < 50:
                    j = i - match_length
                    if j < 0:
                        # Start of input reached.
                        break

                    previous_token = input_ids[-(match_length + 1)]
                    if input_ids[j] != previous_token:
                        # Start of match reached.
                        break

                    if previous_token in self.sequence_breakers:
                        # Sequence-breaking token reached.
                        break

                    match_length += 1

                if next_token in match_lengths:
                    match_lengths[next_token] = max(
                        match_length, match_lengths[next_token]
                    )
                else:
                    match_lengths[next_token] = match_length

            # Apply penalties.
            for token, match_length in match_lengths.items():
                if match_length >= self.allowed_length:
                    penalty = self.multiplier * self.base ** (
                        match_length - self.allowed_length
                    )
                    scores_row[token] -= penalty

        return scores


# Code: https://github.com/oobabooga/text-generation-webui/blob/main/modules/sampler_hijack.py
class TailFreeLogitsWarper(LogitsWarper):
    def __init__(
        self,
        tfs: float,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
    ):
        tfs = float(tfs)
        if tfs < 0 or tfs > 1.0:
            raise ValueError(f"`tfs` has to be a float >= 0 and <= 1, but is {tfs}")
        self.tfs = tfs
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        sorted_logits, sorted_indices = torch.sort(scores, descending=True)
        probs = sorted_logits.softmax(dim=-1)

        # Compute second derivative normalized CDF
        d2 = probs.diff().diff().abs()
        normalized_d2 = d2 / d2.sum(dim=-1, keepdim=True)
        normalized_d2_cdf = normalized_d2.cumsum(dim=-1)

        # Remove tokens with CDF value above the threshold (token with 0 are kept)
        sorted_indices_to_remove = normalized_d2_cdf > self.tfs

        # Centre the distribution around the cutoff as in the original implementation of the algorithm
        sorted_indices_to_remove = torch.cat(
            (
                torch.zeros(scores.shape[0], 1, dtype=torch.bool, device=scores.device),
                sorted_indices_to_remove,
                torch.ones(scores.shape[0], 1, dtype=torch.bool, device=scores.device),
            ),
            dim=-1,
        )

        if self.min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep
            sorted_indices_to_remove[..., : self.min_tokens_to_keep] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


# Code: https://github.com/oobabooga/text-generation-webui/pull/6335/files
class XTCLogitsWarper(LogitsWarper):
    def __init__(
        self,
        threshold: float,
        probability: float,
        special_token_ids: set[int],
        filter_value: float = -float("Inf"),
    ):
        self.threshold = threshold
        self.probability = probability
        self.special_token_ids = special_token_ids
        self.filter_value = filter_value

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        # `random` returns values in the half-open range [0, 1), so setting `probability`
        # to 0 means the sampler never takes action, while setting it to 1 means the sampler
        # always takes action.
        #
        # Note that while XTC is most intuitively described as "if multiple tokens meet
        # the threshold, then with probability...", reversing the two conditions is logically
        # equivalent, and improves performance because processing can immediately be stopped
        # if the random check fails.
        if torch.rand(1).item() >= self.probability:
            return scores

        sorted_logits, sorted_indices = torch.sort(scores, descending=True)
        probs = sorted_logits.softmax(dim=-1)

        sorted_indices_to_remove = torch.full_like(probs, False, dtype=torch.bool)

        # This operation sets exactly those indices to `True` for which the next index has
        # probability above the threshold. Since `probs` is sorted, those are the indices
        # of all tokens that meet the threshold, *except* the least probable one.
        sorted_indices_to_remove[..., :-1] = probs[..., 1:] >= self.threshold

        # Convert sorted_indices_to_remove to the original indices
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )

        # If newline or EOS tokens would be removed, return the original scores
        if indices_to_remove[:, self.special_token_ids].any():
            return scores

        # Otherwise, remove tokens with the mask
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


class UnifiedLogitsWarper(LogitsWarper):
    def __init__(self, linear: float, quad: float, conf: float):
        self.linear = linear
        self.quad = quad
        self.conf = conf

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        # calculate entropy
        log_p = torch.nn.functional.log_softmax(scores, dim=-1)
        p = torch.exp(log_p)
        entropy = -(log_p * p).nansum(-1, keepdim=True)
        # Warning: This is using log probabilities as scores
        scores = log_p * (self.linear + entropy * self.conf) - log_p**2 * self.quad
        return scores
