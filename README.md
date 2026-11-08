here's what I learnt

The model reads text as a list of numbers, one vector per word piece.
Those vectors pass through 30 identical rounds. Each round has two steps. In the first, a word looks back at earlier words and borrows information from the relevant ones (attention, which is how “Paris” can depend on “France”). In the second, it thinks on its own about what it now holds (MLP).
Each round adds its result onto the same running vector, like everyone editing one shared whiteboard (the residual).
At the end, the final vector is compared against every word in the vocabulary. The best match is the guess for the next word.
That guess is appended to the text, and the whole thing runs again. This is the entire loop, and the KV cache just remembers earlier work so it isn’t redone.