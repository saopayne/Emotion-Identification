# Emotion & Sentiment Analysis

It is an emotion identification project that tries to identify emotions from text namely : anger, disgust,
fear, sadness, joy, surprise.

Currently fine grained evaluations are being done i.e. each of the emotion classes are further subdivided
into high or low, eg. high anger / low anger.
Initially each of the sentence is annotated with a continious value [0,100] for each emotion.
Fined grained classification treats [0,50) as low and [50,100] as high.
Will be enhanced for coarse grained evaluation.

For each emotion, false positives are currently high (will be improved).
