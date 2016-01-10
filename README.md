# Emotion Identification

It is an emotion identification project that tries to identify emotions from text namely : anger, disgust,
fear, sadness, joy, surprise. 

Some notable works had been done on classifying sentences based on emotions. But this one is ambitious. It assumes that a sentence can
be a mix of emotions.


Currently fine grained evaluations are being done i.e. each of the emotion classes are further subdivided
into high or low, eg. high anger / low anger.

Initially each of the sentence is annotated with a continious value [0,100] for each emotion.

A sentence can be therefore 30% anger, 20% disgust,10% fear and so on. 

Fined grained classification treats [0,50) as low and [50,100] as high.

Will be enhanced for coarse grained evaluation where it may be possible to tell the quantity of emotion.

For each emotion, false positives are currently high (will be improved).

# Rough desciption of working : 

Every sentence is passed through a POS-tagger.

Only adjectives are taken as features.

TF-IDF scores are applied for each feature and occurence matrix is created.

The occurence matrix is then trained using softmax regression of Tensorflow.

Sentences are classified with fine-grained evaluation.

#Discussion : 
Lots of false positives. Feature selection needs to be improved. Dimensionality needs to be reduced.

True positive is high. 

Coarse grained classification is to be done.
