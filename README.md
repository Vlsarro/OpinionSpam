# OpinionSpam
Research code for opinion spam detection
## Dataset:
- 400 truthful positive reviews from TripAdvisor (described in [1])
- 400 truthful negative reviews from Expedia, Hotels.com, Orbitz, Priceline,TripAdvisor and Yelp (described in [2])

## Linguistic Approaches:
- word_embedding.py : using word-embedding method to detect opinion spam
- tf_idf.py : using TF-IDF method to detect opinion spam (PCA deduction is optional)
- unigram.py : using unigram method to detect opinion spam (PCA deduction is optional)
##References
[1] M. Ott, Y. Choi, C. Cardie, and J.T. Hancock. 2011. Finding Deceptive
Opinion Spam by Any Stretch of the Imagination. In Proceedings of the 49th
Annual Meeting of the Association for Computational Linguistics: Human Language
Technologies.

[2] M. Ott, C. Cardie, and J.T. Hancock. 2013. Negative Deceptive Opinion Spam.
In Proceedings of the 2013 Conference of the North American Chapter of the
Association for Computational Linguistics: Human Language Technologies.