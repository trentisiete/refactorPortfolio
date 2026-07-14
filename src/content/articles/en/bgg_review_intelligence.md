---
title: "BGG Review Intelligence: From Reviews to Structured Opinion Knowledge"
topic: "Natural Language Processing"
date: 2025-11-15
excerpt: "BGG Review Intelligence explores how BoardGameGeek reviews can be transformed into reproducible sentiment predictions and structured opinion knowledge."
kind: "article"
draft: false
readingTime: 10
lang: "en"
---


## A multi-stage study of linguistic preprocessing, sentiment classification, contextual learning, and aspect-based analysis

Board-game reviews contain much more than a statement of whether a player liked or disliked a game. They describe mechanics, duration, complexity, artwork, balance, accessibility, player interaction, replayability, and the social conditions under which a game succeeds or fails. A single review may praise one of these aspects and reject another. It may contain irony, comparisons, conditional recommendations, or a change of opinion after repeated play. This combination of global judgment and detailed experience makes BoardGameGeek reviews a particularly rich domain for Natural Language Processing.

This project studies how that unstructured language can be transformed into reproducible and interpretable opinion knowledge. Its development followed a cumulative sequence. The first stage established a reliable linguistic and corpus-processing foundation. The second introduced sparse lexical representations, explicit opinion features, and classical machine-learning models. The third explored static word embeddings and neural sequence models. The fourth applied contextual transfer learning with BERT. The final stage moved beyond global sentiment labels and examined whether a large language model could extract structured aspects, polarities, and supporting evidence.

The result is not simply a collection of isolated exercises. It is a connected body of technical and empirical knowledge. Every stage was built on data structures, preprocessing decisions, and evaluation artifacts produced in the stages before it. The project therefore documents both model performance and the methodological path required to obtain, interpret, and compare that performance.

<div class="graph-flow" data-title="From review text to research evidence" data-note="Each stage inherits the data decisions and artifacts of the stages before it."><div data-step="Corpus">Acquisition, cleaning, language and annotation with traceability</div><div data-step="Sparse models">TF-IDF, opinion cues and controlled classical benchmarks</div><div data-step="Sequence">FastText, FNN and LSTM to test the value of word order</div><div data-step="Context">BERT fine-tuning with token meaning conditioned on the full review</div><div data-step="Explanation">Aspects, polarity and evidence grounded in review language</div></div>

*The research progression is cumulative: later models depend on the corpus, preprocessing decisions, and evaluation artifacts established earlier.*

## The problem behind the classification task

At document level, the initial objective is to classify a review as negative, neutral, or positive. The target labels are derived from the numeric rating associated with the review. Ratings from 0 to 3 are treated as negative, ratings from 4 to 7 as neutral, and ratings above 7 as positive.

```text
negative: rating <= 3
neutral:  4 <= rating <= 7
positive: rating > 7
```

This policy creates a practical supervised-learning task and produces three equally sized classes. Nevertheless, it is important to understand what the label represents. The model is learning to predict a rating-derived polarity category from the language of the review. It is not learning from a separate human annotation that directly evaluates the sentiment of the text.

That distinction matters because ratings and language do not always align perfectly. A user may assign a moderate rating while writing an emotionally negative review. Another user may describe serious problems but still recommend the game to a particular audience. A review can also contain positive and negative aspects at the same time. The broad neutral interval, which includes ratings 4, 5, 6, and 7, therefore contains several different linguistic situations: weak sentiment, mixed sentiment, conditional approval, uncertainty, or disagreement between the text and the rating.

The project consequently addresses a more interesting problem than ordinary polarity detection. It studies the relationship between textual evaluation and an external behavioral signal, while exposing the ambiguity introduced when a continuous or ordinal judgment is converted into three discrete classes.

## Building the linguistic and data foundation

Before comparing classifiers, the project established a consistent representation of every review. Raw user-generated text is noisy. It can contain markup, inconsistent spacing, malformed Unicode, repeated characters, emojis, spelling variation, domain-specific vocabulary, and fragments in different languages. If these elements are processed inconsistently, later differences between models may reflect differences in data preparation rather than differences in learning capacity.

The preprocessing architecture was designed as a sequence of independent transformations sharing a common contract. Each transformation receives a document, modifies only the fields related to its responsibility, and returns the enriched document. The structure can be summarized as follows:

```python
class PreprocStep(Protocol):
    name: str
    version: str
    params: dict

    def transform(self, doc: dict) -> dict:
        ...
```

The pipeline normalizes encoding, removes or converts markup, cleans whitespace and repeated characters, handles emojis, detects language, segments sentences, tokenizes words, filters stopwords, and records the resulting annotations. The later version uses spaCy as a unified linguistic processor for tokenization, lemmatization, part-of-speech tags, and dependency relations. This unification is methodologically important because syntactic dependencies, lemmas, and token positions must refer to the same tokenization.

Stopword processing was also adapted to the sentiment task. Common grammatical words can be removed from lexical representations, but negation terms such as “not,” “no,” and “never” must remain available because they can reverse the interpretation of an evaluative expression. Punctuation is preserved when it is needed for linguistic analysis and excluded from the final lexical representation when it would only add noise.

The processed documents are stored in a structured JSONL corpus accompanied by metadata, indexes, statistics, and schema information. Stable review identifiers make individual records retrievable, while the corpus reader supports both streaming and targeted access. Each document can preserve raw text, cleaned text, linguistic annotations, derived features, vector inputs, and information about the pipeline configuration that created it.

This corpus layer is one of the central contributions of the project. It separates data access from model logic and makes it possible to apply different representations to the same underlying documents. It also creates a traceable path from an experimental result back to the text, preprocessing configuration, and split from which that result was obtained.

## Corpus composition and experimental partitions

After filtering for English-language reviews with usable text and valid numeric ratings, the P2 corpus contains 526,869 reviews. The class-balancing procedure produces exactly 175,623 reviews in each polarity category.

The corpus is divided using a stratified procedure with a fixed random seed of 42. The training partition contains 368,808 reviews, the validation partition contains 52,687 reviews, and the held-out test partition contains 105,374 reviews. Class proportions remain effectively identical in every partition.

| Split | Total reviews | Negative | Neutral | Positive |
|---|---:|---:|---:|---:|
| Training | 368,808 | 122,936 | 122,936 | 122,936 |
| Validation | 52,687 | 17,563 | 17,562 | 17,562 |
| Test | 105,374 | 35,124 | 35,125 | 35,125 |
| **Complete corpus** | **526,869** | **175,623** | **175,623** | **175,623** |

The fixed validation partition is used for model and hyperparameter selection. The test partition remains separate until a final configuration has been selected. This distinction prevents the test set from becoming part of the optimization process and gives the final values a clear interpretation as held-out performance.

Because the corpus is perfectly balanced, accuracy is easier to interpret than it would be in a highly imbalanced dataset. Even so, macro-F1 remains the principal evaluation measure because it assigns the same importance to performance on the negative, neutral, and positive classes. Per-class precision, recall, F1, and confusion matrices are retained so that aggregate performance does not conceal systematic errors.

## Representing opinion with lexical and linguistic information

The classical modeling stage creates two complementary inputs from every processed review. The first, `tfidf_input`, contains a lemmatized version of the text without stopwords. It is used to construct TF-IDF vectors with word n-grams. The second, `opinion_input`, is a sparse dictionary of explicit linguistic information related to polarity.

The opinion representation captures phenomena such as positive and negative lexical items, negation, intensification, mitigation, and domain-related evaluative patterns. These features attempt to represent knowledge that a conventional bag of words does not encode explicitly. A phrase such as “not enjoyable,” for example, contains an interaction between a negation and a positive evaluative term. A linguistic representation can record that interaction directly, while a lexical representation must infer it from the observed n-gram.

The `CombinedVectorizer` allows the same experimental pipeline to operate in three modes. TF-IDF mode uses only lexical n-grams, opinion mode uses only the explicit feature dictionary, and combined mode concatenates both sparse matrices.

```python
vectorizer = CombinedVectorizer(
    mode="combined",
    tfidf_params={
        "ngram_range": (1, 2),
        "max_features": 20_000,
        "min_df": 10,
        "sublinear_tf": True,
    },
)

X_train = vectorizer.fit_transform(train_documents)
X_validation = vectorizer.transform(validation_documents)
X_test = vectorizer.transform(test_documents)
```

The vectorizer is fitted only on the training documents. Validation and test documents are transformed using the vocabulary and inverse-document-frequency values learned from training data. This prevents lexical information from the held-out partitions from influencing the representation.

TF-IDF remains a powerful representation for this domain. Unigrams capture individual evaluative terms, while bigrams and longer n-grams capture short contextual patterns such as negation, recommendations, or common expressions of dissatisfaction. Sublinear term frequency reduces the influence of repeated words, and `min_df` removes extremely rare features that may represent noise rather than stable evidence.

The explicit opinion representation serves a different purpose. Its dimensions are fewer and more interpretable, but it inevitably simplifies the language through manually designed categories and lexical resources. The comparison between TF-IDF, opinion features, and their combination therefore asks whether structured linguistic knowledge provides information beyond a large lexical space.

## Classical machine-learning experiments

Four model families were evaluated on the three representation modes. Naive Bayes provides an efficient probabilistic baseline for sparse text. A linear Support Vector Machine learns maximum-margin decision boundaries in the high-dimensional lexical space. Random Forest introduces nonlinear interactions through an ensemble of decision trees. XGBoost uses sequentially constructed, regularized trees to model more complex feature relationships.

Candidate configurations are trained on the training split and ranked by validation macro-F1. Once the strongest configuration in a model family has been identified, it is retrained using the combined training and validation data and evaluated once on the test set. Predictions, per-class reports, confusion matrices, vectorizers, models, and summary artifacts are saved with the experiment.

The exact test values below were verified against the stored evaluation reports.

| Model | Winning representation | Accuracy | Macro-F1 | Negative F1 | Neutral F1 | Positive F1 |
|---|---|---:|---:|---:|---:|---:|
| Linear SVM | TF-IDF | **0.7333** | **0.7309** | **0.8040** | **0.6291** | **0.7595** |
| Multinomial Naive Bayes | TF-IDF | 0.7277 | 0.7254 | 0.7974 | 0.6237 | 0.7552 |
| XGBoost | Combined | 0.6904 | 0.6871 | 0.7537 | 0.5894 | 0.7182 |
| Random Forest | TF-IDF | 0.6439 | 0.6346 | 0.7169 | 0.5085 | 0.6785 |

The strongest classical configuration is a linear SVM operating on TF-IDF vectors composed of unigrams and bigrams. Its vocabulary is limited to 20,000 features, terms must occur in at least ten documents, sublinear term frequency is enabled, and the SVM regularization parameter is `C=1.0`.

```python
tfidf_params = {
    "ngram_range": (1, 2),
    "max_features": 20_000,
    "min_df": 10,
    "sublinear_tf": True,
}

model = LinearSVC(C=1.0, class_weight=None)
```

The result demonstrates the strength of sparse linear modeling for large text collections. The SVM is only slightly ahead of Multinomial Naive Bayes, with a difference of approximately 0.0055 macro-F1. Both linear models outperform the tree-based families. In a high-dimensional sparse space, a linear boundary can exploit the accumulated evidence from many lexical dimensions efficiently. Tree ensembles must partition that space through successive decisions and can struggle to represent the same geometry without considerably greater complexity.

XGBoost obtains its strongest verified result with the combined representation, indicating that the explicit opinion features provide useful high-level information for boosted trees. Nevertheless, its final macro-F1 remains below the two linear models. Random Forest produces the weakest result and shows a particularly large decline on the neutral class.

The class-specific scores reveal a pattern that aggregate accuracy alone cannot express. Negative reviews are consistently the easiest to identify. Positive reviews also achieve relatively strong results. Neutral reviews remain substantially more difficult for every model. Even the winning SVM obtains neutral F1 of 0.6291, compared with 0.8040 for negative and 0.7595 for positive reviews.

This is not merely a consequence of class imbalance because the dataset is balanced. The difficulty belongs to the semantic structure of the labels and the language. Reviews near the middle of the rating scale may include weak opinions, contradictory aspects, conditional recommendations, or language that resembles one of the extreme classes even when the overall rating is moderate.

## Static embeddings and neural architectures

The next stage examines whether dense pretrained word representations can capture semantic relations that sparse lexical features treat as independent dimensions. Word2Vec GoogleNews, FastText crawl embeddings, and GloVe Dolma embeddings were compared using vectors of 300 dimensions.

Two architectures were used to expose the consequences of document representation. The feedforward network receives one vector per review. That vector is obtained by averaging the embeddings of all in-vocabulary words in the document.

```text
document_vector = (1 / n) * sum(word_vector_i)
```

Mean pooling is efficient and produces a fixed-size semantic representation regardless of document length. Its limitation is equally clear: it removes word order and assigns the same structural importance to every included token. “The game is good but too long” and a rearrangement of the same words can produce nearly identical averages even though discourse order and contrast affect interpretation.

The LSTM receives the sequence of word embeddings instead of their average. Reviews are padded or truncated to a fixed maximum length, and the recurrent network processes the embeddings in order. This enables the model to retain local relationships, changes of tone, negation, and other sequential information that disappears during mean pooling.

FastText provided the strongest base representation in the reported comparisons. Its use of character subword information gives it an advantage with rare words, morphological variants, spelling differences, and specialized board-game vocabulary. The final FastText models used a feedforward network with a hidden dimension of 512 and an LSTM with a hidden dimension of 128.

| Model | Representation | Accuracy | Macro-F1 | Negative F1 | Neutral F1 | Positive F1 |
|---|---|---:|---:|---:|---:|---:|
| FastText + FNN | Mean-pooled document embedding | 0.697 | 0.695 | 0.770 | 0.590 | 0.726 |
| FastText + LSTM | Ordered word-embedding sequence | **0.752** | **0.752** | **0.823** | **0.657** | **0.776** |

The LSTM improves macro-F1 by approximately 0.057 and produces gains in all three classes. Neutral F1 rises from 0.590 to 0.657. This comparison provides direct evidence that word order and sequence composition carry information that is lost when static word vectors are reduced to an average.

The result does not make the earlier sparse models obsolete. The linear SVM remains computationally efficient, reproducible, and highly competitive. Instead, the neural comparison clarifies what additional information a sequential representation can recover and how that information affects the class that is most difficult for simpler models.

## Contextual transfer learning with BERT

Static embeddings assign one fixed vector to a word regardless of its use. BERT replaces that assumption with contextual representations. The vector associated with a token depends on the surrounding sequence, allowing the model to distinguish meanings and functions that a fixed embedding merges together.

The project fine-tunes `bert-base-uncased` with a three-class sequence-classification head. Raw review text is processed with WordPiece tokenization. Special tokens are added, sequences are padded or truncated to 128 tokens, and an attention mask distinguishes real tokens from padding. Both the classification layer and the pretrained transformer are updated during fine-tuning.

```python
MODEL_NAME = "bert-base-uncased"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 2
LEARNING_RATE = 2e-5

model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
)
```

The BERT experiment used a separate balanced corpus of approximately 60,000 reviews because of the computational constraints of the Colab training environment. It also used raw text rather than the lemmatized, stopword-filtered input used by several earlier models. These differences are essential when interpreting the scores.

After the first epoch, the model reached validation accuracy of 0.8327 and validation macro-F1 of 0.8304. After the second epoch, validation accuracy reached 0.8464 and macro-F1 reached 0.8468. The reported held-out results were accuracy 0.8431, macro-F1 0.8432, weighted-F1 0.8432, and evaluation loss 0.4420.

| BERT evaluation measure | Value |
|---|---:|
| Test accuracy | 0.8431 |
| Test macro-F1 | 0.8432 |
| Test weighted-F1 | 0.8432 |
| Test evaluation loss | 0.4420 |

These results provide strong evidence that contextual transfer learning is highly suitable for the domain. BERT can relate every token to its surrounding context and preserve function words, punctuation, full word forms, and discourse signals that are reduced or removed in the classical pipeline.

The BERT score is not presented as a direct numerical victory over the full-corpus SVM or LSTM because the models were not evaluated on the same benchmark. The result establishes the value of the contextual approach within its own documented setting. It also shows how pretrained linguistic knowledge can achieve high performance after only two fine-tuning epochs on a corpus much smaller than the complete P2 collection.

<div class="graph-duo"><div class="graph-hbar" data-title="Classical models" data-subtitle="Macro-F1 on the shared balanced P2 test set" data-note="Controlled comparison: same corpus partitions and target labels." data-max="0.9" data-decimals="4"><div data-label="Linear SVM" data-value="0.7309"></div><div data-label="Naive Bayes" data-value="0.7254"></div><div data-label="XGBoost" data-value="0.6871"></div><div data-label="Random forest" data-value="0.6346"></div></div><div class="graph-hbar" data-title="Neural and contextual models" data-subtitle="Macro-F1 in the reported P3 settings" data-note="BERT used a separate ~60k-review corpus and raw text; FNN and LSTM share a static-embedding setting." data-max="0.9" data-decimals="4"><div data-label="BERT" data-value="0.8432"></div><div data-label="FastText LSTM" data-value="0.7520"></div><div data-label="FastText FNN" data-value="0.6950"></div></div></div>

*The classical models share a controlled test set. The neural and contextual results preserve the settings reported in the P3 record, so the panels are intentionally separated rather than presented as one league table.*

## Moving from global polarity to aspect-level understanding

Document classification compresses an entire review into one label. That answer is useful for large-scale measurement, but it cannot explain which parts of the game produced the judgment. The last stage of the project therefore changes the output structure rather than simply replacing one classifier with another.

The aspect-based experiment uses a large language model to analyze twenty selected RoboRally reviews. The reviews are loaded through the existing corpus reader and inserted into a structured prompt. Instead of requesting an unrestricted summary, the prompt defines a JSON structure containing the game name, an aggregate conclusion, a list of aspects, a polarity for each aspect, and literal supporting evidence.

```json
{
  "juego": "Game name",
  "conclusion": "Aggregate sentiment",
  "aspectos": [
    {
      "aspecto": "Duration / Mechanics / Art / ...",
      "polaridad": "Positive / Negative / Mixed",
      "evidencia_i": "Exact supporting review text"
    }
  ]
}
```

The generated analysis characterized the selected reviews as overwhelmingly negative and organized their criticism into coherent categories. The extracted aspects included duration, randomness and chaos, player elimination, accessibility, interaction, balance, player agency, and general experience. Evidence strings were associated with the conclusions so that the structured output remained connected to the source language.

This experiment demonstrates a different type of capability from the sentiment classifiers. BERT, LSTM, and SVM estimate a global class with measurable predictive performance. The prompted language model decomposes opinions into interpretable units and explains why the selected reviewers reacted negatively.

The aspect result is qualitative rather than a benchmark score. The twenty reviews belong to a deliberately selected negative sequence, and no independently annotated aspect dataset was used. The experiment therefore demonstrates feasibility and expressive value, not measured aspect-extraction accuracy. Its importance within the project is conceptual: it connects the scalable corpus infrastructure to a representation of opinion that more closely reflects the internal structure of a review.

## What the combined evidence shows

The complete sequence of experiments reveals several consistent findings. The first is that careful data and linguistic design remain important even when sophisticated models are available. Stable identifiers, deterministic splits, traceable preprocessing, saved predictions, and per-class evaluation are what make model results interpretable. Without this foundation, a higher score cannot be separated from differences in data preparation or evaluation conditions.

The second finding is the continuing strength of lexical baselines. TF-IDF with a linear SVM reaches macro-F1 of 0.7309 on more than one hundred thousand held-out reviews. Multinomial Naive Bayes performs only slightly below it. These results show that a large proportion of sentiment information in the corpus is expressed through recurring lexical patterns that can be captured without a deep architecture.

The third finding is that order and context add measurable information. The LSTM improves substantially over a mean-pooled feedforward model using the same FastText foundation. The BERT experiment reaches an even higher score within its separate setting, showing the value of contextualized representations and transfer learning.

The fourth and most persistent finding is the difficulty of neutrality. This pattern appears in classical models and neural networks. The negative and positive classes have clearer lexical boundaries, while the neutral class absorbs mixed evaluations, weak sentiment, boundary ratings, and text-rating disagreement. Neutrality in this corpus is therefore not simply an intermediate emotional state. It is a heterogeneous region created by language, user behavior, and the chosen label construction.

The fifth finding is that explicit linguistic opinion features have conditional rather than universal value. They add interpretable information about negation, intensification, and domain expressions, but high-dimensional lexical models already encode many of the same patterns through n-grams. Their effect depends on the model family and representation geometry. XGBoost obtains its strongest verified configuration from the combined input, while the strongest SVM and Naive Bayes models use TF-IDF alone.

Finally, the project establishes that prediction and explanation must be treated as different outputs. A classifier can estimate global polarity efficiently over hundreds of thousands of documents. An aspect-based system can expose the topics and evidence underlying that polarity. Together, these levels provide a more complete representation of opinion than either one alone.

## What we learned, what remains uncertain, and where the work leads

The most important lesson from this process is that the quality of an NLP result cannot be separated from the structure of the corpus that produced it. At the beginning, it was tempting to think of preprocessing as a preliminary task that had to be completed before the “real” modeling could begin. The later experiments changed that view. Decisions about language filtering, tokenization, lemmatization, stopwords, punctuation, label thresholds, and document splits define what the models are able to observe and what their scores mean. The corpus is not a passive input to the research; it is part of the research object itself.

We also learned that complexity does not invalidate simplicity. The linear SVM and Multinomial Naive Bayes remain strong even after the project introduces recurrent and contextual architectures. Their performance shows that repeated lexical patterns contain a large amount of information about rating-derived polarity. This provides an important reference point for interpreting every more complex model. A deeper architecture is scientifically interesting when it captures information that the baseline cannot represent, not merely because it is newer.

The comparison between the feedforward network and the LSTM made this distinction visible. Both models started from static FastText embeddings, but only the LSTM preserved the order of the words. Its improvement, especially on the neutral class, suggests that some of the missing information is compositional. Negation, contrast, qualification, and changes of tone cannot be reduced completely to an average of word meanings. BERT develops the same idea further by allowing the representation of each token to depend on its complete context.

At the same time, the BERT result created one of the main doubts that remains in the project. Its reported macro-F1 is the highest, but it was obtained on a smaller, separately prepared corpus and with raw text rather than the processed lexical input used by several earlier models. We can say that contextual transfer learning worked very well in that setting. We cannot yet say exactly how much of the difference comes from the architecture, the corpus subset, the input text, or the training conditions. This uncertainty is not a weakness to hide; it is part of what the experimental record has taught us about fair model comparison.

The natural research path that follows from this observation begins with comparability. The existing experiments have already identified the models worth comparing. What is now missing is one frozen benchmark on which the strongest sparse model, the recurrent model, and the contextual model observe the same reviews and are evaluated through the same procedure. Such a comparison would transform the current progression from a set of strong stage-specific results into a controlled account of what each representation contributes.

The neutral class raises a deeper doubt. Its difficulty persists across algorithms that make very different assumptions. This suggests that the problem does not belong only to the classifiers. It may also belong to the definition of neutrality itself. Ratings 4 through 7 cover a wide interval, and the associated reviews may express disappointment, moderate approval, mixed aspects, uncertainty, or conditional recommendation. Calling all of these cases “neutral” creates a convenient category, but perhaps not a single linguistic phenomenon.

This leads to a question that the current automatic labels cannot answer: when a model disagrees with the rating-derived class, is the model wrong, or has it identified sentiment in the text that the rating threshold does not represent? The answer would require reading and independently annotating a carefully selected group of reviews, particularly those near rating boundaries and those on which models disagree. Human disagreement would also be informative. If readers do not consistently assign one sentiment to these reviews, uncertainty should be treated as part of the data rather than as an error that a classifier is expected to eliminate.

The aspect-based experiment makes this question even more significant. Once a review is decomposed into duration, interaction, accessibility, balance, or player agency, mixed sentiment becomes easier to understand. A review does not need to be globally neutral in the ordinary sense; it may be strongly positive about one aspect and strongly negative about another. The global rating then represents an aggregation made by the reviewer, while an aspect model attempts to recover the components of that aggregation.

This changed our understanding of the final objective. At first, aspect extraction appeared to be an explanatory layer placed after classification. The experiment suggests that it may also offer a better representation of the problem itself. Global sentiment and aspect sentiment are not competing outputs. The global label provides scale and comparability, while the aspects preserve the internal structure of the opinion. The relation between them may be more informative than either result considered alone.

The LLM experiment also leaves an important methodological doubt. Its output is coherent and supported by recognizable phrases from the selected reviews, but coherence is not the same as measured faithfulness. A generative model can organize language convincingly even when an aspect is incomplete, an evidence phrase is altered, or a conclusion is stronger than the source permits. The experiment therefore taught us to separate readability from validity. Structured JSON and quoted evidence make the result inspectable, but a real evaluation depends on comparison with human-annotated aspects and exact source spans.

The roadmap that emerges from these lessons is not a sequence of software additions. It is a sequence of questions. A common benchmark clarifies the contribution of each representation. Human reading of difficult reviews clarifies what neutrality means. Aspect annotation clarifies how global judgments are composed. Evidence-based evaluation clarifies whether generative explanations are faithful. Only after those relationships are understood does it become meaningful to study how well the conclusions transfer to other games, time periods, languages, or review domains.

There is also uncertainty about what the models learn from the identity of the domain. A random review-level split may allow similar vocabulary from the same games or the same types of users to appear in training and test data. Strong performance can therefore reflect genuine sentiment understanding, familiarity with particular game vocabularies, or a combination of both. The current results describe generalization to held-out reviews from the same broad corpus. They do not yet describe generalization to entirely unseen games, new communities, or later periods.

These doubts do not diminish the project. They are evidence that the work has reached a stage at which the interesting questions are no longer limited to choosing an algorithm. The experiments have exposed relationships among ratings, language, context, aspects, and evidence. They have also shown where numerical performance is sufficient and where it leaves the phenomenon underexplained.

The boundaries of the evidence remain clear. The classical models share one large, balanced corpus and a common test set, so their comparison is controlled. The FNN and LSTM share the static-embedding setting, making their architectural comparison meaningful. BERT belongs to a different dataset and input setting. The aspect analysis is a qualitative demonstration based on a small, deliberately negative group of reviews. Each result is valuable within its own conditions, and keeping those conditions visible is what allows the combined project to serve as a trustworthy base of knowledge.

## Conclusion

This project constructs a continuous path from raw user reviews to structured opinion intelligence. It begins by treating corpus design and linguistic consistency as part of the scientific problem. It then tests whether sentiment is best represented through lexical statistics, explicit linguistic knowledge, dense semantic vectors, sequential processing, or contextual transfer learning. Finally, it extends the task from global classification to aspect-level interpretation grounded in review evidence.

The classical experiments establish a strong and efficient baseline: a linear SVM with TF-IDF achieves macro-F1 of 0.7309 on a balanced test set of 105,374 reviews. The static-embedding experiments demonstrate the value of word order, with FastText and an LSTM reaching macro-F1 of 0.752 in the reported large-corpus setting. The BERT experiment shows the potential of contextual transfer learning, reaching macro-F1 of 0.8432 on its separate balanced corpus. The LLM experiment demonstrates how the same data foundation can support structured explanations of duration, randomness, accessibility, interaction, balance, and player agency.

More important than any individual score is the knowledge created by the progression itself. Sparse lexical models show how much can be learned from recurring language. Linguistic features reveal which evaluative structures can be made explicit. Recurrent models demonstrate the role of sequence. BERT captures context through pretrained attention. Aspect extraction restores the multidimensional nature of opinion that a global label necessarily compresses.

Taken together, these stages form a substantial foundation for understanding opinion in domain-specific review text. The project does not treat NLP as the isolated selection of a model. It treats data provenance, linguistic structure, representation, evaluation, interpretation, and evidence as connected parts of the same research object.
