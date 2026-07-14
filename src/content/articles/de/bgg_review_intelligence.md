---
title: "BGG Review Intelligence: Von Rezensionen zu strukturiertem Meinungswissen"
date: 2025-11-15
excerpt: "BGG Review Intelligence untersucht, wie sich BoardGameGeek-Rezensionen in reproduzierbare Sentiment-Vorhersagen und strukturiertes Meinungswissen verwandeln lassen."
kind: "article"
draft: false
readingTime: 10
lang: "de"
translated: true
sourceHash: "2a1f03b5cc521ca4"
---

## Eine mehrstufige Studie zu linguistischer Vorverarbeitung, Sentiment-Klassifikation, kontextuellem Lernen und aspektbasierter Analyse

Rezensionen zu Brettspielen enthalten weit mehr als die Aussage, ob einem Spieler ein Spiel gefallen hat oder nicht. Sie beschreiben Mechaniken, Spieldauer, Komplexität, Artwork, Balance, Zugänglichkeit, Spielerinteraktion, Wiederspielwert und die sozialen Bedingungen, unter denen ein Spiel gelingt oder scheitert. Eine einzelne Rezension kann einen dieser Aspekte loben und einen anderen ablehnen. Sie kann Ironie, Vergleiche, bedingte Empfehlungen oder einen Meinungswechsel nach wiederholtem Spielen enthalten. Diese Kombination aus globalem Urteil und detaillierter Erfahrung macht BoardGameGeek-Rezensionen zu einer besonders ergiebigen Domäne für Natural Language Processing.

Dieses Projekt untersucht, wie sich diese unstrukturierte Sprache in reproduzierbares und interpretierbares Meinungswissen überführen lässt. Seine Entwicklung folgte einer kumulativen Abfolge. Die erste Phase legte eine verlässliche linguistische und korpusverarbeitende Grundlage. Die zweite führte spärliche lexikalische Repräsentationen, explizite Meinungsmerkmale und klassische Machine-Learning-Modelle ein. Die dritte untersuchte statische Word Embeddings und neuronale Sequenzmodelle. Die vierte wandte kontextuelles Transfer-Learning mit BERT an. Die letzte Phase ging über globale Sentiment-Labels hinaus und untersuchte, ob ein großes Sprachmodell strukturierte Aspekte, Polaritäten und stützende Belege extrahieren kann.

Das Ergebnis ist nicht einfach eine Sammlung isolierter Übungen. Es ist ein zusammenhängender Korpus aus technischem und empirischem Wissen. Jede Phase baute auf Datenstrukturen, Vorverarbeitungsentscheidungen und Evaluationsartefakten auf, die in den vorangegangenen Phasen entstanden waren. Das Projekt dokumentiert daher sowohl die Modellleistung als auch den methodischen Weg, der nötig war, um diese Leistung zu erhalten, zu interpretieren und zu vergleichen.

![Forschungsverlauf von der Korpuserstellung bis zur strukturierten Meinungserklärung](/assets/articles/bgg_review_intelligence/research-progression.svg)

*Der Forschungsverlauf ist kumulativ: spätere Modelle hängen vom Korpus, den Vorverarbeitungsentscheidungen und den Evaluationsartefakten ab, die zuvor festgelegt wurden.*

## Das Problem hinter der Klassifikationsaufgabe

Auf Dokumentebene besteht das anfängliche Ziel darin, eine Rezension als negativ, neutral oder positiv zu klassifizieren. Die Ziel-Labels werden aus der numerischen Bewertung abgeleitet, die der Rezension zugeordnet ist. Bewertungen von 0 bis 3 gelten als negativ, Bewertungen von 4 bis 7 als neutral und Bewertungen über 7 als positiv.

```text
negative: rating <= 3
neutral:  4 <= rating <= 7
positive: rating > 7
```

Diese Regel schafft eine praktikable überwachte Lernaufgabe und erzeugt drei gleich große Klassen. Dennoch ist es wichtig zu verstehen, wofür das Label eigentlich steht. Das Modell lernt, eine aus der Bewertung abgeleitete Polaritätskategorie aus der Sprache der Rezension vorherzusagen. Es lernt nicht aus einer separaten menschlichen Annotation, die das Sentiment des Textes direkt bewertet.

Diese Unterscheidung ist wichtig, weil Bewertungen und Sprache nicht immer perfekt übereinstimmen. Ein Nutzer kann eine moderate Bewertung vergeben und dabei eine emotional negative Rezension schreiben. Ein anderer Nutzer kann ernsthafte Probleme beschreiben und das Spiel dennoch einem bestimmten Publikum empfehlen. Eine Rezension kann auch gleichzeitig positive und negative Aspekte enthalten. Das breite neutrale Intervall, das die Bewertungen 4, 5, 6 und 7 umfasst, enthält daher mehrere unterschiedliche sprachliche Situationen: schwaches Sentiment, gemischtes Sentiment, bedingte Zustimmung, Unsicherheit oder eine Diskrepanz zwischen Text und Bewertung.

Das Projekt behandelt folglich ein interessanteres Problem als gewöhnliche Polaritätserkennung. Es untersucht die Beziehung zwischen textlicher Bewertung und einem externen Verhaltenssignal und legt dabei die Mehrdeutigkeit offen, die entsteht, wenn ein kontinuierliches oder ordinales Urteil in drei diskrete Klassen umgewandelt wird.

## Aufbau der linguistischen und Daten-Grundlage

Bevor Klassifikatoren verglichen wurden, etablierte das Projekt eine konsistente Repräsentation jeder Rezension. Roher, nutzergenerierter Text ist verrauscht. Er kann Markup, uneinheitliche Abstände, fehlerhaftes Unicode, wiederholte Zeichen, Emojis, Rechtschreibvarianten, domänenspezifisches Vokabular und Fragmente in verschiedenen Sprachen enthalten. Werden diese Elemente uneinheitlich verarbeitet, können spätere Unterschiede zwischen Modellen eher Unterschiede in der Datenaufbereitung widerspiegeln als Unterschiede in der Lernfähigkeit.

Die Vorverarbeitungsarchitektur wurde als Abfolge unabhängiger Transformationen konzipiert, die einen gemeinsamen Vertrag teilen. Jede Transformation erhält ein Dokument, verändert nur die Felder, die zu ihrer Zuständigkeit gehören, und gibt das angereicherte Dokument zurück. Die Struktur lässt sich wie folgt zusammenfassen:

```python
class PreprocStep(Protocol):
    name: str
    version: str
    params: dict

    def transform(self, doc: dict) -> dict:
        ...
```

Die Pipeline normalisiert die Kodierung, entfernt oder konvertiert Markup, bereinigt Leerzeichen und wiederholte Zeichen, behandelt Emojis, erkennt die Sprache, segmentiert Sätze, tokenisiert Wörter, filtert Stoppwörter und protokolliert die entstehenden Annotationen. Die spätere Version nutzt spaCy als einheitlichen linguistischen Prozessor für Tokenisierung, Lemmatisierung, Wortarten-Tags und Abhängigkeitsrelationen. Diese Vereinheitlichung ist methodisch bedeutsam, weil syntaktische Abhängigkeiten, Lemmata und Token-Positionen sich auf dieselbe Tokenisierung beziehen müssen.

Auch die Stoppwortverarbeitung wurde an die Sentiment-Aufgabe angepasst. Häufige grammatische Wörter können aus lexikalischen Repräsentationen entfernt werden, aber Negationsbegriffe wie „not“, „no“ und „never“ müssen verfügbar bleiben, da sie die Interpretation eines bewertenden Ausdrucks umkehren können. Interpunktion wird beibehalten, wenn sie für die linguistische Analyse benötigt wird, und aus der endgültigen lexikalischen Repräsentation ausgeschlossen, wenn sie nur Rauschen hinzufügen würde.

Die verarbeiteten Dokumente werden in einem strukturierten JSONL-Korpus gespeichert, begleitet von Metadaten, Indizes, Statistiken und Schema-Informationen. Stabile Rezensions-Identifikatoren machen einzelne Datensätze auffindbar, während der Korpus-Reader sowohl Streaming- als auch gezielten Zugriff unterstützt. Jedes Dokument kann Rohtext, bereinigten Text, linguistische Annotationen, abgeleitete Merkmale, Vektor-Eingaben und Informationen über die Pipeline-Konfiguration bewahren, die es erzeugt hat.

Diese Korpusebene ist einer der zentralen Beiträge des Projekts. Sie trennt den Datenzugriff von der Modelllogik und ermöglicht es, unterschiedliche Repräsentationen auf dieselben zugrunde liegenden Dokumente anzuwenden. Sie schafft außerdem einen nachvollziehbaren Weg von einem experimentellen Ergebnis zurück zum Text, zur Vorverarbeitungskonfiguration und zum Split, aus dem dieses Ergebnis stammt.

## Korpuszusammensetzung und experimentelle Partitionen

Nach Filterung auf englischsprachige Rezensionen mit verwertbarem Text und gültigen numerischen Bewertungen enthält der P2-Korpus 526.869 Rezensionen. Das Verfahren zur Klassenbalancierung erzeugt genau 175.623 Rezensionen in jeder Polaritätskategorie.

Der Korpus wird mittels eines stratifizierten Verfahrens mit einem festen Zufalls-Seed von 42 unterteilt. Die Trainingspartition enthält 368.808 Rezensionen, die Validierungspartition 52.687 Rezensionen und die zurückgehaltene Testpartition 105.374 Rezensionen. Die Klassenanteile bleiben in jeder Partition praktisch identisch.

| Split | Rezensionen gesamt | Negativ | Neutral | Positiv |
|---|---:|---:|---:|---:|
| Training | 368.808 | 122.936 | 122.936 | 122.936 |
| Validierung | 52.687 | 17.563 | 17.562 | 17.562 |
| Test | 105.374 | 35.124 | 35.125 | 35.125 |
| **Gesamtkorpus** | **526.869** | **175.623** | **175.623** | **175.623** |

Die feste Validierungspartition wird für die Modell- und Hyperparameterauswahl verwendet. Die Testpartition bleibt separat, bis eine endgültige Konfiguration ausgewählt wurde. Diese Trennung verhindert, dass der Testdatensatz Teil des Optimierungsprozesses wird, und verleiht den finalen Werten eine klare Interpretation als Leistung auf zurückgehaltenen Daten.

Weil der Korpus perfekt ausgeglichen ist, lässt sich die Genauigkeit leichter interpretieren als bei einem stark unausgeglichenen Datensatz. Dennoch bleibt Makro-F1 das wichtigste Bewertungsmaß, weil es der Leistung bei den negativen, neutralen und positiven Klassen die gleiche Bedeutung beimisst. Klassenweise Precision, Recall, F1 und Konfusionsmatrizen werden festgehalten, damit die aggregierte Leistung systematische Fehler nicht verdeckt.

## Meinung mit lexikalischer und linguistischer Information repräsentieren

Die klassische Modellierungsphase erzeugt aus jeder verarbeiteten Rezension zwei sich ergänzende Eingaben. Die erste, `tfidf_input`, enthält eine lemmatisierte Version des Textes ohne Stoppwörter. Sie dient zur Konstruktion von TF-IDF-Vektoren mit Wort-n-Grammen. Die zweite, `opinion_input`, ist ein spärliches Wörterbuch expliziter linguistischer Information mit Bezug zur Polarität.

Die Meinungsrepräsentation erfasst Phänomene wie positive und negative lexikalische Einheiten, Negation, Intensivierung, Abschwächung und domänenbezogene bewertende Muster. Diese Merkmale versuchen, Wissen darzustellen, das ein herkömmlicher Bag of Words nicht explizit kodiert. Eine Formulierung wie „not enjoyable“ etwa enthält ein Zusammenspiel zwischen einer Negation und einem positiven Bewertungsbegriff. Eine linguistische Repräsentation kann dieses Zusammenspiel direkt festhalten, während eine lexikalische Repräsentation es aus dem beobachteten n-Gramm ableiten muss.

Der `CombinedVectorizer` erlaubt es derselben experimentellen Pipeline, in drei Modi zu arbeiten. Der TF-IDF-Modus verwendet nur lexikalische n-Gramme, der Meinungsmodus verwendet nur das explizite Merkmalswörterbuch, und der kombinierte Modus verkettet beide spärlichen Matrizen.

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

Der Vectorizer wird ausschließlich auf den Trainingsdokumenten angepasst. Validierungs- und Testdokumente werden mit dem Vokabular und den inversen Dokumenthäufigkeitswerten transformiert, die aus den Trainingsdaten gelernt wurden. Dies verhindert, dass lexikalische Informationen aus den zurückgehaltenen Partitionen die Repräsentation beeinflussen.

TF-IDF bleibt eine leistungsfähige Repräsentation für diese Domäne. Unigramme erfassen einzelne bewertende Begriffe, während Bigramme und längere n-Gramme kurze kontextuelle Muster wie Negation, Empfehlungen oder gängige Ausdrücke der Unzufriedenheit erfassen. Sublineare Termfrequenz reduziert den Einfluss wiederholter Wörter, und `min_df` entfernt extrem seltene Merkmale, die eher Rauschen als stabile Evidenz darstellen könnten.

Die explizite Meinungsrepräsentation dient einem anderen Zweck. Ihre Dimensionen sind weniger zahlreich und interpretierbarer, vereinfachen die Sprache aber zwangsläufig durch manuell entworfene Kategorien und lexikalische Ressourcen. Der Vergleich zwischen TF-IDF, Meinungsmerkmalen und ihrer Kombination stellt daher die Frage, ob strukturiertes linguistisches Wissen Informationen liefert, die über einen großen lexikalischen Raum hinausgehen.

## Klassische Machine-Learning-Experimente

Vier Modellfamilien wurden auf den drei Repräsentationsmodi evaluiert. Naive Bayes bietet eine effiziente probabilistische Baseline für spärlichen Text. Eine lineare Support Vector Machine lernt Entscheidungsgrenzen mit maximalem Rand im hochdimensionalen lexikalischen Raum. Random Forest führt nichtlineare Interaktionen durch ein Ensemble von Entscheidungsbäumen ein. XGBoost verwendet sequenziell konstruierte, regularisierte Bäume, um komplexere Merkmalsbeziehungen zu modellieren.

Kandidatenkonfigurationen werden auf dem Trainingssplit trainiert und nach Validierungs-Makro-F1 gerankt. Sobald die stärkste Konfiguration einer Modellfamilie identifiziert wurde, wird sie mit den kombinierten Trainings- und Validierungsdaten neu trainiert und einmalig am Testdatensatz evaluiert. Vorhersagen, klassenweise Berichte, Konfusionsmatrizen, Vectorizer, Modelle und Zusammenfassungsartefakte werden mit dem Experiment gespeichert.

Die genauen Testwerte unten wurden gegen die gespeicherten Evaluationsberichte überprüft.

| Modell | Siegreiche Repräsentation | Genauigkeit | Makro-F1 | Negativ-F1 | Neutral-F1 | Positiv-F1 |
|---|---|---:|---:|---:|---:|---:|
| Lineare SVM | TF-IDF | **0,7333** | **0,7309** | **0,8040** | **0,6291** | **0,7595** |
| Multinomial Naive Bayes | TF-IDF | 0,7277 | 0,7254 | 0,7974 | 0,6237 | 0,7552 |
| XGBoost | Kombiniert | 0,6904 | 0,6871 | 0,7537 | 0,5894 | 0,7182 |
| Random Forest | TF-IDF | 0,6439 | 0,6346 | 0,7169 | 0,5085 | 0,6785 |

Die stärkste klassische Konfiguration ist eine lineare SVM, die auf TF-IDF-Vektoren aus Unigrammen und Bigrammen arbeitet. Ihr Vokabular ist auf 20.000 Merkmale begrenzt, Begriffe müssen in mindestens zehn Dokumenten vorkommen, sublineare Termfrequenz ist aktiviert, und der SVM-Regularisierungsparameter beträgt `C=1.0`.

```python
tfidf_params = {
    "ngram_range": (1, 2),
    "max_features": 20_000,
    "min_df": 10,
    "sublinear_tf": True,
}

model = LinearSVC(C=1.0, class_weight=None)
```

Das Ergebnis demonstriert die Stärke spärlicher linearer Modellierung für große Textsammlungen. Die SVM liegt nur knapp vor Multinomial Naive Bayes, mit einer Differenz von etwa 0,0055 Makro-F1. Beide linearen Modelle übertreffen die baumbasierten Familien. In einem hochdimensionalen spärlichen Raum kann eine lineare Grenze die akkumulierte Evidenz aus vielen lexikalischen Dimensionen effizient nutzen. Baum-Ensembles müssen diesen Raum durch aufeinanderfolgende Entscheidungen partitionieren und tun sich schwer, dieselbe Geometrie ohne erheblich größere Komplexität abzubilden.

XGBoost erzielt sein stärkstes verifiziertes Ergebnis mit der kombinierten Repräsentation, was darauf hindeutet, dass die expliziten Meinungsmerkmale nützliche übergeordnete Informationen für Boosted Trees liefern. Dennoch bleibt sein endgültiger Makro-F1-Wert unter den beiden linearen Modellen. Random Forest liefert das schwächste Ergebnis und zeigt einen besonders großen Rückgang bei der neutralen Klasse.

Die klassenspezifischen Werte zeigen ein Muster, das die aggregierte Genauigkeit allein nicht ausdrücken kann. Negative Rezensionen sind durchweg am leichtesten zu erkennen. Positive Rezensionen erzielen ebenfalls relativ starke Ergebnisse. Neutrale Rezensionen bleiben für jedes Modell erheblich schwieriger. Selbst die siegreiche SVM erreicht bei neutral nur einen F1-Wert von 0,6291, verglichen mit 0,8040 bei negativ und 0,7595 bei positiv.

Dies ist nicht bloß eine Folge von Klassenungleichgewicht, da der Datensatz ausgeglichen ist. Die Schwierigkeit liegt in der semantischen Struktur der Labels und der Sprache selbst. Rezensionen in der Mitte der Bewertungsskala können schwache Meinungen, widersprüchliche Aspekte, bedingte Empfehlungen oder Sprache enthalten, die einer der extremen Klassen ähnelt, selbst wenn die Gesamtbewertung moderat ausfällt.

## Statische Embeddings und neuronale Architekturen

Die nächste Phase untersucht, ob dichte vortrainierte Wortrepräsentationen semantische Beziehungen erfassen können, die spärliche lexikalische Merkmale als unabhängige Dimensionen behandeln. Word2Vec GoogleNews, FastText-Crawl-Embeddings und GloVe-Dolma-Embeddings wurden mit Vektoren von 300 Dimensionen verglichen.

Zwei Architekturen wurden verwendet, um die Konsequenzen der Dokumentrepräsentation offenzulegen. Das Feedforward-Netzwerk erhält einen Vektor pro Review. Dieser Vektor wird durch Mittelung der Embeddings aller im Vokabular enthaltenen Wörter des Dokuments gewonnen.

```text
document_vector = (1 / n) * sum(word_vector_i)
```

Mean Pooling ist effizient und erzeugt eine semantische Repräsentation fester Größe unabhängig von der Dokumentlänge. Seine Einschränkung ist ebenso klar: Es entfernt die Wortreihenfolge und weist jedem enthaltenen Token dieselbe strukturelle Bedeutung zu. „The game is good but too long“ und eine Umstellung derselben Wörter können nahezu identische Mittelwerte erzeugen, obwohl die Reihenfolge des Diskurses und der Kontrast die Interpretation beeinflussen.

Das LSTM erhält stattdessen die Sequenz der Wort-Embeddings anstelle ihres Mittelwerts. Reviews werden auf eine feste Maximallänge aufgefüllt oder gekürzt, und das rekurrente Netzwerk verarbeitet die Embeddings in ihrer Reihenfolge. Dies ermöglicht es dem Modell, lokale Beziehungen, Stimmungswechsel, Negation und andere sequenzielle Informationen zu bewahren, die beim Mean Pooling verloren gehen.

FastText lieferte in den berichteten Vergleichen die stärkste Basisrepräsentation. Die Nutzung von Subwort-Informationen auf Zeichenebene verschafft ihm einen Vorteil bei seltenen Wörtern, morphologischen Varianten, Rechtschreibunterschieden und fachspezifischem Brettspiel-Vokabular. Die finalen FastText-Modelle verwendeten ein Feedforward-Netzwerk mit einer versteckten Dimension von 512 und ein LSTM mit einer versteckten Dimension von 128.

| Modell | Repräsentation | Accuracy | Macro-F1 | Negative F1 | Neutral F1 | Positive F1 |
|---|---|---:|---:|---:|---:|---:|
| FastText + FNN | Gemittelte Dokument-Embedding | 0.697 | 0.695 | 0.770 | 0.590 | 0.726 |
| FastText + LSTM | Geordnete Wort-Embedding-Sequenz | **0.752** | **0.752** | **0.823** | **0.657** | **0.776** |

Das LSTM verbessert das Macro-F1 um etwa 0,057 und bringt Gewinne in allen drei Klassen. Der Neutral-F1-Wert steigt von 0,590 auf 0,657. Dieser Vergleich liefert direkte Belege dafür, dass Wortreihenfolge und Sequenzaufbau Informationen tragen, die verloren gehen, wenn statische Wortvektoren auf einen Mittelwert reduziert werden.

Das Ergebnis macht die früheren spärlichen Modelle nicht obsolet. Die lineare SVM bleibt rechnerisch effizient, reproduzierbar und hochgradig konkurrenzfähig. Stattdessen klärt der neuronale Vergleich, welche zusätzlichen Informationen eine sequenzielle Repräsentation zurückgewinnen kann und wie sich diese Information auf die Klasse auswirkt, die für einfachere Modelle am schwierigsten ist.

## Kontextuelles Transferlernen mit BERT

Statische Embeddings weisen einem Wort unabhängig von seiner Verwendung einen festen Vektor zu. BERT ersetzt diese Annahme durch kontextuelle Repräsentationen. Der einem Token zugeordnete Vektor hängt von der umgebenden Sequenz ab, wodurch das Modell Bedeutungen und Funktionen unterscheiden kann, die ein festes Embedding zusammenführt.

Das Projekt feintunt `bert-base-uncased` mit einem dreiklassigen Sequenzklassifikationskopf. Der Rohtext der Reviews wird mit WordPiece-Tokenisierung verarbeitet. Spezielle Tokens werden hinzugefügt, Sequenzen werden auf 128 Tokens aufgefüllt oder gekürzt, und eine Attention-Maske unterscheidet echte Tokens von Padding. Sowohl die Klassifikationsschicht als auch der vortrainierte Transformer werden während des Feintunings aktualisiert.

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

Das BERT-Experiment nutzte aufgrund der rechnerischen Beschränkungen der Colab-Trainingsumgebung einen separaten, ausbalancierten Korpus von etwa 60.000 Reviews. Es verwendete außerdem Rohtext anstelle des lemmatisierten, stoppwortgefilterten Inputs, der bei mehreren früheren Modellen genutzt wurde. Diese Unterschiede sind bei der Interpretation der Werte wesentlich.

Nach der ersten Epoche erreichte das Modell eine Validierungsgenauigkeit von 0,8327 und ein Validierungs-Macro-F1 von 0,8304. Nach der zweiten Epoche erreichte die Validierungsgenauigkeit 0,8464 und das Macro-F1 0,8468. Die berichteten Ergebnisse auf dem Hold-out-Set waren eine Accuracy von 0,8431, ein Macro-F1 von 0,8432, ein Weighted-F1 von 0,8432 und ein Evaluationsverlust von 0,4420.

| BERT-Auswertungsmaß | Wert |
|---|---:|
| Test-Accuracy | 0.8431 |
| Test-Macro-F1 | 0.8432 |
| Test-Weighted-F1 | 0.8432 |
| Test-Evaluationsverlust | 0.4420 |

Diese Ergebnisse liefern starke Belege dafür, dass kontextuelles Transferlernen für die Domäne hervorragend geeignet ist. BERT kann jedes Token mit seinem umgebenden Kontext in Beziehung setzen und Funktionswörter, Interpunktion, vollständige Wortformen und Diskurssignale bewahren, die in der klassischen Pipeline reduziert oder entfernt werden.

Der BERT-Wert wird nicht als direkter numerischer Sieg über die SVM auf dem vollständigen Korpus oder das LSTM dargestellt, da die Modelle nicht auf demselben Benchmark evaluiert wurden. Das Ergebnis belegt den Wert des kontextuellen Ansatzes innerhalb seines eigenen dokumentierten Rahmens. Es zeigt auch, wie vortrainiertes sprachliches Wissen nach nur zwei Feintuning-Epochen auf einem Korpus, der viel kleiner als die vollständige P2-Sammlung ist, hohe Leistung erzielen kann.

![Berichtete Macro-F1-Ergebnisse, getrennt nach vergleichbarem experimentellem Rahmen](/assets/articles/bgg_review_intelligence/model-performance-overview.svg)

*Die klassischen Modelle teilen sich ein kontrolliertes Test-Set. Die neuronalen und kontextuellen Ergebnisse bewahren die im P3-Datensatz berichteten Rahmenbedingungen, weshalb die Panels bewusst getrennt und nicht als eine einzige Rangtabelle dargestellt werden.*

## Vom globalen Polaritätswert zum aspektbasierten Verständnis

Die Dokumentklassifikation komprimiert ein ganzes Review auf ein einziges Label. Diese Antwort ist für großangelegte Messungen nützlich, kann aber nicht erklären, welche Teile des Spiels das Urteil hervorgebracht haben. Die letzte Phase des Projekts ändert daher die Ausgabestruktur, statt einfach einen Klassifikator durch einen anderen zu ersetzen.

Das aspektbasierte Experiment nutzt ein großes Sprachmodell, um zwanzig ausgewählte RoboRally-Reviews zu analysieren. Die Reviews werden über den vorhandenen Corpus-Reader geladen und in einen strukturierten Prompt eingefügt. Statt eine unbegrenzte Zusammenfassung anzufordern, definiert der Prompt eine JSON-Struktur mit dem Spielnamen, einer Gesamtschlussfolgerung, einer Liste von Aspekten, einer Polarität für jeden Aspekt und wörtlichen Belegstellen.

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

Die generierte Analyse charakterisierte die ausgewählten Reviews als überwiegend negativ und ordnete ihre Kritik in kohärente Kategorien ein. Zu den extrahierten Aspekten gehörten Dauer, Zufälligkeit und Chaos, Spielerausschluss, Zugänglichkeit, Interaktion, Balance, Spieler-Handlungsspielraum und Gesamteindruck. Belegstellen wurden mit den Schlussfolgerungen verknüpft, sodass die strukturierte Ausgabe mit der Quellsprache verbunden blieb.

Dieses Experiment zeigt eine andere Art von Fähigkeit als die Sentiment-Klassifikatoren. BERT, LSTM und SVM schätzen eine globale Klasse mit messbarer Vorhersageleistung. Das prompt-gesteuerte Sprachmodell zerlegt Meinungen in interpretierbare Einheiten und erklärt, warum die ausgewählten Rezensentinnen und Rezensenten negativ reagierten.

Das Aspekt-Ergebnis ist qualitativ und kein Benchmark-Wert. Die zwanzig Reviews gehören zu einer bewusst ausgewählten negativen Sequenz, und es wurde kein unabhängig annotierter Aspekt-Datensatz verwendet. Das Experiment demonstriert daher Machbarkeit und Ausdruckswert, nicht gemessene Genauigkeit der Aspektextraktion. Seine Bedeutung innerhalb des Projekts ist konzeptueller Natur: Es verbindet die skalierbare Korpus-Infrastruktur mit einer Repräsentation der Meinung, die die innere Struktur eines Reviews genauer widerspiegelt.

## Was die zusammengeführten Belege zeigen

Die vollständige Abfolge der Experimente offenbart mehrere durchgängige Erkenntnisse. Die erste ist, dass sorgfältiges Daten- und Sprachdesign auch dann wichtig bleibt, wenn ausgefeilte Modelle verfügbar sind. Stabile Identifikatoren, deterministische Aufteilungen, nachvollziehbare Vorverarbeitung, gespeicherte Vorhersagen und klassenweise Auswertung sind das, was Modellergebnisse interpretierbar macht. Ohne dieses Fundament lässt sich ein höherer Wert nicht von Unterschieden in der Datenvorbereitung oder den Evaluationsbedingungen unterscheiden.

Die zweite Erkenntnis ist die anhaltende Stärke lexikalischer Baselines. TF-IDF mit einer linearen SVM erreicht ein Macro-F1 von 0,7309 auf mehr als einhunderttausend zurückgehaltenen Reviews. Multinomial Naive Bayes schneidet nur geringfügig schlechter ab. Diese Ergebnisse zeigen, dass ein großer Anteil der Sentiment-Information im Korpus durch wiederkehrende lexikalische Muster ausgedrückt wird, die ohne tiefe Architektur erfasst werden können.

Die dritte Erkenntnis ist, dass Reihenfolge und Kontext messbare Information hinzufügen. Das LSTM verbessert sich erheblich gegenüber einem gemittelten Feedforward-Modell auf derselben FastText-Grundlage. Das BERT-Experiment erreicht innerhalb seines eigenen Rahmens einen noch höheren Wert und zeigt den Wert kontextualisierter Repräsentationen und Transferlernens.

Die vierte und beständigste Erkenntnis ist die Schwierigkeit der Neutralität. Dieses Muster zeigt sich sowohl bei klassischen Modellen als auch bei neuronalen Netzen. Die negative und die positive Klasse haben klarere lexikalische Grenzen, während die neutrale Klasse gemischte Bewertungen, schwaches Sentiment, Grenzwertbewertungen und Diskrepanzen zwischen Text und Bewertung aufnimmt. Neutralität ist in diesem Korpus daher nicht einfach ein mittlerer emotionaler Zustand. Sie ist eine heterogene Region, die durch Sprache, Nutzerverhalten und die gewählte Label-Konstruktion entsteht.

Die fünfte Erkenntnis ist, dass explizite linguistische Meinungsmerkmale einen bedingten und keinen universellen Wert haben. Sie fügen interpretierbare Information über Negation, Intensivierung und domänenspezifische Ausdrücke hinzu, doch hochdimensionale lexikalische Modelle kodieren bereits viele derselben Muster über N-Gramme. Ihre Wirkung hängt von der Modellfamilie und der Geometrie der Repräsentation ab. XGBoost erzielt seine stärkste verifizierte Konfiguration aus dem kombinierten Input, während die stärksten SVM- und Naive-Bayes-Modelle allein TF-IDF verwenden.

Schließlich zeigt das Projekt, dass Vorhersage und Erklärung als unterschiedliche Ausgaben behandelt werden müssen. Ein Klassifikator kann die globale Polarität effizient über Hunderttausende Dokumente hinweg schätzen. Ein aspektbasiertes System kann die Themen und Belege offenlegen, die dieser Polarität zugrunde liegen. Zusammen liefern diese Ebenen eine vollständigere Repräsentation der Meinung als jede für sich allein.

## Was wir gelernt haben, was ungewiss bleibt und wohin die Arbeit führt

Die wichtigste Lehre aus diesem Prozess ist, dass sich die Qualität eines NLP-Ergebnisses nicht von der Struktur des Korpus trennen lässt, der es hervorgebracht hat. Am Anfang war es verlockend, das Preprocessing als vorbereitende Aufgabe zu betrachten, die abgeschlossen sein musste, bevor das „eigentliche“ Modellieren beginnen konnte. Die späteren Experimente haben diese Sichtweise verändert. Entscheidungen über Sprachfilterung, Tokenisierung, Lemmatisierung, Stoppwörter, Interpunktion, Label-Schwellenwerte und Dokumentaufteilungen definieren, was die Modelle überhaupt beobachten können und was ihre Werte bedeuten. Der Korpus ist keine passive Eingabe für die Forschung; er ist selbst Teil des Forschungsgegenstands.

Wir haben auch gelernt, dass Komplexität Einfachheit nicht ungültig macht. Die lineare SVM und Multinomial Naive Bayes bleiben stark, selbst nachdem das Projekt rekurrente und kontextuelle Architekturen einführt. Ihre Leistung zeigt, dass wiederkehrende lexikalische Muster einen großen Teil der Information über die aus Bewertungen abgeleitete Polarität enthalten. Das liefert einen wichtigen Referenzpunkt für die Interpretation jedes komplexeren Modells. Eine tiefere Architektur ist wissenschaftlich dann interessant, wenn sie Information erfasst, die die Baseline nicht darstellen kann – nicht bloß, weil sie neuer ist.

Der Vergleich zwischen dem Feedforward-Netz und dem LSTM machte diesen Unterschied sichtbar. Beide Modelle gingen von statischen FastText-Embeddings aus, aber nur das LSTM bewahrte die Reihenfolge der Wörter. Seine Verbesserung, besonders bei der neutralen Klasse, deutet darauf hin, dass ein Teil der fehlenden Information kompositioneller Natur ist. Verneinung, Kontrast, Einschränkung und Tonwechsel lassen sich nicht vollständig auf einen Durchschnitt von Wortbedeutungen reduzieren. BERT entwickelt diesen Gedanken weiter, indem es zulässt, dass die Repräsentation jedes Tokens von seinem vollständigen Kontext abhängt.

Gleichzeitig hat das BERT-Ergebnis einen der wichtigsten Zweifel geschaffen, der im Projekt bestehen bleibt. Sein berichteter Makro-F1-Wert ist der höchste, wurde jedoch auf einem kleineren, separat aufbereiteten Korpus und mit Rohtext statt der verarbeiteten lexikalischen Eingabe erzielt, die bei mehreren früheren Modellen verwendet wurde. Wir können sagen, dass kontextuelles Transfer-Learning in diesem Setting sehr gut funktioniert hat. Wir können noch nicht genau sagen, wie viel des Unterschieds von der Architektur, dem Korpus-Teilbestand, dem Eingabetext oder den Trainingsbedingungen herrührt. Diese Ungewissheit ist keine Schwäche, die es zu verbergen gilt; sie ist Teil dessen, was uns die experimentelle Aufzeichnung über fairen Modellvergleich gelehrt hat.

Der naheliegende Forschungsweg, der sich aus dieser Beobachtung ergibt, beginnt bei der Vergleichbarkeit. Die bestehenden Experimente haben bereits die Modelle identifiziert, die einen Vergleich wert sind. Was jetzt fehlt, ist ein einziger fixierter Benchmark, auf dem das stärkste sparse Modell, das rekurrente Modell und das kontextuelle Modell dieselben Reviews betrachten und nach demselben Verfahren evaluiert werden. Ein solcher Vergleich würde den aktuellen Verlauf von einer Sammlung starker, stufenspezifischer Ergebnisse in eine kontrollierte Darstellung dessen verwandeln, was jede Repräsentation beiträgt.

Die neutrale Klasse wirft einen tieferen Zweifel auf. Ihre Schwierigkeit besteht über Algorithmen hinweg fort, die sehr unterschiedliche Annahmen treffen. Das legt nahe, dass das Problem nicht allein bei den Klassifikatoren liegt. Es könnte auch in der Definition der Neutralität selbst liegen. Bewertungen von 4 bis 7 decken ein breites Intervall ab, und die zugehörigen Reviews können Enttäuschung, moderate Zustimmung, gemischte Aspekte, Unsicherheit oder bedingte Empfehlung ausdrücken. All diese Fälle „neutral“ zu nennen, schafft eine praktische Kategorie, aber vielleicht kein einheitliches sprachliches Phänomen.

Das führt zu einer Frage, die die aktuellen automatischen Labels nicht beantworten können: Wenn ein Modell der aus der Bewertung abgeleiteten Klasse widerspricht, liegt das Modell dann falsch, oder hat es Sentiment im Text erkannt, das der Bewertungsschwellenwert nicht abbildet? Die Antwort würde erfordern, eine sorgfältig ausgewählte Gruppe von Reviews zu lesen und unabhängig zu annotieren, insbesondere solche nahe den Bewertungsgrenzen und solche, bei denen Modelle uneinig sind. Auch menschliche Uneinigkeit wäre aufschlussreich. Wenn Leser diesen Reviews nicht konsistent ein Sentiment zuordnen, sollte Unsicherheit als Teil der Daten behandelt werden, nicht als Fehler, den ein Klassifikator beseitigen soll.

Das aspektbasierte Experiment macht diese Frage noch bedeutsamer. Sobald ein Review in Spieldauer, Interaktion, Zugänglichkeit, Balance oder Spieler-Handlungsfreiheit zerlegt wird, wird gemischtes Sentiment leichter verständlich. Ein Review muss nicht im üblichen Sinne global neutral sein; es kann bei einem Aspekt stark positiv und bei einem anderen stark negativ sein. Die globale Bewertung stellt dann eine Aggregation dar, die der Rezensent vorgenommen hat, während ein Aspektmodell versucht, die Bestandteile dieser Aggregation wiederherzustellen.

Das hat unser Verständnis des endgültigen Ziels verändert. Zunächst erschien Aspektextraktion als eine erklärende Schicht, die nach der Klassifikation angesiedelt ist. Das Experiment legt nahe, dass sie auch eine bessere Repräsentation des Problems selbst bieten könnte. Globales Sentiment und Aspekt-Sentiment stehen nicht in Konkurrenz zueinander. Das globale Label liefert Skalierung und Vergleichbarkeit, während die Aspekte die innere Struktur der Meinung bewahren. Die Beziehung zwischen beiden könnte informativer sein als jedes Ergebnis für sich genommen.

Das LLM-Experiment hinterlässt ebenfalls einen wichtigen methodischen Zweifel. Seine Ausgabe ist kohärent und wird durch erkennbare Formulierungen aus den ausgewählten Reviews gestützt, aber Kohärenz ist nicht dasselbe wie gemessene Treue zur Quelle. Ein generatives Modell kann Sprache überzeugend organisieren, selbst wenn ein Aspekt unvollständig ist, eine Belegformulierung verändert wurde oder eine Schlussfolgerung stärker ausfällt, als die Quelle es zulässt. Das Experiment hat uns daher gelehrt, Lesbarkeit von Validität zu trennen. Strukturiertes JSON und zitierte Belege machen das Ergebnis überprüfbar, aber eine echte Bewertung hängt vom Vergleich mit menschlich annotierten Aspekten und exakten Quellstellen ab.

Die Roadmap, die sich aus diesen Lehren ergibt, ist keine Abfolge von Software-Erweiterungen. Sie ist eine Abfolge von Fragen. Ein gemeinsamer Benchmark klärt den Beitrag jeder Repräsentation. Menschliches Lesen schwieriger Reviews klärt, was Neutralität bedeutet. Aspekt-Annotation klärt, wie globale Urteile zusammengesetzt sind. Beleggestützte Evaluation klärt, ob generative Erklärungen wahrheitsgetreu sind. Erst wenn diese Zusammenhänge verstanden sind, wird es sinnvoll, zu untersuchen, wie gut sich die Schlussfolgerungen auf andere Spiele, Zeiträume, Sprachen oder Review-Domänen übertragen lassen.

Es besteht auch Ungewissheit darüber, was die Modelle aus der Identität der Domäne lernen. Eine zufällige Aufteilung auf Review-Ebene kann dazu führen, dass ähnliches Vokabular aus denselben Spielen oder denselben Nutzertypen sowohl in Trainings- als auch in Testdaten erscheint. Eine starke Leistung kann daher echtes Sentiment-Verständnis widerspiegeln, Vertrautheit mit bestimmten Spiel-Vokabularen, oder eine Kombination aus beidem. Die aktuellen Ergebnisse beschreiben Generalisierung auf zurückgehaltene Reviews aus demselben breiten Korpus. Sie beschreiben noch nicht Generalisierung auf völlig unbekannte Spiele, neue Communities oder spätere Zeiträume.

Diese Zweifel schmälern das Projekt nicht. Sie sind ein Beleg dafür, dass die Arbeit ein Stadium erreicht hat, in dem die interessanten Fragen nicht mehr auf die Wahl eines Algorithmus beschränkt sind. Die Experimente haben Beziehungen zwischen Bewertungen, Sprache, Kontext, Aspekten und Belegen offengelegt. Sie haben auch gezeigt, wo numerische Leistung ausreicht und wo sie das Phänomen unzureichend erklärt lässt.

Die Grenzen der Evidenz bleiben klar. Die klassischen Modelle teilen sich einen großen, ausgewogenen Korpus und einen gemeinsamen Testsatz, sodass ihr Vergleich kontrolliert ist. Das FNN und das LSTM teilen sich das Setting mit statischen Embeddings, was ihren architektonischen Vergleich aussagekräftig macht. BERT gehört zu einem anderen Datensatz und Eingabe-Setting. Die Aspektanalyse ist eine qualitative Demonstration, die auf einer kleinen, bewusst negativ ausgewählten Gruppe von Reviews basiert. Jedes Ergebnis ist innerhalb seiner eigenen Bedingungen wertvoll, und diese Bedingungen sichtbar zu halten ist das, was dem Gesamtprojekt erlaubt, als vertrauenswürdige Wissensgrundlage zu dienen.

## Fazit

Dieses Projekt konstruiert einen durchgehenden Weg von rohen Nutzer-Reviews zu strukturierter Meinungsintelligenz. Es beginnt damit, Korpusgestaltung und sprachliche Konsistenz als Teil des wissenschaftlichen Problems zu behandeln. Anschließend wird geprüft, ob Sentiment am besten durch lexikalische Statistiken, explizites sprachliches Wissen, dichte semantische Vektoren, sequenzielle Verarbeitung oder kontextuelles Transfer-Learning dargestellt wird. Schließlich wird die Aufgabe von globaler Klassifikation auf aspektbezogene Interpretation erweitert, die in Review-Belegen verankert ist.

Die klassischen Experimente etablieren eine starke und effiziente Baseline: Eine lineare SVM mit TF-IDF erreicht einen Makro-F1-Wert von 0,7309 auf einem ausgewogenen Testsatz von 105.374 Reviews. Die Experimente mit statischen Embeddings zeigen den Wert der Wortreihenfolge, wobei FastText und ein LSTM im berichteten Großkorpus-Setting einen Makro-F1-Wert von 0,752 erreichen. Das BERT-Experiment zeigt das Potenzial kontextuellen Transfer-Learnings und erreicht einen Makro-F1-Wert von 0,8432 auf seinem separaten, ausgewogenen Korpus. Das LLM-Experiment zeigt, wie dieselbe Datengrundlage strukturierte Erklärungen zu Spieldauer, Zufälligkeit, Zugänglichkeit, Interaktion, Balance und Spieler-Handlungsfreiheit unterstützen kann.

Wichtiger als jeder einzelne Wert ist das Wissen, das durch den Verlauf selbst entsteht. Sparse lexikalische Modelle zeigen, wie viel sich aus wiederkehrender Sprache lernen lässt. Linguistische Merkmale offenbaren, welche bewertenden Strukturen explizit gemacht werden können. Rekurrente Modelle demonstrieren die Rolle der Sequenz. BERT erfasst Kontext durch vortrainierte Attention. Aspektextraktion stellt die mehrdimensionale Natur der Meinung wieder her, die ein globales Label zwangsläufig komprimiert.

Zusammengenommen bilden diese Stufen eine solide Grundlage für das Verständnis von Meinung in domänenspezifischem Review-Text. Das Projekt behandelt NLP nicht als isolierte Auswahl eines Modells. Es behandelt Datenherkunft, sprachliche Struktur, Repräsentation, Evaluation, Interpretation und Evidenz als verbundene Teile desselben Forschungsgegenstands.
