---
title: "Ersatzmodelle für die Suche nach optimalen Eingaben"
topic: "Bachelorarbeit"
date: 2026-07-10
excerpt: "Gaußprozesse, Bayesianische Optimierung und Expected Improvement zur Priorisierung neuer Auswertungen, wenn Experimentieren teuer ist."
# tags: ["Ersatzmodelle", "Gaußprozesse", "Bayesianische Optimierung"]
draft: false
kind: "article"
readingTime: 125
lang: "de"
translated: true
sourceHash: "d91308047d3212eb"
---

Este artículo recoge mi Trabajo de Fin de Grado en Ciencia e Ingeniería de Datos. Estudia cómo utilizar procesos gaussianos como modelos sustitutos para aprender con pocas evaluaciones, cuantificar la incertidumbre y decidir qué configuraciones conviene probar a continuación.

Dieser Artikel fasst meine Abschlussarbeit (Trabajo de Fin de Grado) in Datenwissenschaft und -technik zusammen. Sie untersucht, wie man Gaußprozesse als Surrogatmodelle nutzen kann, um mit wenigen Auswertungen zu lernen, Unsicherheit zu quantifizieren und zu entscheiden, welche Konfigurationen als Nächstes ausprobiert werden sollten.

Der Code, die Experimente und die Materialien zur Reproduktion sind auf GitHub verfügbar:

<https://github.com/trentisiete/surrogate_models>

---

## Zusammenfassung

In zahlreichen experimentellen Kontexten liegt die Hauptschwierigkeit nicht nur in der Analyse der erzielten Ergebnisse, sondern auch in der Entscheidung, welche Experimente als Nächstes durchgeführt werden sollen, wenn jede Auswertung mit hohen Kosten verbunden ist. In diesem Zusammenhang stellte uns die Grupo de Ingredientes Alimentarios Saludables (Gruppe für gesunde Lebensmittelzutaten) der UAM ein Problem vor, das auf die Optimierung der Auswahl neuer experimenteller Konfigurationen bei der Fütterung von *Hermetia illucens*-Larven ausgerichtet ist.

Das geeignete Werkzeug für solche Probleme sind Surrogatmodelle, die darin bestehen, ein Modell des maschinellen Lernens zu erstellen, das die Beziehung zwischen den Eigenschaften einer Diät und den davon zu erwartenden Ergebnissen annähert. Auf diese Weise kann das Modell vor der Durchführung eines neuen Versuchs dabei helfen, abzuschätzen, welche Futterkombinationen am vielversprechendsten erscheinen und mit welchem Grad an Vertrauen solche Vorhersagen getroffen werden. Insbesondere werden die Surrogatmodelle auf Gaußprozessen basieren, da diese es erlauben, Vorhersage und Unsicherheit in ein und demselben Rahmen zu vereinen.

Die Abschlussarbeit gliedert sich in drei Etappen. Zunächst wird der theoretische Rahmen festgelegt, der notwendig ist, um Surrogatmodelle, Gaußprozesse und die Art und Weise zu verstehen, wie Unsicherheit genutzt werden kann, um neue Auswertungen vorzuschlagen. Zweitens werden synthetische Experimente an bekannten Funktionen durchgeführt, was es erlaubt, das Verhalten dieser Werkzeuge kontrolliert zu untersuchen: ausgehend von wenigen Anfangsdaten, ein Modell anpassen, neue Punkte mittels *Expected Improvement* auswählen und prüfen, ob der Prozess das beste gefundene Ergebnis verbessert. Diese Experimente zeigen, dass das vorgeschlagene Verfahren das anfängliche Design in allen analysierten Benchmarks verbessert, wobei seine Leistung jedoch von verschiedenen Faktoren abhängt.

Schließlich wird die Methodik auf den realen Fall von Diäten für *Hermetia illucens* angewendet. In diesem Szenario wird das Modell bewertet, indem vollständige Diäten während des Trainings ausgelassen werden und geprüft wird, ob es in der Lage ist, deren Ergebnisse vorherzusagen. Diese Form der Bewertung wurde gewählt, weil sie sich der tatsächlichen angestrebten Nutzung annähert: das Verhalten einer noch nicht getesteten Diät abzuschätzen und das Modell auszuwählen, das am besten auf neue Vorschläge generalisiert. Die Ergebnisse zeigen, dass der Gaußprozess bei den drei letztlich analysierten Zielgrößen das Basismodell verbessert, wobei die prädiktive Unsicherheit jedoch nur teilweise kalibriert erscheint.

Insgesamt zeigt die Arbeit, dass Surrogatmodelle die experimentelle Validierung nicht ersetzen, aber durchaus den Suchraum verringern, Unsicherheit quantifizieren und neue Kandidatenkonfigurationen priorisieren können, wenn das experimentelle Budget begrenzt ist.

## Einleitung

Viele reale Optimierungsprobleme bestehen darin, eine Eingabekonfiguration zu finden, die eine günstige Antwort erzeugt. In zahlreichen Kontexten der Ingenieurwissenschaft, der Datenwissenschaft oder der physikalischen Experimentierung kann die Bewertung jeder Konfiguration jedoch kostspielig, langsam oder begrenzt sein. In diesen Fällen ist es nicht praktikabel, den gesamten Suchraum erschöpfend zu erkunden oder direkt Methoden anzuwenden, die viele Auswertungen der Zielfunktion erfordern. Das Problem besteht dann nicht mehr allein darin, den besten Punkt zu finden, sondern darin, zu entscheiden, wie man aus einer begrenzten Anzahl von Beobachtungen möglichst viel lernen kann.

Dieses Szenario tritt auf, wenn sich die zu optimierende Funktion wie eine Black Box verhält: Man kann Ausgaben für bestimmte Eingaben beobachten, verfügt aber weder über einen einfachen analytischen Ausdruck noch über Ableitungen, die die Suche direkt leiten könnten. Zudem können die Beobachtungen durch Rauschen oder experimentelle Variabilität beeinflusst sein. Daher muss jede Optimierungsstrategie zwei Bedürfnisse in Einklang bringen: das Verhalten der Funktion mit den verfügbaren Daten anzunähern und zu entscheiden, welche neuen Konfigurationen es zu bewerten gilt.

In der Literatur zur Optimierung auf Basis von Surrogatmodellen wird diese Art von Problemen angegangen, indem aus einer begrenzten Menge von Beobachtungen kostengünstig auszuwertende Näherungen konstruiert werden. Dieser Ansatz wurde besonders in Design- und Ingenieurkontexten verwendet, in denen die Auswertungen aus hochpräzisen Simulationen, physikalischen Versuchen oder kostspieligen experimentellen Prozessen stammen können. Im Allgemeinen kombiniert das Verfahren ein anfängliches Versuchsdesign, die Anpassung eines Näherungsmodells, dessen Validierung und, wenn möglich, eine sequenzielle Phase der Anreicherung des Datensatzes durch neue Auswertungspunkte.

Innerhalb dieser Methodenfamilie nehmen Gaußprozesse eine herausragende Rolle ein, weil sie nicht nur eine Punktvorhersage liefern, sondern auch eine mit jeder Vorhersage verbundene Unsicherheitsschätzung. Diese Unsicherheit erlaubt es, zwischen Regionen, die durch die verfügbaren Daten gut erklärt sind, und Regionen, die noch wenig erforscht sind, zu unterscheiden. Deshalb ist sie besonders nützlich in *Infill*-Strategien, bei denen Akquisitionsfunktionen wie Expected Improvement (EI) die Ausnutzung vielversprechender Bereiche und die Erkundung unsicherer Bereiche kombinieren, um neue Auswertungen vorzuschlagen.

In diesem Zusammenhang untersucht die vorliegende Abschlussarbeit den Einsatz von Gaußprozessen als probabilistische Surrogatmodelle in zwei sich ergänzenden Szenarien. Zunächst werden synthetische Benchmarks verwendet, bei denen die Zielfunktion bekannt ist und kontrolliert ausgewertet werden kann, um den vollständigen Zyklus der sequenziellen Optimierung mittels GP + EI zu analysieren. Zweitens wird als angewandter Test ein realer Fall auf Grundlage experimenteller Daten zu Diäten für *Hermetia illucens* einbezogen. In diesem zweiten Szenario stammen die Beobachtungen aus bereits durchgeführten Versuchen, und die Gewinnung neuer Stichproben würde zusätzliche Experimente erfordern. Das Ziel besteht daher nicht darin, eine endgültige optimale Diät zu bestimmen, sondern zu bewerten, ob das Modell auf ungesehene Diäten generalisieren und vernünftige Kandidaten für eine mögliche zukünftige Bewertung vorschlagen kann.

Die Arbeit beabsichtigt daher nicht, Surrogatmodelle als Ersatz für die experimentelle Validierung darzustellen. Ihre Rolle wird als unterstützendes Werkzeug verstanden: Sie erlauben es, den Suchraum zu verringern, Unsicherheit zu quantifizieren und neue Auswertungen zu priorisieren, wenn das experimentelle Budget begrenzt ist. Diese Unterscheidung ist besonders wichtig im realen Fall, wo die Vorhersagen des Modells als experimentelle Hypothesen und nicht als abgeschlossene biologische Schlussfolgerungen interpretiert werden müssen.

### Ziele

Das allgemeine Ziel dieser Abschlussarbeit besteht darin, einen Rahmen für die Optimierung auf Basis probabilistischer Surrogatmodelle zu entwickeln und zu bewerten, mit Schwerpunkt auf Gaußprozessen (Gaussian Processes, GP), um Zielfunktionen mit begrenzten Auswertungen anzunähern und die Suche nach vielversprechenden Konfigurationen zu unterstützen.

Um dieses allgemeine Ziel zu erreichen, werden die folgenden spezifischen Ziele formuliert:

1.  Die Grundlagen der Surrogatmodelle und insbesondere der Gaußprozesse als probabilistische Modelle untersuchen, die in der Lage sind, sowohl Punktvorhersagen als auch Unsicherheit zu liefern.

2.  Einen experimentellen Rahmen auf Basis von Gaußprozessen und EI implementieren, um den vollständigen Zyklus der sequenziellen Optimierung an synthetischen Benchmark-Funktionen zu analysieren.

3.  In den Benchmarks bewerten, ob der *Infill*-Prozess es ermöglicht, den besten gefundenen Wert gegenüber dem Anfangsdesign zu verbessern, und analysieren, wie Faktoren wie der Kernel, die Anfangsgröße, das Rauschen und die Dimensionalität des Problems dies beeinflussen.

4.  Die prädiktive Qualität des GP mit seinem Nutzen zur Steuerung der Optimierung vergleichen, wobei zwischen Fehlermetriken, probabilistischen Metriken und der Verbesserung des *Incumbent* unterschieden wird.

5.  Den Ansatz auf einen realen Fall von Diäten für *Hermetia illucens* anwenden, indem eine überwachte Repräsentation aus Formulierungsvariablen, Nährstoffzusammensetzung und Art des Nebenprodukts konstruiert wird.

6.  Die Generalisierungsfähigkeit des Modells im realen Fall mittels eines *Leave-One-Diet-Out*-Protokolls bewerten, wobei Informationslecks zwischen Wiederholungen derselben Diät vermieden werden.

7.  Analysieren, ob die prädiktive Unsicherheit des GP als unterstützendes Signal dienen kann, um die Zuverlässigkeit der Vorhersagen zu interpretieren und neue Kandidatenformulierungen zu priorisieren.

8.  Eine prospektive Pre-*Infill*-Phase vorschlagen, die es erlaubt, Kandidaten für zukünftige experimentelle Bewertungen zu identifizieren.

Diese Ziele sind so konzipiert, dass sie sowohl den methodischen als auch den angewandten Teil der Arbeit abdecken. Die Benchmarks erlauben es, das Verhalten des Ansatzes in einer kontrollierten Umgebung zu überprüfen, während der reale Fall es erlaubt, seine Möglichkeiten und Grenzen zu untersuchen, wenn die Daten knapp, nach Diät gruppiert und aus experimentellen Versuchen des realen Falls stammen.

### Aufbau des Dokuments

Der Rest des Dokuments ist wie folgt gegliedert. Kapitel 2 führt in den für die Arbeit notwendigen theoretischen Rahmen ein, einschließlich der Optimierung mit kostspieligen Auswertungen, der Surrogatmodelle, der Gaußprozesse, des Versuchsdesigns, der Akquisitionsfunktionen und der Bewertungsmetriken. Kapitel 3 stellt die durchgeführten Experimente vor, zunächst an synthetischen Benchmarks und anschließend am realen Fall von Diäten für Insekten. Abschließend fasst Kapitel 4 die wichtigsten gewonnenen Schlussfolgerungen zusammen und zeigt mögliche Linien zukünftiger Arbeit auf.

## Theoretische Grundlagen der Optimierung auf Basis von Surrogatmodellen

Viele reale Optimierungsprobleme in der Ingenieur- und Datenwissenschaft lassen sich mittels einer Zielfunktion $f : X \to \mathbb{R}$ formulieren, deren Wert nur durch die Auswertung eines externen Verfahrens gewonnen werden kann, etwa einer komplexen numerischen Simulation oder eines physikalischen Experiments. In diesem Zusammenhang spricht man von einer *Black-Box*-Funktion, wenn für praktische Zwecke weder ein handhabbarer analytischer Ausdruck von $f$ noch Ableitungen noch strukturelle Eigenschaften zur Verfügung stehen, die es erlauben würden, klassische Optimierungsmethoden direkt anzuwenden. Mit anderen Worten: Die Funktion kann nur an konkreten Punkten des Definitionsbereichs abgefragt werden und liefert eine mit jeder Eingabe verbundene Ausgabe.

Zudem ist bei vielen dieser Probleme jede Auswertung von $f(x)$ mit hohen Kosten verbunden, sei es an Rechenzeit oder Ressourceneinsatz. Die Schwierigkeit liegt also nicht nur darin, dass die Funktion intern unbekannt ist, sondern auch darin, dass sie nur eine begrenzte Anzahl von Malen ausgewertet werden kann.

Um dieses Problem zu formalisieren, führen wir einen kontinuierlichen Suchraum $\mathcal{X}\subset\mathbb{R}^{d}$ ein, der alle möglichen Konfigurationen der Eingabevariablen enthält, zusammen mit einer Zielfunktion $f:\mathcal{X}\to\mathbb{R}$, die das Verhalten des Systems modelliert. Der Zweck der *globalen Optimierung* besteht darin, einen globalen Minimierer zu finden

$$
\bm{x}^{\star} \in \arg\min_{\bm{x}\in\mathcal{X}} f(\bm{x}).
$$

Unsere Arbeitshypothese lautet, dass die Kosten $c(\bm{x})$ für die Auswertung der Zielfunktion hoch sind. Folglich wird die Anzahl der möglichen Auswertungen durch ein maximales Budget $B$ begrenzt, sodass nur eine begrenzte Menge an Beobachtungen zur Verfügung steht:

$$
\mathcal{D}_n = \{(\bm{x}_i, y_i)\}_{i=1}^{n}, \qquad n \leq B,
$$

wobei $y_i$ den beobachteten Wert bei der Auswertung der Funktion am Punkt $\bm{x}_i$ darstellt.

Selbst in den Fällen, in denen die Kosten einer einzelnen Auswertung tragbar wären, bliebe eine erschöpfende Suche undurchführbar, wenn die Dimension des Raums $\mathcal{X}$ hoch ist. Tatsächlich wächst die Anzahl der Punkte, die erforderlich sind, um den Definitionsbereich mit vergleichbarer Dichte zu erkunden, exponentiell mit der Dimension – ein Phänomen, das als *Fluch der Dimensionalität* bekannt ist. Die Schwierigkeit des Problems rührt also nicht nur von den Kosten pro Auswertung her, sondern auch von der effektiven Größe des Suchraums.

Neben den Kosten sind bei vielen Problemen die Auswertungen zudem durch *Rauschen* beeinflusst. Bei physikalischen Experimenten stammt das Rauschen von Messfehlern, Umgebungsvariabilität oder Unterschieden zwischen Wiederholungen. Bei komplexen Simulationen kann *numerisches Rauschen* aus verschiedenen Ursachen wie internen stochastischen Phänomenen auftreten. Eine gängige Möglichkeit, diese Situation zu modellieren, besteht darin anzunehmen, dass die Beobachtung eine Störung der zugrunde liegenden Funktion ist:

$$
y_i = f(\bm{x}_i) + \varepsilon_i,
\qquad
\mathbb{E}[\varepsilon_i]=0,
\qquad
\mathrm{Var}(\varepsilon_i)=\sigma_\varepsilon^2.
$$

Diese beiden Einschränkungen (begrenztes Budget und/oder Rauschen) machen viele klassische Optimierer wenig geeignet: gradientenbasierte Methoden benötigen Ableitungen (die bei Black-Box-Funktionen normalerweise nicht verfügbar sind) und können unter Rauschen instabil werden. Folglich verwandelt sich das Problem in eine Suche unter Datenknappheit: Mit wenigen Auswertungen müssen wir (i) $f$ hinreichend annähern, um die Suche zu leiten, und (ii) adaptiv entscheiden, wo die nächsten Auswertungen investiert werden sollen, um die globale Lösung zu verbessern.

Diese Motivation führt auf natürliche Weise zum Ansatz der *Optimierung auf Basis von Surrogatmodellen* (SBO), der darin besteht, aus $\mathcal{D}_n$ ein statistisches Modell $\hat{f}$ zu konstruieren und es zu verwenden, um neue Auswertungen vorzuschlagen, wodurch die Anzahl der direkten Abfragen der kostspieligen Funktion drastisch reduziert wird (Forrester et al. 2008; Jiang et al. 2020). In den folgenden Abschnitten werden diese Modelle vorgestellt und ihre Einbindung erläutert.

### Surrogatmodelle: Definition und Taxonomie

In diesem Kontext der Optimierung mit kostspieligen Auswertungen wird ein *Surrogatmodell* (oder *Metamodell*) als eine aus einer endlichen Menge von Beobachtungen $\mathcal{D}_n=\{(\bm{x}_i,y_i)\}_{i=1}^n$ gelernte Näherung $\hat{f}$ definiert, mit dem Ziel, die Antwort des realen Systems zu einem deutlich geringeren Rechenaufwand nachzubilden, als es die direkte Auswertung der ursprünglichen Zielfunktion erfordern würde (Forrester et al. 2008; Jiang et al. 2020).
Ein Surrogatmodell kann zu zwei Hauptzwecken eingesetzt werden:

1.  **Schnelle Vorhersage und Analyse des Designraums.** Einmal trainiert, erlaubt $\hat{f}$ die Auswertung vieler Eingabekonfigurationen, um Tendenzen zu verstehen, Sensitivität zu analysieren oder interessante Regionen zu erkunden, ohne das kostspielige System auszuführen.

2.  **Optimierung mit begrenztem Budget.** Bei SBO wird das Surrogat verwendet, um die Auswahl neuer Auswertungen des realen Systems zu leiten, wodurch die Anzahl der kostspieligen Ausführungen reduziert wird, die notwendig sind, um sich der optimalen Lösung anzunähern (Jiang et al. 2020).

Es existiert eine große Vielfalt an Modellen, die als Surrogate dienen können. Eine praxisorientierte Taxonomie ist die folgende:

- **Nach der Natur der Vorhersage:**

**Deterministisch**

  Liefern eine Punktprognose $\hat{f}(\bm{x})$.

  **Probabilistisch**

  Quantifizieren Unsicherheit (z. B. mittels einer prädiktiven Verteilung). Diese Unterscheidung ist relevant, wenn adaptive Stichprobenkriterien auf Basis von Exploration/Exploitation gewünscht werden (Forrester et al. 2008; Jiang et al. 2020).

- **Nach der Flexibilität des Modells:**

  **Parametrisch**

  (z. B. polynomiale Regressionen). Einfach und interpretierbar, aber potenziell starr, wenn die Antwort Multimodalität oder starke Nichtlinearität aufweist.

  **Nicht-parametrisch oder von hoher Flexibilität**

  (z. B. Kernel-Methoden, Bäume, *Ensembles*). Fähig, komplexe Antworten zu erfassen, auf Kosten von mehr Parametern und dem Risiko der Überanpassung bei wenigen Daten (Forrester et al. 2008).

- **Nach der Art der Integration in die Optimierung:**

  **Offline**

  Ein einziges Training mit $\mathcal{D}_n$, das anschließend zum Vorschlagen von Kandidaten verwendet wird (nützlich, wenn das Gewinnen neuer Auswertungen kurzfristig nicht möglich ist).

  **Sequenziell/online**

  Das Surrogat wird iterativ aktualisiert, indem neue, gezielt ausgewählte Auswertungen hinzugefügt werden (infill/adaptive sampling), bis das Budget erschöpft ist oder ein Abbruchkriterium erreicht wird (Forrester et al. 2008; Jiang et al. 2020).

Trotz der Vielfalt der Modelle folgt der allgemeine Prozess der Surrogatmodellierung einem standardisierten Arbeitsablauf:

1.  **Variablen und Domäne definieren.** Es werden die Entscheidungsvariablen (Eingaben) und ihre Wertebereiche festgelegt, wodurch die Domäne $\mathcal{X}$ bestimmt wird.

2.  **Initiale Stichprobenziehung (DoE).** Es wird eine anfängliche Menge von Punkten $\{\bm{x}_i\}_{i=1}^n$ ausgewählt, die den Raum einigermaßen gleichmäßig abdeckt. Bei kontinuierlichen Problemen ist es üblich, *space-filling*-Stichprobenverfahren wie Latin Hypercube Sampling (LHS) oder SOBOL zu verwenden (Forrester et al. 2008).

3.  **Auswertung des realen Systems.** Die Ausgaben $y_i$ werden mit dem realen System berechnet, und der anfängliche Datensatz $\mathcal{D}_n$ wird erstellt.

4.  **Anpassung und Validierung des Surrogats.** Es wird eine Modellfamilie ausgewählt, ihre Parameter werden geschätzt, und ihre prädiktive Qualität wird mit gängigen Verfahren bewertet (z. B. Kreuzvalidierung), wobei Kompromisse zwischen Komplexität und Robustheit berücksichtigt werden (Forrester et al. 2008).

5.  **Sequenzielle Verfeinerung: adaptives Sampling oder *Infill*-Prozess.** Sofern das Budget es erlaubt, werden neue Punkte (*infill points*) hinzugefügt, die nach einem Kriterium ausgewählt werden, das Folgendes priorisiert: (i) Regionen, in denen das Modell wahrscheinlich ungenau ist (Exploration), und/oder (ii) vielversprechende Regionen in Bezug auf das Optimum (Exploitation). Dieser Zyklus wird wiederholt, bis ein Abbruchkriterium erreicht wird (basierend auf maximalem Budget, marginaler Verbesserung oder Zielgenauigkeit) (Forrester et al. 2008; Jiang et al. 2020).

In dieser Abschlussarbeit werden Surrogatmodelle vorrangig zu Optimierungszwecken unter begrenztem Budget eingesetzt, ohne dabei ihren Nutzen als Analyse- und Interpretationswerkzeug des Problems zu verlieren. Diese Idee wird in zwei komplementären Szenarien untersucht: einer Reihe von *Benchmarks*, bei denen die Zielfunktion kontrolliert ausgewertet werden kann, um den vollständigen sequenziellen Zyklus zu analysieren (von $n=0$ oder $n$ mit initialer Stichprobenziehung bis zu $B$ Stichproben), und einem *realen Fall*, bei dem das Modell mit den verfügbaren Daten trainiert wird, um seine Leistung zu bewerten und Kandidatenkonfigurationen $\bm{x}^\star$ für eine mögliche zukünftige Validierung vorzuschlagen.

### Der Gauß-Prozess als Surrogatmodell

Vorwiegend Rasmussen und Williams folgend (Rasmussen and Williams 2006), wird in diesem Unterabschnitt der GP als bayesianisches Modell der nicht-parametrischen Regression eingeführt. Sein Interesse im Kontext probabilistischer Surrogatmodelle liegt darin, dass er neben einer mittleren Vorhersage auch eine analytische Quantifizierung der Unsicherheit liefert, die für die bayesianische Optimierung (BO) wesentlich ist. Dazu wird zunächst die Definition des GP als Verteilung über Funktionen vorgestellt und anschließend seine Interpretation ausgehend von der bayesianischen linearen Regression im Merkmalsraum. Abbildung 2.1 veranschaulicht diese Idee, indem sie den Übergang von einer Prior-Verteilung über Funktionen zu einer Posterior-Verteilung nach Beobachtung von Daten zeigt.

![(a) Muestras de funciones generadas por el prior del GP. (b) Distribución posterior tras condicionar en dos observaciones; la línea negra representa la media predictiva y la banda sombreada el intervalo predictivo del 95 %.](/assets/articles/surrogate_models/fig_gp_prior_posterior.png)

#### Definition und Konsistenz

Formal ist ein GP ein stochastischer Prozess, der über dem Eingaberaum $\mathcal{X}$ definiert ist, sodass für jede endliche Menge von Punkten $X=\{x_1,\dots,x_n\}\subset\mathcal{X}$ der Vektor der Funktionswerte

$$
\mathbf{f}_X = \bigl(f(x_1),\dots,f(x_n)\bigr)^\top
$$

einer multivariaten Normalverteilung folgt. Folglich existieren eine Mittelwertfunktion $m:\mathcal{X}\to\mathbb{R}$ und eine Kovarianzfunktion $k:\mathcal{X}\times\mathcal{X}\to\mathbb{R}$, sodass

$$
\mathbf{f}_X \sim \mathcal{N}\!\bigl(\mathbf{m}_X,\mathbf{K}_{XX}\bigr),
$$

wobei

$$


\mathbf{m}_X =
\begin{bmatrix}
m(x_1)\\
\vdots\\
m(x_n)
\end{bmatrix},
\qquad
(\mathbf{K}_{XX})_{ij}=k(x_i,x_j).

$$

Abgekürzt schreibt man

$$
f(x)\sim \mathcal{GP}\!\bigl(m(x),k(x,x')\bigr).
$$

Der obige Ausdruck definiert den Prior des GP über jede endliche Menge von Punkten. Einen Gauß-Prozess zu spezifizieren bedeutet daher, zwei Objekte festzulegen: eine Mittelwertfunktion $m(x)$, die die global erwartete Tendenz der Funktion erfasst, und eine Kovarianzfunktion $k(x,x')$, auch Kernel genannt, die bestimmt, wie die Funktionswerte an verschiedenen Punkten der Domäne zueinander in Beziehung stehen.

Die Wahl der Mittelwertfunktion ist Teil der Prior-Spezifikation. Im Allgemeinen erfordert ein GP keinen Mittelwert null; eine übliche Wahl ist jedoch $m(x)=0.$ Diese Annahme ist sinnvoll, wenn keine klare Vorinformation über eine globale deterministische Tendenz vorliegt, da sie es erlaubt, den Großteil der Modellstruktur der Kovarianzfunktion zu überlassen. Zudem ergibt sie sich auf natürliche Weise, wenn man von einem zentrierten bayesianischen linearen Modell ausgeht,

$$
f(x)=\phi(x)^\top w,
\qquad
w\sim\mathcal{N}(0,\Sigma_p),
$$

dann erfüllt der induzierte Mittelwert über die Funktion

$$
\mathbb{E}[f(x)] = \phi(x)^\top\mathbb{E}[w] = 0.
$$

Gäbe es eine bekannte oder relevante Tendenz, könnte diese mittels einer von null verschiedenen Mittelwertfunktion oder mittels expliziter Basisfunktionen einbezogen werden.

Nachdem der Mittelwert festgelegt wurde, ist das zweite notwendige Element zur Definition des Priors die Kovarianzfunktion $k(x,x')$, die die in der vorherigen Gleichung definierte Kovarianzmatrix $\mathbf{K}_{XX}$ induziert, welche symmetrisch und positiv semidefinit sein muss, um eine gültige Kovarianz darzustellen. Intuitiv kodiert der Kernel die a-priori-Annahmen über die Regularität, Glattheit und Variationsskala der unbekannten Funktion.

Neben der Festlegung von Mittelwert und Kovarianz müssen die Gaußverteilungen, die verschiedenen endlichen Teilmengen der Domäne zugeordnet sind, untereinander kohärent sein. Diese Kohärenz ergibt sich aus einer fundamentalen Eigenschaft der multivariaten Gaußverteilung: der Konsistenz unter Marginalisierung.

**Proposition: Marginalisierung einer multivariaten Gaußverteilung.**

Sei

$$
\begin{bmatrix}
\mathbf{f}_A\\
\mathbf{f}_B
\end{bmatrix}
\sim
\mathcal{N}\!\left(
\begin{bmatrix}
\boldsymbol{\mu}_A\\
\boldsymbol{\mu}_B
\end{bmatrix},
\begin{bmatrix}
\Sigma_{AA} & \Sigma_{AB}\\
\Sigma_{BA} & \Sigma_{BB}
\end{bmatrix}
\right),
$$

wobei $\mathbf{f}_A$ und $\mathbf{f}_B$ zwei Teilvektoren des gemeinsamen Gauß-Vektors sind. Dann ist die Randverteilung von $\mathbf{f}_A$ gegeben durch

$$
\mathbf{f}_A \sim \mathcal{N}(\boldsymbol{\mu}_A,\Sigma_{AA}).
$$

In dieser Blockpartition ist $\Sigma_{AA}$ die Kovarianz-Untermatrix, die ausschließlich den Komponenten von $\mathbf{f}_A$ zugeordnet ist, während $\Sigma_{AB}$ und $\Sigma_{BA}$ die Kreuzkovarianzen zwischen $\mathbf{f}_A$ und $\mathbf{f}_B$ erfassen.

Diese Eigenschaft garantiert, dass die Inferenz, obwohl der GP über einer potenziell unendlichen Domäne definiert ist, exakt unter Verwendung nur einer endlichen Anzahl von Punkten durchgeführt werden kann. Insbesondere genügt es, die gemeinsame Verteilung der beobachteten Werte und gegebenenfalls der Vorhersagepunkte zu betrachten, da jede Teilmenge von Variablen des Prozesses weiterhin einer Gaußverteilung folgt, die mit der globalen Spezifikation kohärent ist.

Die vorangehende Definition führt den GP direkt im Funktionsraum ein (*function-space view*): Ein Prior wird über seine Mittelwert- und Kovarianzfunktion spezifiziert, und die Konsistenz unter Marginalisierung garantiert die Kohärenz aller induzierten endlichen Verteilungen. Diese Formulierung zeigt jedoch noch nicht klar, woher der Kernel stammt und wie das Lernen des Modells zu interpretieren ist. Dazu ist es hilfreich, zu einer vertrauteren Konstruktion zurückzukehren: der bayesianischen linearen Regression in einem Merkmalsraum, aus der der GP durch Marginalisierung der Gewichte hervorgeht.

#### Vom Gewichtsraum zum Funktionsraum

Um den Ursprung des Kernels und die Bedeutung des Lernens in einem GP genauer zu verstehen, ist es hilfreich, von der klassischen bayesianischen linearen Regression auszugehen und zu beobachten, wie man von einem durch explizite Parameter definierten Modell (*weight-space view*) zu einer Formulierung übergeht, die ausschließlich auf Kovarianzen zwischen Daten beruht (*function-space view*). Betrachten wir eine latente Funktion, die als lineare Kombination von Basisfunktionen modelliert wird:

$$
f(\bm{x}) = \bm{\phi}(\bm{x})^\top \bm{w},
$$

wobei $\bm{\phi}(\bm{x})\in\mathbb{R}^N$ den Merkmalsvektor darstellt, der der Eingabe $\bm{x}$ zugeordnet ist, und $\bm{w}\in\mathbb{R}^N$ der Gewichtsvektor ist. Anstatt anzunehmen, dass es einen einzigen wahren, unbekannten Wert von $\bm{w}$ gibt (frequentistischer Ansatz), führt der bayesianische Ansatz Unsicherheit über diese Parameter mittels eines Gauß-Priors ein: $\bm{w}\sim\mathcal{N}(\bm{0},\bm{\Sigma}_p).$ Die Beobachtungen werden nicht als gleich der latenten Funktion betrachtet, sondern als verrauschte Realisierungen derselben. Für jeden beobachteten Datenpunkt wird angenommen

$$
y_i = f(\bm{x}_i)+\varepsilon_i,
\qquad
\varepsilon_i\sim\mathcal{N}(0,\sigma_n^2),
$$

mit unabhängigem und identisch verteiltem Gauß-Rauschen. Definiert man $\bm{y}=[y_1,\dots,y_n]^\top$ und die Designmatrix im Merkmalsraum als

$$
\bm{\Phi}=
\begin{bmatrix}
\bm{\phi}(\bm{x}_1) & \cdots & \bm{\phi}(\bm{x}_n)
\end{bmatrix}
\in\mathbb{R}^{N\times n},
$$

dann lässt sich das probabilistische Modell für die Beobachtungen schreiben als

$$
\bm{y}\mid X,\bm{w}\sim\mathcal{N}(\bm{\Phi}^\top\bm{w},\sigma_n^2\bm{I}).
$$

In der Gewichtsraum-Ansicht entspricht das Lernen aus den Daten der bayesianischen Aktualisierung der Verteilung über $\bm{w}$. Wendet man die Bayes-Regel an,

$$
p(\bm{w}\mid X,\bm{y}) \propto p(\bm{y}\mid X,\bm{w})\,p(\bm{w}),
$$

und da sowohl die Likelihood als auch der Prior gaußförmig sind, ist auch die Posterior wieder gaußförmig. Konkret gilt

$$
p(\bm{w}\mid X,\bm{y})=
\mathcal{N}(\bar{\bm{w}},\bm{A}^{-1}),
\qquad
\bm{A}=\sigma_n^{-2}\bm{\Phi}\bm{\Phi}^\top+\bm{\Sigma}_p^{-1},
\qquad
\bar{\bm{w}}=\sigma_n^{-2}\bm{A}^{-1}\bm{\Phi}\bm{y}.
$$

Dieser Ausdruck fasst zusammen, was Lernen in der *weight-space view* bedeutet: Die Daten legen keinen einzigen Gewichtsvektor fest, sondern aktualisieren die Wahrscheinlichkeitsverteilung über diese. Der Posterior-Mittelwert $\bar{\bm{w}}$ stellt die plausibelste Lage der Gewichte nach Beobachtung der Daten dar, während $\bm{A}^{-1}$ die verbleibende Unsicherheit über diese Parameter quantifiziert.

Aus dieser Posterior lässt sich bereits eine prädiktive Verteilung für einen neuen Punkt $\bm{x}_*$ ableiten. Ist nämlich $f_* = f(\bm{x}_*)=\bm{\phi}(\bm{x}_*)^\top \bm{w},$ so erhält man die Prädiktive durch Marginalisierung der Unsicherheit über $\bm{w}$:

$$
p(f_*\mid \bm{x}_*,X,\bm{y})
=
\int p(f_*\mid \bm{x}_*,\bm{w})\,p(\bm{w}\mid X,\bm{y})\,d\bm{w}.
$$

Da $f_*$ eine lineare Transformation einer Gauß-Variablen ist, ist diese Verteilung ebenfalls gaußförmig:

$$
p(f_*\mid \bm{x}_*,X,\bm{y})
=
\mathcal{N}\!\Bigl(
\bm{\phi}(\bm{x}_*)^\top \bar{\bm{w}},
\;
\bm{\phi}(\bm{x}_*)^\top \bm{A}^{-1}\bm{\phi}(\bm{x}_*)
\Bigr).
$$

Diese Formulierung zeigt explizit, wie sich die Unsicherheit über die Gewichte auf die Vorhersage überträgt. Sie offenbart jedoch auch eine Einschränkung: Die Inferenz hängt von $\bm{A}^{-1}$ ab, einer Matrix der Größe $N\times N$, wobei $N$ die Dimension des Merkmalsraums ist. Ist dieser Raum sehr groß oder sogar unendlich, verliert diese Darstellung ihre Praktikabilität. Der entscheidende Schritt besteht dann darin, das Modell ausschließlich in Form von inneren Produkten zwischen Merkmalen umzuschreiben.

Tatsächlich ist unter dem Gauß-Prior über $\bm{w}$ die an einem beliebigen Punkt $\bm{x}$ ausgewertete latente Funktion ebenfalls eine gaußsche Zufallsvariable, da sie eine lineare Kombination von Gauß-Variablen ist. Daher ist für jede endliche Menge von Eingaben die gemeinsame Verteilung der Funktionswerte vollständig durch ihren Mittelwert und ihre Kovarianz bestimmt. Insbesondere gilt

$$
\mathbb{E}[f(\bm{x})]
=
\mathbb{E}[\bm{\phi}(\bm{x})^\top\bm{w}]
=
\bm{\phi}(\bm{x})^\top\mathbb{E}[\bm{w}]
=
0,
$$

und für zwei beliebige Punkte $\bm{x}$ und $\bm{x}'$

$$
\operatorname{cov}\bigl(f(\bm{x}),f(\bm{x}')\bigr)
=
\mathbb{E}\!\left[
\bm{\phi}(\bm{x})^\top \bm{w}\bm{w}^\top \bm{\phi}(\bm{x}')
\right]
=
\bm{\phi}(\bm{x})^\top \bm{\Sigma}_p \bm{\phi}(\bm{x}').
$$

Die Abhängigkeit von den Gewichten und ihrer Kovarianz lässt sich in einer einzigen, über Eingabepaaren definierten Funktion zusammenfassen: $k(\bm{x},\bm{x}') = \bm{\phi}(\bm{x})^\top \bm{\Sigma}_p \bm{\phi}(\bm{x}').$ So wird die ursprünglich über die Parameter $\bm{w}$ modellierte Unsicherheit auf die Kovarianzstruktur zwischen den Funktionswerten übertragen.

Aus dieser Perspektive kann für jede endliche Menge $X=\{\bm{x}_1,\dots,\bm{x}_n\}$ der Vektor der latenten Werte $\bm{f}_X = \bigl(f(\bm{x}_1),\dots,f(\bm{x}_n)\bigr)^\top$ durch Marginalisierung der Gewichte gewonnen werden:

$$
p(\bm{f}_X\mid X)
=
\int p(\bm{f}_X\mid X,\bm{w})\,p(\bm{w})\,d\bm{w}.
$$

Da $\bm{f}_X=\bm{\Phi}^\top \bm{w}$ eine lineare Transformation einer Gauß-Variablen ist, ist auch ihre Randverteilung gaußförmig, mit

$$
\mathbb{E}[\bm{f}_X]=\bm{0},
\qquad
\operatorname{cov}(\bm{f}_X)=\bm{\Phi}^\top \bm{\Sigma}_p \bm{\Phi}=\bm{K},
\quad \text{mit } K_{ij}=k(\bm{x}_i,\bm{x}_j).
$$

Somit gilt

$$
\bm{f}_X \sim \mathcal{N}(\bm{0}_X,\bm{K}_X).
$$

Das heißt, durch die Marginalisierung der Gewichte erhält man direkt eine Gaußverteilung über die Funktionswerte an jeder endlichen Menge von Eingaben. Dies ist genau die Formulierung des GP im Funktionsraum: Das Modell wird nicht mehr explizit in Form von $\bm{w}$ ausgedrückt, sondern ist ausschließlich durch seine Mittelwert- und seine Kovarianzfunktion bestimmt.

###### Der Kernel-Trick

Um den Kernel-Trick mit seiner funktionalen Interpretation zu verbinden, ist es hilfreich, auf die Theorie der Hilberträume mit reproduzierendem Kernel zurückzugreifen. In diesem Teil folgt man der Darstellung der Kernel-Methoden von Bishop (Bishop 2006) und der Interpretation von RKHS und Regularisierung von Hastie, Tibshirani und Friedman (Hastie et al. 2009).

Der vorherige Ausdruck $k(\bm{x},\bm{x}') = \bm{\phi}(\bm{x})^\top \bm{\Sigma}_p \bm{\phi}(\bm{x}')$ zeigt, dass die Kovarianz zwischen zwei Auswertungen der Funktion nicht explizit von den Gewichten $\bm{w}$ abhängt, sondern ausschließlich davon, wie sich die Eingaben $\bm{x}$ und $\bm{x}'$ im durch $\bm{\phi}$ induzierten und mit $\bm{\Sigma}_p$ gewichteten Merkmalsraum zueinander verhalten. Dies ist der Punkt, an dem der Kernel-Trick ins Spiel kommt.

Da $\bm{\Sigma}_p$ positiv semidefinit ist, kann sie mittels einer Matrixwurzel $\bm{\Sigma}_p^{1/2}$ geschrieben werden. Definiert man dann eine neue Merkmalsabbildung $\bm{\psi}(\bm{x})=\bm{\Sigma}_p^{1/2}\bm{\phi}(\bm{x}),$ so lässt sich die obige Kovarianz umschreiben als

$$
k(\bm{x},\bm{x}')
=
\bm{\psi}(\bm{x})^\top \bm{\psi}(\bm{x}').
$$

Der Kernel kann somit als inneres Produkt in einem transformierten Merkmalsraum interpretiert werden. Diese Beobachtung hat eine grundlegende Konsequenz: Wenn die Ausdrücke des Modells ausschließlich von inneren Produkten zwischen transformierten Eingaben abhängen, ist es nicht notwendig, den Merkmalsvektor $\bm{\phi}(\bm{x})$ explizit zu konstruieren oder zu manipulieren. Es genügt, über eine Funktion $k(\bm{x},\bm{x}')$ zu verfügen, die dieses innere Produkt direkt liefert.

Folglich kann das Modell implizit in Merkmalsräumen sehr hoher, sogar unendlicher Dimension operieren, ohne eine berechenbare Formulierung zu verlassen. Im Fall der Gaußprozesse bedeutet dies, dass sich Inferenz und Vorhersage ausschließlich in Bezug auf die Kernel-Matrix $\bm{K}$ ausdrücken lassen, deren Elemente $K_{ij}=k(\bm{x}_i,\bm{x}_j)$ sind. Die algebraische Komplexität hängt somit nicht mehr von der expliziten Konstruktion des Merkmalsraums ab, sondern verlagert sich auf die Handhabung von Kovarianzmatrizen, die über den beobachteten Daten definiert sind.

Aus konzeptioneller Sicht ist der Kernel-Trick nicht nur eine rechnerische Vereinfachung. Er erlaubt es auch, zwei Modellierungsebenen zu trennen: einerseits die Wahl einer Kernel-Funktion $k$, die Vorannahmen über Glattheit, Regularität oder Struktur der unbekannten Funktion kodiert; andererseits die konkrete Darstellung in Form von Basisfunktionen, die implizit bleiben kann. Diese Umformulierung rechtfertigt, dass, sobald der GP ausgehend von der bayesianischen linearen Regression eingeführt wurde, die weitere Analyse vollständig in Begriffen von Kovarianzfunktionen entwickelt werden kann.

###### RKHS und Repräsentationssatz.

Die vorherige Interpretation lässt sich mithilfe des Konzepts des Hilbertraums mit reproduzierendem Kernel präzisieren (*Reproducing Kernel Hilbert Space*, RKHS). Gegeben ein symmetrischer und positiv semidefiniter Kernel $k$, existiert ein Hilbertraum $\mathcal{H}_k$ bestehend aus Funktionen $f:\mathcal{X}\to\mathbb{R}$, in dem jeder Schnitt $k(\cdot,\bm{x})$ zum Raum gehört und die reproduzierende Eigenschaft erfüllt:

$$
f(\bm{x})=\langle f,\;k(\cdot,\bm{x})\rangle_{\mathcal{H}_k}.
$$

Darüber hinaus erfüllt der Kernel selbst

$$
k(\bm{x},\bm{x}')=
\langle k(\cdot,\bm{x}),\,k(\cdot,\bm{x}')\rangle_{\mathcal{H}_k}.
$$

Diese Eigenschaft erlaubt es, den Kernel als inneres Produkt in einem möglicherweise sehr hochdimensionalen oder unendlichdimensionalen Merkmalsraum zu interpretieren. Im ursprünglichen Raum $\mathcal{X}$ kann die Beziehung zwischen den Eingaben und der Ausgabe nichtlinear sein; durch eine geeignete Anhebung in einen reicheren Funktionsraum kann diese Beziehung jedoch mittels linearer Operationen in diesem neuen Raum behandelt werden. Im Rahmen von RKHS wird diese Anhebung kanonisch durch $\Phi_k(\bm{x})=k(\cdot,\bm{x})$ gegeben, sodass

$$
k(\bm{x},\bm{x}')=
\langle \Phi_k(\bm{x}),\Phi_k(\bm{x}')\rangle_{\mathcal{H}_k}.
$$

Der Kernel wirkt somit nicht nur als Kovarianzfunktion oder als implizites inneres Produkt, sondern auch als das Objekt, das die Geometrie des Funktionsraums definiert, in dem die Lösung lebt.

Diese Interpretation verstärkt die zentrale Idee des Kernel-Tricks: Es ist nicht notwendig, die Anhebung $\Phi_k(\bm{x})$ explizit zu konstruieren, die in der Praxis unbekannt oder nicht berechenbar sein kann, da der Kernel es erlaubt, direkt mit den relevanten inneren Produkten zu operieren. So kann das Modell nichtlineare Beziehungen im ursprünglichen Raum erfassen, während es implizit in einem reicheren Merkmalsraum arbeitet.

Das zentrale strukturelle Ergebnis in diesem Kontext ist der Repräsentationssatz (Schölkopf et al. 2001). Für ein regularisiertes Lernproblem der Form

$$
J[f]=Q\bigl(y_1,\dots,y_n,f(\bm{x}_1),\dots,f(\bm{x}_n)\bigr)
+\frac{\lambda}{2}\|f\|_{\mathcal{H}_k}^2,
$$

wobei $Q$ die Anpassung an die beobachteten Daten misst, kann jedes minimierende $f\in \mathcal{H}_k$ als endliche Kombination von Kernels geschrieben werden, die auf den Trainingsdaten zentriert sind:

$$
f(\bm{x})=\sum_{i=1}^n \alpha_i\,k(\bm{x},\bm{x}_i).
$$

Dieses Ergebnis ist besonders wichtig, weil es zeigt, dass, obwohl $\mathcal{H}_k$ unendlichdimensional sein kann, die durch einen endlichen Datensatz induzierte Lösung im von $\{k(\cdot,\bm{x}_1),\dots,k(\cdot,\bm{x}_n)\}$ erzeugten Unterraum enthalten ist.

Im Fall der Gaußprozesse taucht dieselbe Struktur im posterioren Vorhersagemittelwert wieder auf, der als lineare Kombination von Kernel-Auswertungen an den beobachteten Punkten geschrieben werden kann. So laufen die probabilistische Sichtweise des GP und die funktionale Sichtweise auf Basis von RKHS in derselben Idee zusammen: Das Lernen hängt von den Daten ausschließlich über die vom Kernel induzierte Geometrie ab.

#### Kovarianzfunktionen (Kernels)

Der Kernel kapselt die Vorannahmen über Glattheit, Skala und Korrelationsstruktur der latenten Funktion. Damit er eine gültige Kovarianzmatrix definiert, muss er symmetrisch und positiv semidefinit sein. In dieser Arbeit werden die folgenden Kernels betrachtet:

- **Linear (Dot Product).** Definiert als

  $$
  k_{\mathrm{lin}}(\bm{x},\bm{x}')
      =
      \sigma_0^2 + \bm{x}^\top \bm{x}'.
  $$

  Dieser Kernel entspricht einer bayesianischen linearen Regression im ursprünglichen Eingaberaum. Der Term $\bm{x}^\top\bm{x}'$ misst die lineare Ähnlichkeit zwischen zwei Punkten, während $\sigma_0^2\geq 0$ als Bias- oder Achsenabschnitts-Term wirkt und es erlaubt, die Funktion gegenüber dem Ursprung zu verschieben. Er ist nützlich, wenn erwartet wird, dass die Antwort einen global annähernd linearen Trend aufweist.

- **Squared Exponential (SE/RBF).** Ausgedrückt als

  $$
  k_{\mathrm{SE}}(\bm{x},\bm{x}')
      =
      \sigma_f^2
      \exp\left(
      -\frac{1}{2}(\bm{x}-\bm{x}')^\top
      \bm{M}
      (\bm{x}-\bm{x}')
      \right).
  $$

  Der Parameter $\sigma_f^2>0$ ist die Signalvarianz und steuert die vertikale Amplitude der vom Prior erzeugten Funktionen. Die Matrix $\bm{M}$ steuert die Variationsskala bezüglich der Eingabevariablen. Im isotropen Fall gilt $\bm{M}=\ell^{-2}\bm{I},$ wobei $\ell>0$ die charakteristische Längenskala oder *length-scale* ist.

  Im ARD-Fall (*Automatic Relevance Determination*) wird üblicherweise $\bm{M}=\mathrm{diag}(\ell_1^{-2},\dots,\ell_d^{-2})$ verwendet, was eine unterschiedliche Skala für jede Dimension erlaubt. Kleine Werte von $\ell$ ermöglichen schnelle Variationen der Funktion, während große Werte glattere Funktionen mit Korrelationen größerer Reichweite induzieren.

  Der SE/RBF-Kernel induziert unendlich oft differenzierbare Funktionen, was eine sehr starke Glattheitshypothese darstellt. Diese Eigenschaft kann für sehr regelmäßige Antworten angemessen sein, kann sich aber auch bei physikalischen Phänomenen mit lokalen Änderungen, Unregelmäßigkeiten oder geringerem Grad an Differenzierbarkeit als übermäßig erweisen, wo das Modell die tatsächliche Antwort übermäßig glätten kann.

- **Matérn.** Der Matérn-Kernel verallgemeinert den SE/RBF-Kernel, indem er einen Glattheitsparameter $\nu>0$ einführt, der den Grad der Differenzierbarkeit der erzeugten Funktionen steuert. Bezeichnet $r=\|\bm{x}-\bm{x}'\|$ den euklidischen Abstand zwischen zwei Eingaben und $\ell>0$ die charakteristische Längenskala, so wird der Matérn-Kernel definiert als

  $$

      k_{\text{Mat\'ern}}(r)=
      \sigma_f^2
      \frac{2^{1-\nu}}{\Gamma(\nu)}
      \left(\frac{\sqrt{2\nu}\, r}{\ell}\right)^\nu
      K_\nu\left(\frac{\sqrt{2\nu}\, r}{\ell}\right).

  $$

  In diesem Ausdruck ist $\Gamma(\nu)$ die Gammafunktion, die die Fakultät auf positive reelle Werte verallgemeinert, und $K_\nu(\cdot)$ ist die modifizierte Bessel-Funktion zweiter Art (NIST Digital Library of Mathematical Functions 2010). Wie beim SE-Kernel steuert $\sigma_f^2$ die Amplitude der Funktion und $\ell$ bestimmt die räumliche Korrelationsskala. Diese Parameter werden nicht über eine einzige geschlossene Formel berechnet, sondern als Kernel-Parameter behandelt und anhand der Daten angepasst, zum Beispiel durch Maximierung der marginalen Log-Likelihood (LML).

  Die Fälle $\nu=3/2$ und $\nu=5/2$ sind besonders gebräuchlich, weil sie einfache geschlossene Ausdrücke liefern und es erlauben, die Glattheit realistischer zu steuern als beim SE-Kernel. Insbesondere induziert $\nu=3/2$ weniger glatte Funktionen, während $\nu=5/2$ regelmäßigere Funktionen erlaubt, ohne jedoch die unendliche Glattheit des SE/RBF-Kernels aufzuzwingen. Deshalb eignet sich der Matérn-Kernel in der Regel, wenn man eine stetige, aber nicht notwendigerweise unendlich oft differenzierbare Funktion modellieren möchte.

- **Weißes Rauschen (White Kernel).**

  $$
  k_{\mathrm{White}}(\bm{x},\bm{x}')
      =
      \sigma_n^2 \delta_{\bm{x},\bm{x}'}.
  $$

  In diesem Ausdruck stellt $\sigma_n^2>0$ die Varianz des Beobachtungsrauschens dar, und $\delta_{\bm{x},\bm{x}'}$ ist das Kronecker-Delta, das $1$ ist, wenn die Eingaben übereinstimmen, und sonst $0$. Dieser Kernel modelliert keine glatte Struktur der latenten Funktion, sondern unabhängige Variabilität, die mit den Beobachtungen verbunden ist.

  Schreibt man das beobachtete Modell als

  $$
  y_i = f(\bm{x}_i)+\varepsilon_i,
      \qquad
      \varepsilon_i\sim\mathcal{N}(0,\sigma_n^2),
  $$

  dann ist die Kovarianz der Beobachtungen nicht nur die Kovarianz der latenten Funktion, sondern

  $$
  \operatorname{cov}(y_i,y_j)
      =
      k_f(\bm{x}_i,\bm{x}_j)
      +
      \sigma_n^2\delta_{ij}.
  $$

  Daher wird der Weißes-Rauschen-Kernel üblicherweise zu einem anderen Kernel, wie RBF oder Matérn, hinzugefügt, um darzustellen, dass jede Beobachtung Messfehler oder unabhängiges numerisches Rauschen enthalten kann, ohne räumliche Korrelation zwischen unterschiedlichen Punkten einzuführen.

**Isotropie und Automatic Relevance Determination (ARD).**
Ein Kernel heißt *isotrop*, wenn die Korrelation nur vom euklidischen Abstand $\|\bm{x}-\bm{x}'\|$ abhängt, wobei angenommen wird, dass die Funktion in allen Richtungen gleich variiert. In technischen Problemstellungen ist es jedoch üblich, dass manche Eingabevariablen die Antwort viel stärker beeinflussen als andere.

Um dies zu erfassen, verwenden wir **ARD**, indem wir die Skalenmatrix $\bm{M} = \mathrm{diag}(\ell_1^{-2}, \dots, \ell_d^{-2})$ definieren. Die Parameter $\ell_d$ sind die **charakteristischen Längenskalen** (length-scales), die eine direkte physikalische Interpretation zulassen:

- Ist $\ell_d$ klein, variiert die Funktion in dieser Dimension schnell (sehr relevante Variable).

- Ist $\ell_d$ sehr groß ($\ell_d \to \infty$), ist die Funktion in dieser Richtung nahezu konstant (irrelevante Variable), wodurch die Abhängigkeit effektiv eliminiert wird.

#### Vorhersage und Unsicherheit

Gegeben eine Menge verrauschter Beobachtungen $\mathcal{D}_n=\{(\bm{x}_i,y_i)\}_{i=1}^n,$ wird das Beobachtungsmodell

$$
y_i=f(\bm{x}_i)+\varepsilon_i,
\qquad
\varepsilon_i\sim\mathcal{N}(0,\sigma_n^2),
$$

mit unabhängigem gaußschem Rauschen angenommen. In diesem Unterabschnitt wird die Vorhersage des latenten Funktionswerts an einem neuen Testpunkt $\bm{x}_*$ betrachtet, den wir mit $f_* := f(\bm{x}_*)$ bezeichnen.

Der Einfachheit halber, und in Übereinstimmung mit der in Abschnitt 2.2.1 eingeführten Begründung für den Mittelwert null, wird $m(\bm{x})=0$ angenommen. Diese Annahme kann gelockert werden, indem eine Mittelwertfunktion ungleich null einbezogen wird, wenn Vorinformationen über einen globalen Trend der Funktion vorliegen. Unter dieser Konvention und unter Verwendung der Konsistenz der endlichen gaußschen Verteilungen, die in der vorherigen Proposition beschrieben wurde, ist die gemeinsame Verteilung der Beobachtungen $\bm{y}$ und des latenten Werts $f_*$ gegeben durch

$$

\begin{bmatrix}
\bm{y} \\
f_*
\end{bmatrix}
\sim
\mathcal{N}\left(
\bm{0},
\begin{bmatrix}
\bm{K}+\sigma_n^2\bm{I} & \bm{k}_* \\
\bm{k}_*^\top & k_{**}
\end{bmatrix}
\right).


$$

In diesem Ausdruck ist $\bm{K}=K(X,X)\in\mathbb{R}^{n\times n}$ die Kernel-Matrix zwischen den Trainingspunkten, mit Einträgen $K_{ij}=k(\bm{x}_i,\bm{x}_j).$ Der Vektor $\bm{k}_*=K(X,\bm{x}_*)\in\mathbb{R}^n$ fasst die Kovarianzen zwischen den beobachteten Punkten und dem Testpunkt zusammen:

$$
\bm{k}_*
=
\begin{bmatrix}
k(\bm{x}_1,\bm{x}_*) \\
\vdots \\
k(\bm{x}_n,\bm{x}_*)
\end{bmatrix},
$$

während $k_{**}=k(\bm{x}_*,\bm{x}_*)$ die A-priori-Varianz der latenten Funktion an $\bm{x}_*$ ist. Der Term $\sigma_n^2\bm{I}$ erscheint nur im Block, der den Beobachtungen $\bm{y}$ entspricht, da diese Messrauschen enthalten, während $f_*$ den nicht verrauschten latenten Wert der Funktion darstellt.

Wendet man die Konditionierungsregeln einer multivariaten Gaußverteilung an (Rasmussen and Williams 2006), erhält man die posteriore prädiktive Verteilung

$$
p(f_*\mid X,\bm{y},\bm{x}_*)
=
\mathcal{N}(\bar{f}_*,\mathbb{V}[f_*]),
$$

wobei

$$
\begin{aligned}
\bar{f}_*
&=
\bm{k}_*^\top
(\bm{K}+\sigma_n^2\bm{I})^{-1}
\bm{y},

\\
\mathbb{V}[f_*]
&=
k_{**}
-
\bm{k}_*^\top
(\bm{K}+\sigma_n^2\bm{I})^{-1}
\bm{k}_*.

\end{aligned}
$$

Die erste Gleichung liefert den posterioren prädiktiven Mittelwert, das heißt die durchschnittliche Schätzung des latenten Funktionswerts an $\bm{x}_*$ unter dem Modell. Die zweite quantifiziert die posteriore Unsicherheit über diesen latenten Wert. Es ist wichtig, diese posteriore Unsicherheit über $f_*$ von der Rauschvarianz $\sigma_n^2$ zu unterscheiden. Erstere spiegelt die Unsicherheit des Modells über die latente Funktion wider und kann durch die Einbeziehung informativer Beobachtungen reduziert werden. Wenn die Beobachtungen jedoch sehr verrauscht sind, kann diese Unsicherheit selbst in bereits ausgewerteten Bereichen hoch bleiben, insbesondere wenn keine ausreichenden Wiederholungen vorliegen oder das Rauschen das Signal dominiert. Wollte man eine neue verrauschte Beobachtung $y_*$ vorhersagen, und nicht nur den latenten Wert $f_*$, müsste die prädiktive Varianz zusätzlich den Rauschterm $\sigma_n^2$ enthalten.

#### Training: Parameterinferenz

In parametrischen Modellen mit Punktschätzung besteht das Training in der Regel darin, einen konkreten Wert der Modellparameter auszuwählen, etwa mittels Maximum-Likelihood-Schätzung (MLE) oder durch die Minimierung einer regularisierten Verlustfunktion. Aus bayesianischer Perspektive dagegen können die Parameter als Zufallsvariablen behandelt werden: Man definiert einen Prior über sie und erhält nach Beobachtung der Daten eine Posterior-Verteilung. Möchte man anschließend einen einzigen repräsentativen Wert dieser Parameter auswählen, ist eine mögliche Option der MAP-Schätzer. Bei der Regression mit Gauß-Prozessen ist es jedoch üblich, die Gewichte oder latenten Werte zu marginalisieren und die Parameter durch Maximierung der marginalen Likelihood anzupassen.

Im Fall der Gauß-Prozesse nimmt das Training eine andere Form an als bei klassischen parametrischen Modellen. Wie in Abschnitt 2.2.2.0.1 gezeigt wurde, werden die Gewichte $\bm{w}$ der Darstellung im Merkmalsraum marginalisiert, und ihre Wirkung wird stattdessen im Kernel kodiert. Das Training eines GP besteht daher nicht darin, eine Kovarianzfamilie festzulegen und die sie definierenden Hyperparameter zusammen mit dem Rauschniveau des Modells zu schätzen (Rasmussen and Williams 2006). In der Praxis besteht eine übliche Strategie darin, die marginale Likelihood der Daten zu maximieren,

$$
\hat{\bm{\theta}}
=
\arg\max_{\bm{\theta}}
\log p(\bm{y}\mid X,\bm{\theta}).
$$

Dieses Verfahren wird als Maximierung der marginalen Likelihood oder *type-II maximum likelihood* bezeichnet. Wird zusätzlich ein expliziter Prior $p(\bm{\theta})$ über die Parameter eingeführt, dann wäre das entsprechende Kriterium ein MAP über $\bm{\theta}$:

$$
\hat{\bm{\theta}}_{\mathrm{MAP}}
=
\arg\max_{\bm{\theta}}
\left[
\log p(\bm{y}\mid X,\bm{\theta})
+
\log p(\bm{\theta})
\right].
$$

In dieser Arbeit wird die Standardformulierung auf Basis der LML vorgestellt, ohne einen zusätzlichen Prior über $\bm{\theta}$ einzuführen. Die LML misst die Wahrscheinlichkeit, die Trainingsdaten unter dem durch die Parameter $\bm{\theta}$ definierten Modell zu beobachten. Bei der GP-Regression mit gaußschem Rauschen ergibt sie sich durch Marginalisierung der latenten Funktionswerte:

$$
p(\bm{y}\mid X,\bm{\theta})
=
\int p(\bm{y}\mid \bm{f},X,\bm{\theta})p(\bm{f}\mid X,\bm{\theta})\,d\bm{f}.
$$

Da sowohl der Prior über $\bm{f}$ als auch die Likelihood gaußsch sind, lässt sich dieses Integral analytisch lösen und führt zu:

$$

\log p(\bm{y}\mid X,\bm{\theta})
=
\underbrace{-\frac{1}{2}\bm{y}^\top \bm{K}_y^{-1}\bm{y}}_{\text{Ajuste a los datos}}
\quad
\underbrace{-\frac{1}{2}\log|\bm{K}_y|}_{\text{Penalizacion por complejidad}}
\quad
\underbrace{-\frac{n}{2}\log 2\pi}_{\text{Constante}},


$$

wobei $\bm{K}_y=\bm{K}_f+\sigma_n^2\bm{I}$ die Kovarianzmatrix der verrauschten Beobachtungen ist.

Die drei Terme des obigen Ausdrucks haben komplementäre Interpretationen:

1.  **Datenanpassung.** Der Term $-\frac{1}{2}\bm{y}^\top \bm{K}_y^{-1}\bm{y}$ misst die Kompatibilität zwischen den Beobachtungen und der durch den Kernel induzierten Kovarianzstruktur. Er kann als Mahalanobis-Distanz zum Mittelwert des Modells interpretiert werden. Er bestraft Parameterkonfigurationen, unter denen die beobachteten Daten unwahrscheinlich sind.

2.  **Komplexitätsstrafe.** Der Term $-\frac{1}{2}\log|\bm{K}_y|$ hängt von der Kovarianzmatrix und damit von den Parametern des Kernels und des Rauschens ab. Er wirkt als Komplexitätsstrafe: Zu flexible Modelle können vielen möglichen Mustern Wahrscheinlichkeit zuweisen und dabei die Wahrscheinlichkeit verwässern, die dem konkreten Satz beobachteter Daten zugewiesen wird. Umgekehrt können übermäßig starre Modelle die Daten nicht gut erklären.

3.  **Normalisierungskonstante.** Der Term $-\frac{n}{2}\log 2\pi$ hängt ausschließlich von der Anzahl der Beobachtungen ab und beeinflusst die Optimierung von $\bm{\theta}$ nicht, obwohl er Teil der vollständigen gaußschen Dichte ist.

Diese Zerlegung steht im Zusammenhang mit dem Kompromiss zwischen Anpassung und Modellkomplexität, sollte jedoch nicht wörtlich mit der klassischen Bias-Varianz-Rausch-Zerlegung verwechselt werden. Beim GP erscheint das Rauschen explizit in $\bm{K}_y$ über $\sigma_n^2\bm{I}$, während die Flexibilität des Modells durch die Kernel-Parameter kontrolliert wird.

**Interpretation als bayesianische Regularisierung**
Die Maximierung der LML integriert auf natürliche Weise eine Form bayesianischer Regularisierung. Der Anpassungsterm bevorzugt Parameter, die die Beobachtungen gut erklären, während der mit der Determinante verbundene Term Modelle bestraft, deren Flexibilität die Wahrscheinlichkeitsmasse auf zu viele mögliche Konfigurationen verteilt. Dieser Effekt hängt mit dem sogenannten bayesianischen Ockhamschen Rasiermesser zusammen: Unter Modellen, die in der Lage sind, die Daten zu erklären, tendiert die Evidenz dazu, jene zu bevorzugen, die dies ohne unnötige Komplexität tun.

Diese Regularisierung sollte jedoch nicht als absolute Garantie gegen Overfitting oder als universeller Ersatz für die Modellvalidierung interpretiert werden. Die LML bewertet die Daten unter den Annahmen des GP und des gewählten Kernels. Ist die Kernelfamilie falsch spezifiziert, können andere Kriterien wie die Kreuzvalidierung oder der Vergleich mit alternativen Kernels notwendig sein, um die Vorhersagefähigkeit des Modells zu bewerten. Zudem kann die LML-Oberfläche mehrere lokale Optima aufweisen, sodass das Ergebnis von der Initialisierung des Optimierers und der Parametrisierung des Kernels abhängen kann.

Es sei angemerkt, dass Regularisierung nicht ausschließlich einem frequentistischen Ansatz vorbehalten ist. Zum Beispiel kann die Ridge-Regression sowohl als Minimierung eines bestraften Verlusts als auch als MAP-Schätzer einer bayesianischen linearen Regression mit gaußschem Prior über die Gewichte betrachtet werden. Wenn nämlich $p(\bm{w})=\mathcal{N}(\bm{0},\sigma_w^2\bm{I})$ gilt und das Beobachtungsrauschen die Varianz $\sigma_D^2$ hat, führt der MAP zu einer Strafe

$$
\lambda\|\bm{w}\|^2,
\qquad
\lambda=\frac{\sigma_D^2}{\sigma_w^2}.
$$

Der Regularisierungsterm kann somit als eine a-priori-Präferenz für Modelle mit kleinen Gewichten interpretiert werden. Der Unterschied beim GP besteht nicht darin, dass jede Form von Regularisierung verschwindet, sondern darin, dass diese in die Wahl des funktionalen Priors, das heißt in den Kernel und seine Parameter, integriert wird.

### Versuchsplanung und initiales Sampling

Die Leistung der SBO hängt in großem Maße von der Qualität der anfänglichen Menge an Beobachtungen $\mathcal{D}_n$ ab. Konzentrieren sich die ersten Punkte auf einen kleinen Bereich der Domäne $\mathcal{X}\subset\mathbb{R}^d$, verfügt das Ersatzmodell über wenig Information über den Rest des Suchraums. Bei einem GP wirkt sich dies direkt auf den Mittelwert und die prädiktive Varianz aus, die durch eine schlechte anfängliche Abdeckung stark beeinflusst werden können. Deshalb ist es üblich, vor Beginn der sequenziellen Optimierungsphase DoE-Strategien einzusetzen, die darauf abzielen, die Domäne repräsentativ abzudecken.

Die uniforme Zufallsstichprobe ist die einfachste Alternative, kann aber lokale Ballungen und wenig erkundete Regionen erzeugen, insbesondere wenn das Auswertungsbudget begrenzt ist. Um dieses Problem zu verringern, werden *space-filling*-Designs verwendet, deren Ziel es ist, die Anfangspunkte gleichmäßiger innerhalb der Domäne zu verteilen (Forrester et al. 2008). Zu den gängigsten Alternativen gehören das *Latin Hypercube Sampling* (LHS) und Sobol-Sequenzen.

Das *Latin Hypercube Sampling* teilt den Bereich jeder Variable in $n$ Intervalle gleicher Wahrscheinlichkeit und wählt die Punkte so aus, dass in jeder Dimension genau eine Stichprobe pro Intervall vorhanden ist. Dies garantiert eine gute marginale Schichtung, jedoch nicht notwendigerweise eine optimale geometrische Abdeckung im mehrdimensionalen Raum.

In dieser Arbeit werden Sobol-Sequenzen verwendet, eine Familie von *quasi-Monte-Carlo*-Sequenzen niedriger Diskrepanz (Sobol' 1967). Im Gegensatz zum pseudozufälligen Sampling sind diese Sequenzen deterministisch und darauf ausgelegt, den Einheitswürfel gleichmäßiger abzudecken. Die Diskrepanz misst anschaulich den Unterschied zwischen der empirischen Verteilung der erzeugten Punkte und einer idealen Gleichverteilung; eine Sequenz niedriger Diskrepanz tendiert daher dazu, weniger Lücken und weniger Ballungen zu hinterlassen als ein rein zufälliges Sampling.

Diese Eigenschaft ist bei der anfänglichen Konstruktion des Ersatzmodells nützlich, da sie eine erste stabile Abdeckung der Domäne ermöglicht, bevor adaptive Kriterien zur Auswahl neuer Auswertungen angewendet werden. Da sie zudem deterministisch sind, erleichtern Sobol-Sequenzen die Reproduzierbarkeit des Anfangsdesigns.

Bezüglich der Größe des Anfangsdesigns ist eine übliche Faustregel bei der Optimierung mit Ersatzmodellen, eine zur Dimension des Problems proportionale Anzahl von Punkten zu wählen, zum Beispiel $n=10d$, wobei $d$ die Dimension des Eingabevektors ist (Jones et al. 1998). Bei Problemen mit sehr teuren Auswertungen kann diese Zahl reduziert werden, um mehr Budget für die sequenzielle Optimierungsphase zu reservieren.

### Auswahl neuer Auswertungen

Sobald ein anfängliches Ersatzmodell aus dem DoE konstruiert wurde, stellt sich die Herausforderung, wie es genutzt werden kann, um das globale Optimum des ursprünglichen Problems zu finden. In der Literatur gibt es drei Hauptansätze für diese Aufgabe (Jiang et al. 2020):

1.  **Reine Heuristiken:** Algorithmen wie genetische oder evolutionäre Algorithmen, die die Black-Box-Funktion direkt auswerten. Sie erfordern Tausende von hochgenauen Auswertungen, was bei strengen Budgets nicht durchführbar ist.

2.  **Von Ersatzmodellen unterstützte Metaheuristiken:** Nutzen das Ersatzmodell, um Individuen innerhalb einer evolutionären Population vorzubewerten und zu filtern, bevor die reale Simulation ausgeführt wird. Obwohl sie die Kosten senken, erfordert ihre populationsbasierte Natur weiterhin eine beträchtliche Anzahl von Stichproben.

3.  **Optimierung basierend auf Ersatzmodellen (SBO):** Dies ist der effizienteste Ansatz für teure Probleme. Er stützt sich auf ein probabilistisches Modell, um die Suche sequenziell zu leiten, wobei in jeder Iteration ein einzelner Punkt (oder eine kleine Charge) ausgewertet wird, um den gewonnenen Informationsgehalt zu maximieren.

#### Ansatz: sequenzielle Optimierung mit begrenztem Budget

Der Arbeitsablauf bei der SBO, und insbesondere bei der BO, steht in engem Zusammenhang mit Paradigmen des maschinellen Lernens wie dem *Active Learning*, bei dem das Modell aktiv auswählt, welche neuen Beobachtungen am informativsten wären, um sein Lernen zu verbessern (Settles 2009). Im Kontext der Optimierung mit Ersatzmodellen wird dies jedoch üblicherweise spezifischer als *Infill*-Prozess bezeichnet, da das Ziel nicht ausschließlich darin besteht, die globale Genauigkeit des Modells zu verbessern, sondern die Suche nach dem Optimum unter einem begrenzten Auswertungsbudget $B$ durch Optimierung einer *Akquisitionsfunktion* zu leiten.

Diese Funktion, die viel günstiger auszuwerten ist, entscheidet auf intelligente Weise, welcher Kandidat $x^*$ optimal für die nächste Iteration ist. Sobald er bestimmt ist, wird er im realen (teuren) System ausgewertet, zu $\mathcal{D}_n$ hinzugefügt, und das Ersatzmodell wird neu trainiert (wobei seine Parameter aktualisiert werden). Dieser Zyklus wird wiederholt, bis das Auswertungsbudget $B$ erschöpft ist oder ein Konvergenzkriterium erfüllt wird.

#### Exploration vs. Exploitation

Das Design der Akquisitionsfunktion definiert das Verhalten des Algorithmus, wobei zwei gegensätzliche Kräfte ausgeglichen werden müssen (Jiang et al. 2020):

- **Exploitation (lokale Suche):** Priorisierung der Regionen des Suchraums, in denen das Ersatzmodell vorhersagt, dass der Wert der Zielfunktion sehr gut ist (minimal, im Fall der Minimierung). Eine rein exploitative Strategie konvergiert schnell, ist aber stark anfällig dafür, in lokalen Minima gefangen zu bleiben.

- **Exploration (globale Suche):** Priorisierung von Regionen mit hoher Unsicherheit, das heißt, wo das Modell wenige Daten hat und die prädiktive Varianz hoch ist. Eine rein explorative Strategie verhält sich wie ein pseudozufälliges Sampling und verschwendet Auswertungen in wenig vielversprechenden Bereichen.

#### Bayesianische Optimierung und Akquisitionsfunktionen

Bei der BO erfolgt die Auswahl neuer Auswertungen mittels Akquisitionsfunktionen, die die prädiktive Verteilung des Ersatzmodells nutzen, um zu entscheiden, welcher Punkt als Nächstes ausgewertet werden sollte (Jones et al. 1998). Obwohl es verschiedene Kriterien gibt, wie *Probability of Improvement* (PI), *EI* oder *Lower Confidence Bound* (LCB), konzentrieren wir uns in dieser Arbeit auf *EI*, eines der am häufigsten in der BO verwendeten Kriterien wegen seiner Fähigkeit, Exploitation und Unsicherheit explizit zu kombinieren.

**Probability of Improvement (PI)**
Das PI-Kriterium war eines der ersten entwickelten. Sein Ziel ist es, die Wahrscheinlichkeit zu maximieren, dass ein neuer Punkt $\bm{x}$ die bisher beste beobachtete Lösung $f_{\min}$ verbessert. Nimmt man an, dass die Vorhersage bei $\bm{x}$ einer Normalverteilung $\mathcal{N}(\mu(\bm{x}), \sigma^2(\bm{x}))$ folgt, ist die Verbesserungswahrscheinlichkeit:

$$
PI(\bm{x})
=
P(Y(\bm{x}) < f_{\min})
=
\Phi\left(
\frac{f_{\min}-\mu(\bm{x})}{\sigma(\bm{x})}
\right),
$$

wobei $\Phi$ die kumulative Verteilungsfunktion der Standardnormalverteilung ist.

Die Hauptbeschränkung von PI besteht darin, dass es nur die Wahrscheinlichkeit einer Verbesserung des aktuellen Wertes berücksichtigt, nicht aber das erwartete Ausmaß dieser Verbesserung. Deshalb kann es Punkten mit hoher Wahrscheinlichkeit einer sehr kleinen Verbesserung einen zu hohen Wert zuweisen, was die Suche zu einer übermäßigen lokalen Exploitation verzerrt.

**Expected Improvement (EI)**
Der *Expected Improvement* (erwartete Verbesserung) löst diese Einschränkung, indem er nicht nur berücksichtigt, ob ein Punkt den besten beobachteten Wert verbessern kann, sondern auch, um wie viel man erwartet, dass er ihn verbessert (Jones et al. 1998). Für ein Minimierungsproblem definieren wir die Verbesserungsfunktion gegenüber der aktuell besten Beobachtung $f_{\min}$ als:

$$
I(\bm{x})
=
\max(0, f_{\min}-Y(\bm{x})).
$$

Äquivalent lässt sich die stückweise Funktion schreiben als:

$$
I(\bm{x})
=
\begin{cases}
f_{\min}-Y(\bm{x}), & \text{si } Y(\bm{x})<f_{\min},\\
0, & \text{si } Y(\bm{x})\geq f_{\min}.
\end{cases}
$$

Da $Y(\bm{x})$ eine durch den GP modellierte Zufallsvariable ist, erhält man die erwartete Verbesserung durch Integration der Verbesserung über die prädiktive Dichte:

$$
\mathbb{E}[I(\bm{x})]
=
\int_{-\infty}^{f_{\min}}
(f_{\min}-y)
\frac{1}{\sigma(\bm{x})}
\phi\left(
\frac{y-\mu(\bm{x})}{\sigma(\bm{x})}
\right)
\,dy,
$$

wobei $\phi(\cdot)$ die Dichtefunktion der Standardnormalverteilung ist. Führt man die Variablensubstitution

$$
z=\frac{y-\mu(\bm{x})}{\sigma(\bm{x})},
\qquad
dy=\sigma(\bm{x})\,dz,
$$

durch, wird die obere Grenze zu $Z=\frac{f_{\min}-\mu(\bm{x})}{\sigma(\bm{x})}.$ Somit gilt:

$$
\mathbb{E}[I(\bm{x})]
=
\int_{-\infty}^{Z}
\left(f_{\min}-\mu(\bm{x})-\sigma(\bm{x})z\right)\phi(z)\,dz.
$$

Trennt man die Terme:

$$
\mathbb{E}[I(\bm{x})]
=
(f_{\min}-\mu(\bm{x}))\int_{-\infty}^{Z}\phi(z)\,dz
-
\sigma(\bm{x})\int_{-\infty}^{Z}z\phi(z)\,dz.
$$

Da

$$
\int_{-\infty}^{Z}\phi(z)\,dz=\Phi(Z),
\qquad
\int_{-\infty}^{Z}z\phi(z)\,dz=-\phi(Z),
$$

erhält man die geschlossene Formulierung:

$$


\mathbb{E}[I(\bm{x})] =
\begin{cases}
(f_{\min} - \mu(\bm{x})) \Phi(Z) + \sigma(\bm{x}) \phi(Z),
& \text{si } \sigma(\bm{x}) > 0, \\[0.2em]
0,
& \text{si } \sigma(\bm{x}) = 0,
\end{cases}
\qquad
Z = \frac{f_{\min} - \mu(\bm{x})}{\sigma(\bm{x})}.

$$

Der geschlossene Ausdruck von EI kombiniert explizit die beiden Suchstrategien:

- Der erste Term, $(f_{\min} - \mu(\bm{x})) \Phi(Z)$, dominiert, wenn das Modell einen niedrigen Wert von $\mu(\bm{x})$ vorhersagt, also ein besseres erwartetes Ergebnis in einem Minimierungsproblem. Dieser Term treibt die **Exploitation** vielversprechender Zonen voran.

- Der zweite Term, $\sigma(\bm{x}) \phi(Z)$, dominiert, wenn die Unsicherheit $\sigma(\bm{x})$ groß ist, wodurch die **Exploration** von Regionen begünstigt wird, in denen das Modell noch wenig Information besitzt.

Mathematisch lässt sich diese Synthese der Strategien analytisch belegen, indem man die partiellen Ableitungen der EI-Funktion bezüglich der prädiktiven Momente des Modells berechnet. Einerseits ist die Ableitung nach dem prädiktiven Mittelwert:

$$
\frac{\partial E[I(\bm{x})]}{\partial \mu(\bm{x})} = -\Phi(Z)
$$

Da die kumulative Verteilungsfunktion $\Phi(Z)$ stets streng positive Werte annimmt, ist diese Ableitung immer negativ. Dies zeigt, dass die EI-Funktion bezüglich $\mu(\bm{x})$ **monoton fallend** ist. In der Praxis bedeutet dies: Je niedriger der vom Modell für eine Region vorhergesagte Wert ist – und somit besser bei einem Minimierungsproblem –, desto größer wird die erwartete Verbesserung, was den Algorithmus dazu treibt, vielversprechende Zonen zu *exploitieren*.

Andererseits ergibt sich für die Ableitung nach der prädiktiven Unsicherheit:

$$
\frac{\partial E[I(\bm{x})]}{\partial \sigma(\bm{x})} = \phi(Z).
$$

Da die Dichtefunktion der Standardnormalverteilung $\phi(Z)$ stets größer als null ist, ist diese Ableitung strikt positiv. Folglich ist EI bezüglich $\sigma(\bm{x})$ **monoton steigend**. Die praktische Bedeutung ist, dass der Algorithmus bei zwei Stellen mit gleicher mittlerer Vorhersage stets der Stelle mit größerer Unsicherheit einen höheren Akquisitionswert zuweisen wird, wodurch die *Exploration* formalisiert wird. Ein größeres Nichtwissen erweitert die Varianz der prädiktiven Gauß-Glocke und vergrößert damit den Wahrscheinlichkeitsbereich, der unterhalb der Schwelle $f_{\min}$ liegt.

An den bereits beobachteten Designpunkten des Datensatzes (unter der Annahme rauschfreier Beobachtungen) ist die Gewissheit des GP absolut, d.h. $\sigma(\bm{x}) = 0$. Bei der Auswertung dieses Grenzwerts verschwindet die erwartete Verbesserung ($E[I(\bm{x})] = 0$). Diese Eigenschaft ist entscheidend, da sie inhärent eine Stagnation des Algorithmus verhindert und sicherstellt, dass kein wertvolles Rechenbudget dafür verschwendet wird, genau denselben Punkt zweimal zu evaluieren.

![(a) Prädiktiver Mittelwert des GP, bedingt auf die verfügbaren Beobachtungen. (b) EI-Werte, die dem Modell aus Panel (a) zugeordnet sind. (c) Prädiktiver Mittelwert des GP nach Einbeziehung einer neuen Beobachtung mittels *Infill*. (d) Aktualisierte EI-Werte für die nächste Iteration.](/assets/articles/surrogate_models/app_forrester_ei_decision_synthesis_step5.png)

Insgesamt machen diese Eigenschaften EI als Auswahlkriterium innerhalb des *Infill*-Prozesses geeignet: Es ermöglicht, neue Auswertungen vorzuschlagen, indem die Suche in vielversprechenden Regionen mit der Exploration von Zonen kombiniert wird, in denen das Modell noch Unsicherheit aufweist. Dieser sequenzielle Mechanismus wird im experimentellen Teil der Arbeit verwendet und ist schematisch in Abbildung 2.2 zusammengefasst.

Sobald das maximale Auswertungsbudget (B) erschöpft ist oder ein vordefiniertes Konvergenzkriterium erreicht wurde, gilt der Anreicherungsprozess als abgeschlossen.

### Bewertungsmetriken

Um die Leistung der Surrogatmodelle zu validieren, reicht es nicht aus, die Genauigkeit ihrer Punktvorhersagen zu bewerten. Bei probabilistischen Modellen wie Gaußprozessen ist es grundlegend, die Qualität ihrer prädiktiven Unsicherheit zu quantifizieren. Daher werden die Bewertungsmetriken in zwei Kategorien unterteilt: klassische Vorhersagemetriken und probabilistische Metriken.

#### Vorhersagemetriken (MAE, RMSE, R$^2$)

Diese Metriken bewerten die Abweichung zwischen der tatsächlichen Beobachtung $y_i$ und dem prädiktiven Mittelwert des Modells $\mu_i$ (oder $\hat{y}_i$) über eine Menge der Größe $N$.

- **MAE (Mean Absolute Error):** Gegeben durch $\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \mu_i|$. Sie stellt die durchschnittliche Größe der Fehler in den Vorhersagen dar, ohne deren Richtung zu berücksichtigen. Sie wächst linear, was sie robuster gegenüber Ausreißern macht als RMSE.

- **RMSE (Root Mean Squared Error):** Definiert als $\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \mu_i)^2}$. Durch das Quadrieren der Differenzen werden große Fehler (Ausreißer oder katastrophale Vorhersagen) im Vergleich zu kleinen Fehlern stark bestraft.

- **Bestimmtheitsmaß ($R^2$):** Ausgedrückt als $R^2 = 1 - \frac{\sum_{i=1}^{N} (y_i - \mu_i)^2}{\sum_{i=1}^{N} (y_i - \bar{y})^2}$. Es misst den Anteil der Gesamtvarianz der abhängigen Variable, der durch das Modell erklärt wird.

#### Probabilistische Metriken (NLL, NLPD)

Während Metriken wie RMSE, MAE oder $R^2$ ausschließlich die Qualität der Punktvorhersage bewerten, berücksichtigen probabilistische Metriken die vollständige prädiktive Verteilung des Modells. Bei einem GP bedeutet dies, nicht nur zu bewerten, ob sich der prädiktive Mittelwert $\mu_i$ dem beobachteten Wert $y_i$ annähert, sondern auch, ob die vom Modell zugewiesene Unsicherheit mit den gemachten Fehlern kohärent ist.

Diese Unterscheidung ist bei probabilistischen Surrogatmodellen wichtig, da ein guter prädiktiver Mittelwert keine gute Quantifizierung der Unsicherheit garantiert. Ein Modell kann im Durchschnitt richtig liegen, aber zu selbstsicher sein; oder es kann fast alle tatsächlichen Punkte durch übermäßig breite und daher wenig informative Intervalle abdecken. Deshalb werden zusätzlich zu den klassischen Vorhersagemetriken Metriken berücksichtigt, die auf prädiktiver Dichte und Intervallabdeckung basieren (Rasmussen and Williams 2006; Gneiting and Raftery 2007; Kuleshov et al. 2018).

###### Negative Log-Likelihood und Negative Log Predictive Density.

Es lohnt sich, zwei verwandte, aber in unterschiedlichen Kontexten verwendete Größen zu unterscheiden. Während des Trainings des GP ist es üblich, das Negative der LML zu minimieren, auch *negative log marginal likelihood* genannt: $-\log p(\bm{y}\mid X,\bm{\theta}),$ was dem entgegengesetzten Vorzeichen des zuvor definierten LML-Ausdrucks entspricht. Diese Größe wird verwendet, um die Parameter $\bm{\theta}$ des Kernels und des Rauschens anzupassen.

Um dagegen die probabilistische Leistung auf einer Testmenge zu bewerten, ist es angemessener, die *Negative Log Predictive Density* (NLPD) zu verwenden. Diese Metrik bewertet Punkt für Punkt die Wahrscheinlichkeit, die das trainierte Modell dem tatsächlich beobachteten Wert zuweist. Wenn für jeden Testpunkt $\bm{x}_i$ die prädiktive Verteilung $p(y_i\mid \bm{x}_i,\mathcal{D}_n)=\mathcal{N}(\mu_i,s_i^2)$ ist, dann wird der mittlere NLPD über eine Testmenge von $N$ Stichproben definiert als

$$

\mathrm{NLPD}
=
\frac{1}{N}
\sum_{i=1}^{N}
\left[
\frac{1}{2}\log(2\pi s_i^2)
+
\frac{(y_i-\mu_i)^2}{2s_i^2}
\right].


$$

Hier stellt $s_i^2$ die Varianz der prädiktiven Verteilung dar, die zur Bewertung von $y_i$ verwendet wird. Werden verrauschte Beobachtungen evaluiert, muss diese Varianz sowohl die Unsicherheit über die latente Funktion als auch die Varianz des Beobachtungsrauschens einschließen. Wird hingegen direkt eine rauschfreie latente Funktion bewertet, würde nur die Posteriori-Varianz über $f(\bm{x}_i)$ verwendet. Niedrigere NLPD-Werte weisen auf eine bessere prädiktive Verteilung hin. Der obige Ausdruck bestraft merklich zwei unerwünschte Situationen:

1.  **Übermäßiges Vertrauen (*overconfidence*).** Weist das Modell eine sehr kleine Varianz $s_i^2$ zu, während sich der Mittelwert $\mu_i$ vom tatsächlichen Wert $y_i$ entfernt, wächst der quadratische Term $\frac{(y_i-\mu_i)^2}{2s_i^2}$ schnell an. Dies bestraft Modelle, die enge Intervalle liefern, aber mit großen Fehlern versagen.

2.  **Mangelndes Vertrauen (*underconfidence*).** Weist das Modell systematisch sehr große Varianzen zu, kann es zwar viele tatsächliche Werte abdecken, aber der Term $\frac{1}{2}\log(2\pi s_i^2)$ steigt an. Dies bestraft übermäßig breite und wenig informative prädiktive Verteilungen.

Daher misst der NLPD nicht nur den Fehler des Mittelwerts, sondern das Gleichgewicht zwischen Genauigkeit und Kalibrierung der Unsicherheit.

## Experimentierung und Ergebnisse

Dieses Kapitel überträgt den vorangegangenen theoretischen Rahmen auf das experimentelle Design der Arbeit. Ziel ist es, den Einsatz von GPs als probabilistische Surrogatmodelle zur Annäherung einer Beziehung zwischen Eingabekonfigurationen und einer interessierenden Antwort zu bewerten, insbesondere in Szenarien, in denen Auswertungen begrenzt oder kostspielig sind.

Die Experimentierung gliedert sich in zwei Szenarien. Zunächst werden synthetische Benchmark-Funktionen verwendet, bei denen die Zielfunktion bekannt ist und an neuen Punkten ausgewertet werden kann. Dieses Szenario ermöglicht es, den vollständigen Zyklus der surrogatbasierten Optimierung zu untersuchen: anfängliches Design, Anpassung des GP, sequenzielle Auswahl von Punkten mittels eines *Infill*-Kriteriums und Aktualisierung der beobachteten Menge. Zweitens wird ein realer Fall auf Basis experimenteller Daten zu Insektendiäten vorgestellt. In diesem Fall stammen die Beobachtungen aus bereits durchgeführten Versuchen und sind nach Diät gruppiert, sodass das Hauptziel nicht darin besteht, neue physische Auswertungen durchzuführen, sondern zu validieren, ob das Modell auf nicht gesehene Diäten generalisiert.

Beide Szenarien teilen somit denselben allgemeinen Rahmen der Surrogatmodellierung, jedoch nicht dasselbe experimentelle Protokoll: Die Benchmarks ermöglichen die Analyse des sequenziellen Suchprozesses in einer kontrollierten Umgebung, während der reale Fall die Durchführbarkeit des Ansatzes bewertet, bevor eine prospektive Phase der Empfehlung neuer Konfigurationen in Betracht gezogen wird.

### Gemeinsamer Rahmen der Surrogatmodellierung

In beiden Szenarien geht man von einer endlichen Menge von Beobachtungen aus

$$
\mathcal{D}_n = \{(\bm{x}_i, y_i)\}_{i=1}^{n},
$$

wobei $\bm{x}_i \in \mathcal{X}$ eine Menge von Eingabedaten darstellt und $y_i$ die beobachtete Antwort. Anhand dieser Daten wird ein Surrogatmodell trainiert, dessen Ziel es ist, die Beziehung zwischen Eingaben und Ausgaben anzunähern, Vorhersagen für nicht in der Beobachtungsmenge enthaltene Eingaben zu liefern und, im Fall von Gaußprozessen, eine Schätzung der Unsicherheit zu bieten.

#### Daten, Modelle und Vorverarbeitung

Die Eingaben werden in einer Designmatrix $X \in \mathbb{R}^{n \times d}$ organisiert, wobei $n$ die Anzahl der Beobachtungen und $d$ die Dimension des Eingaberaums ist. Die Antworten werden in einem Vektor $\bm{y} \in \mathbb{R}^{n}$ gesammelt. Bei den Benchmarks entspricht jede Beobachtung einer kontrollierten Auswertung einer synthetischen Funktion. Im realen Fall entspricht jede Beobachtung einer experimentellen Messung, die einer Diät zugeordnet ist, weshalb die Trainings- und Bewertungsaufteilungen diese Gruppierung berücksichtigen müssen, um Informationslecks zwischen Wiederholungen derselben Diät zu vermeiden.

Das experimentelle Design vergleicht eine konstante Baseline mit mehreren GP-Konfigurationen. Vor der Anpassung werden die Eingabevariablen mittels *Z-Score* standardisiert: Jede Spalte wird mit dem Mittelwert der Trainingsmenge zentriert und durch ihre Standardabweichung geteilt. Diese Transformation wird innerhalb jeder Partition oder experimentellen Iteration neu geschätzt und anschließend auf die entsprechende Bewertungsmenge angewendet, wodurch vermieden wird, während der Anpassung Testinformationen zu verwenden. Im realen Fall werden die zusätzlichen Transformationen, die zur Darstellung der Diäten erforderlich sind, in Abschnitt 3.3 beschrieben.

Das Design vergleicht eine konstante Baseline mit mehreren GP-Familien. Tabelle 3.1 fasst zusammen, welche Modelle in jedem Experiment verwendet werden und welche Rolle sie innerhalb der Analyse spielen. Die Baseline wird als Mindestreferenz interpretiert; die GPs bilden die untersuchten probabilistischen Surrogatmodelle. Diese Beschreibung definiert das experimentelle Protokoll; die konkreten Klassen und Pakete, die zu seiner Ausführung verwendet werden, werden separat beschrieben.

| **Familie** | **Verwendung** | **Rolle in der Analyse** |
|:---|:---|:---|
| Dummy | Benchmarks und realer Fall | Konstante Vorhersage, berechnet aus den Antworten der Trainingsmenge. Dient als Mindestreferenz gegenüber Modellen, die tatsächlich die Eingabevariablen nutzen. |
| Lineares GP | Benchmarks und realer Fall | GP mit linearem Kernel und Weißrausch-Term. Repräsentiert eine Hypothese einer annähernd linearen Beziehung zwischen Eingaben und Antwort. |
| RBF-GP | Benchmarks und realer Fall | GP mit RBF-Kernel und Weißrausch-Term. Führt eine glatte, nichtlineare Hypothese mit einer gemeinsamen Längenskala ein. |
| Matérn $3/2$-GP | Benchmarks und realer Fall | GP mit Matérn $3/2$-Kernel und Weißrausch-Term. Erlaubt weniger glatte Funktionen als RBF. |
| Matérn $5/2$-GP | Benchmarks und realer Fall | GP mit Matérn $5/2$-Kernel und Weißrausch-Term. Setzt eine glattere Antwort voraus als Matérn $3/2$, ist aber weniger restriktiv als RBF. |
| Zusammengesetztes GP | Realer Fall | GP mit additivem Kernel linear + Matérn $5/2$ + Weißrauschen. Wird verwendet, um einen globalen Trend mit einer glatten nichtlinearen Komponente im experimentellen Diätdatensatz zu kombinieren. |
| ARD-Variante | Je nach Protokoll | Variante mit einer Längenskala pro Variable. Bei den Benchmarks wird sie auf RBF und Matérn $5/2$ angewendet, sofern die Dimension dies zulässt; im realen Fall wird sie nur auf den vorherigen besten Kernel jeder Kombination aus Ziel und Eingabedarstellung getestet. |

*Betrachtete Modellfamilien pro Experiment.*

#### Probabilistische Vorhersage und Unsicherheit

Für eine neue Konfiguration $\bm{x}$ liefert das GP einen prädiktiven Mittelwert $\mu(\bm{x})$ und eine Standardabweichung $\sigma(\bm{x})$. Der Mittelwert wird als Punktvorhersage verwendet und die Standardabweichung als Maß der Unsicherheit. Bei den Benchmarks wird diese Unsicherheit aktiv innerhalb von EI genutzt, um neue Auswertungen auszuwählen. Im realen Fall dient sie als Indikator für die Zuverlässigkeit der Vorhersagen, ohne dass bislang eine physische *Infill*-Phase durchgeführt wird.

#### Implementierungsumgebung

Die Experimente wurden in Python implementiert, wobei `pandas` und `numpy` für die Datenmanipulation, `scikit-learn` für die Anpassung und Validierung der Modelle, `scipy` und `scikit-optimize` für numerische Hilfsfunktionen und Optimierung sowie `matplotlib`, `seaborn` und `plotly` für die Erstellung der Abbildungen verwendet wurden.

In der Implementierung wurden die Modelle in den Klassen `DummySurrogateRegressor` und `GPSurrogateRegressor` gekapselt, letztere aufgebaut auf einer `Pipeline` mit `StandardScaler` und `GaussianProcessRegressor`.

Die Ausführungen erfolgten auf einem Rechner mit Windows 10.0.26200 (64-Bit), Prozessor Intel(R) Core(TM) Ultra 7 155H, 22 logischen Threads und 32 GB RAM.

### Experiment 1: synthetische Benchmarks

Wie bereits angekündigt, verwendet das erste Experiment synthetische Benchmark-Funktionen, bei denen die Zielfunktion an neuen Punkten des Definitionsbereichs ausgewertet werden kann. Dies erlaubt es, den vollständigen Zyklus der Optimierung mit Surrogatmodellen zu untersuchen: initiales Design, Anpassung des GP, Auswahl neuer Auswertungen mittels EI und sequenzielle Aktualisierung der beobachteten Menge. Die Auswertung konzentriert sich auf die Entwicklung des besten gefundenen Werts, die Vorhersagefähigkeit und die Unsicherheit, während neue Beobachtungen hinzukommen.

#### Ausgewählte Benchmarks

Es werden vier Benchmarks aus der globalen Optimierung und dem Surrogatmodellieren betrachtet (Surjanovic and Bingham 2013; Forrester et al. 2008; Dixon and Szegö 1978; Morris et al. 1993), mit einer sinnvollen Schwierigkeitsprogression, von niedrigdimensionalen Problemen mit visueller Interpretierbarkeit bis hin zu höherdimensionalen Funktionen, die vor allem anhand von Metriken analysiert werden müssen. Die Suchbereiche werden in der Implementierung definiert und einheitlich für das initiale Sampling, die Testmenge und die Optimierung des Akquisitionskriteriums verwendet. All diese Informationen sind in Tabelle 3.2 zusammengefasst.

| **Benchmark** | **Dim.** | **Bereich** | **Rolle im Experiment** |
|:---|:--:|:---|:---|
| Forrester | 1 | $x \in [0,1]$ | Eindimensionaler Fall zur Visualisierung der GP-Anpassung, der Unsicherheit und des EI-Verhaltens. |
| Branin | 2 | $x_1 \in [-5,10]$, $x_2 \in [0,15]$ | Zweidimensionale multimodale Funktion, nützlich zur Analyse der Punktauswahl in einem darstellbaren Bereich. |
| Hartmann6 | 6 | $x_i \in [0,1]$, $i=1,\dots,6$ | Höherdimensionaler und multimodaler Benchmark, hauptsächlich anhand von Metriken ausgewertet. |
| Borehole | 8 | Heterogener Bereich mit acht Variablen | Ingenieurwissenschaftlich inspiriertes Problem zur Untersuchung des Methodenverhaltens in höherer Dimension. |

*Für das synthetische Experiment ausgewählte Benchmarks.*

#### Initiales Design, Budget und GP-Konfigurationen

Für jeden Benchmark wird ein initiales Design innerhalb des Suchbereichs erzeugt, das die verfügbare Information vor Beginn des sequenziellen *Infill*-Prozesses darstellt. Es werden zwei Strategien für das initiale Sampling verglichen, Sobol und uniformes Zufalls-Sampling, sowie drei Anfangsgrößen:

$$
n_{\text{train}} \in \{1,\; 1 \times d,\; 4 \times d\},
$$

wobei $d$ die Dimension des Benchmarks ist. Der Fall $n_{\text{train}}=1$ erlaubt es, das Systemverhalten ausgehend von einer extremen Situation mit nur einer einzigen Anfangsbeobachtung zu betrachten, während $1 \times d$ und $4 \times d$ mit der Problemdimension skalierte Designs darstellen.

Ausgehend vom initialen Design fügt der *Infill*-Prozess $5 \times d$ neue Punkte hinzu, bei einem Maximum von 50 Beobachtungen während der Trajektorie. Zudem werden drei Rauschbedingungen betrachtet: Auswertung ohne Rauschen sowie gaußsches Rauschen mit den Stufen $0{,}5$ und $1{,}0$. Die allgemeine Konfiguration ist in Tabelle 3.3 zusammengefasst, die die gemeinsamen Bedingungen festlegt, die anschließend zur Aggregation und zum Vergleich der Ergebnisse verwendet werden.

Der detaillierte Bereich von Borehole, der vollständige Ablauf des *Infill*-Prozesses und die spezifische Seed-Konfiguration sind als ergänzendes Material in Anhang 5 enthalten. Insbesondere ist der Bereich von Borehole in Tabelle 5.1 zusammengefasst, das Schema des sequenziellen Prozesses in Abbildung 5.1 und die verwendeten Seeds in Tabelle 5.2.

| **Element** | **Konfiguration** |
|:---|:---|
| Benchmarks | Forrester, Branin, Hartmann6 und Borehole. |
| Initiales Sampling | Sobol und uniformes Zufalls-Sampling. |
| Anfangsgröße | $n_{\text{train}} = 1$, $1 \times d$ und $4 \times d$. |
| *Infill*-Budget | $5 \times d$ zum initialen Design hinzugefügte Punkte, mit einem Maximum von 50 Beobachtungen während des Prozesses. |
| Rauschen | Ohne Rauschen und gaußsches Rauschen mit den Stufen $0{,}5$ und $1{,}0$. |
| GP-Modelle | Lineare Familie, RBF, Matérn $3/2$ und Matérn $5/2$, beschrieben in Tabelle 3.1; bei einer Dimension größer als eins werden zusätzlich ARD-Varianten für RBF und Matérn $5/2$ ausgewertet. |
| Testmenge | Unabhängige Menge, erzeugt innerhalb des Bereichs jedes Benchmarks. |
| Seeds | Es werden feste Seeds verwendet, um die Reproduzierbarkeit des Experiments zu gewährleisten. Die detaillierte Konfiguration findet sich in Anhang 5.3. |

*Allgemeine Konfiguration des Benchmark-Experiments.*

In jeder Iteration wird der GP mit der verfügbaren Beobachtungsmenge $\mathcal{D}_n$ angepasst. Anschließend wird EI über den Suchbereich berechnet und der nächste Punkt ausgewählt als

$$
\bm{x}_{\mathrm{next}}
    =
    \operatorname*{arg\,max}_{\bm{x}\in\mathcal{X}} (EI(\bm{x})).
$$

Obwohl die Zielfunktion als Minimierungsproblem formuliert ist, wird EI maximiert, da es die erwartete Verbesserung gegenüber dem bislang besten beobachteten Wert darstellt. In der Implementierung wird die Maximierung von EI mittels Differential Evolution approximiert; die Details des Optimierungsverfahrens der Akquisitionsfunktion sind in Anhang 5.2 zusammengefasst.

Die vollständige rechnerische Konfiguration dieses Experiments, einschließlich der Größe der Testmenge, des Explorationsparameters von EI, des Budgets des Optimierers und der verwendeten Seeds, ist in Anhang 8.1 dokumentiert.

#### Metriken des Benchmark-Experiments

Die prädiktiven und probabilistischen Metriken zur Bewertung des GP, wie MAE, RMSE, $R^2$, NLPD und Abdeckung des Vorhersageintervalls, folgen den in Abschnitt 2.5 eingeführten Definitionen. In diesem Experiment werden sie verwendet, um zu analysieren, ob der *Infill*-Prozess die Vorhersagefähigkeit des Modells verbessert, doch die zentrale Auswertung ist nicht nur prädiktiv: Es interessiert auch zu prüfen, ob EI es ermöglicht, kleinere Werte der Zielfunktion zu finden.

Aus Sicht der Optimierung ist die Hauptgröße die relative Verbesserung des während des *Infill*-Prozesses gefundenen Minimums. Der beste bis zu einer Iteration $t$ verfügbare Wert wird als *Incumbent* bezeichnet. Wenn das globale Optimum $f^\star$ bekannt ist, wird diese Verbesserung als relative Reduktion des *Gaps* zum Optimum ausgedrückt. Dazu wird $g_t = f_{\mathrm{best},t}^{\mathrm{clean}} - f^\star,$ definiert, wobei $f_{\mathrm{best},t}^{\mathrm{clean}}$ der beste saubere bis zur Iteration $t$ gefundene Wert ist. Aus diesem Abstand berechnet sich

$$
I_t = \frac{g_0 - g_t}{g_0},
$$

wobei $g_0$ dem initialen *Gap* des Ausgangsdesigns entspricht. Werte nahe null zeigen eine geringe Verbesserung gegenüber dem initialen Design an, während Werte nahe eins eine nahezu vollständige Reduktion des anfänglichen Abstands zum Optimum anzeigen.

Bei Borehole, wo kein in der Implementierung definiertes globales Optimum verfügbar ist, wird kein *Gap* zum Optimum berechnet. In diesem Fall wird dieselbe Auswertung durch die relative Reduktion des besten sauberen gefundenen Werts gegenüber dem initialen Design angenähert. Daher wird im Text "relative Verbesserung des gefundenen Minimums" als allgemeine Bezeichnung verwendet, und "Reduktion des *Gaps*" nur dann, wenn ein Referenzoptimum existiert.

Neben der finalen Verbesserung wird die kumulierte Verbesserung während der Trajektorie betrachtet. Diese Größe fasst die Entwicklung der relativen Verbesserung über den Verlauf des *Infill*-Prozesses zusammen und erlaubt es, zwischen Konfigurationen zu unterscheiden, die früh gute Lösungen finden, und solchen, die sich erst am Ende des verfügbaren Budgets verbessern. Somit misst die finale Verbesserung das am Ende des Prozesses erreichte Ergebnis, während die kumulierte Verbesserung eine zeitliche Auswertung der Suche liefert.

Die Vorhersagefähigkeit wird anhand des relativen MAE gegenüber dem Ausgangszustand sowie durch den Vergleich mit dem Dummy-Modell analysiert. Der relative MAE erlaubt es zu prüfen, ob die hinzugefügten Beobachtungen den Fehler des GP auf einer unabhängigen Testmenge reduzieren.

Die probabilistischen Metriken, insbesondere NLPD und die Abdeckung von 95 %, werden als ergänzende Diagnostik der Unsicherheit interpretiert. Niedrige NLPD-Werte und Abdeckungen nahe dem nominellen Niveau deuten auf kohärentere Vorhersageverteilungen hin, garantieren jedoch für sich genommen keine bessere Suche nach dem Optimum.

#### Ergebnisse des Benchmark-Experiments

Die Ergebnisse sind auf drei Ebenen gegliedert. Zunächst wird analysiert, ob der auf EI basierende *Infill*-Prozess den besten gefundenen Wert der Zielfunktion verbessert. Anschließend wird untersucht, ob die neuen Beobachtungen auch die Vorhersagefähigkeit des GP verbessern. Schließlich werden die experimentellen Faktoren zusammengefasst, wie Anfangsgröße, Rauschen, Dimensionalität, Kernel und Einsatz von ARD, die die beobachtete Leistung am stärksten bedingen.

##### Entwicklung des gefundenen Minimums während des Infill-Prozesses

Die aus Optimierungssicht zentrale Metrik ist die in Abschnitt 3.2.3 definierte relative Verbesserung des gefundenen Minimums. In dieser Analyse stellt *step*=0 den besten im initialen Design verfügbaren Punkt dar, bevor neue Auswertungen hinzugefügt werden. Von dort ausgehend integriert jede Iteration den von EI ausgewählten Punkt und aktualisiert den besten gefundenen Wert.

Tabelle 3.4 fasst das beste Modell jedes Benchmarks anhand der finalen Verbesserung des gefundenen Minimums zusammen. Zudem wird die kumulierte Verbesserung angegeben, die es erlaubt zu unterscheiden, ob die Verbesserung früh in der Trajektorie auftritt oder sich am Ende des *Infill*-Budgets konzentriert.

Die kumulierte Verbesserung wird anhand der Fläche unter der normalisierten Trajektorie der relativen Verbesserung berechnet. Sie misst also nicht nur das Endergebnis, sondern auch, wie schnell die Verbesserung eintritt. Ein Verhältnis nahe eins zeigt an, dass der Großteil der Verbesserung früh erzielt wird, während ein niedrigeres Verhältnis auf eine spätere Verbesserung hindeutet.

Die Ergebnisse zeigen, dass EI in allen vier Benchmarks positive Verbesserungen erzielt, wenn auch mit unterschiedlicher Intensität. Branin weist die klarste finale Verbesserung auf, mit einer durchschnittlichen Reduktion des *Gaps* nahe $0{,}90$. Auch Forrester zeigt eine deutliche Verbesserung, während bei Borehole diese Auswertung der relativen Reduktion des besten sauberen Werts gegenüber dem initialen Design entspricht, da kein Referenzoptimum verfügbar ist. Hartmann6 erweist sich als das anspruchsvollste Szenario: Die Verbesserung existiert, ist jedoch moderater und erfordert ein informativeres initiales Design, konsistent mit seiner höheren Dimensionalität.

| Benchmark | Bestes Modell | Finale Verbesserung | Kum. Verbesserung | Verhältnis | $n_{\text{train}}$ | Auswertung |
|:---|:---|:--:|:--:|:--:|:--:|:---|
| Borehole | $GP\_Linear$ | $0.720$ | $0.694$ | $0.963$ | $1$ | Frühe und starke Verbesserung. |
| Branin | $GP\_Matern52\_ARD$ | $0.898$ | $0.596$ | $0.663$ | $1$ | Sehr hohe finale Verbesserung. |
| Forrester | $GP\_Matern52$ | $0.737$ | $0.390$ | $0.530$ | $4$ | Starke, aber spätere Verbesserung. |
| Hartmann6 | $GP\_RBF$ | $0.434$ | $0.304$ | $0.701$ | $24$ | Moderate Verbesserung bei höherer Dimension. |

*Zusammenfassung des besten Optimierungsmodells je Benchmark. Die finale Verbesserung entspricht der relativen Verbesserung des gefundenen Minimums am Ende des Infill-Prozesses; bei Benchmarks mit bekanntem Optimum entspricht sie der relativen Reduktion des Gaps zum Optimum. Die kumulierte Verbesserung fasst die Entwicklung während der Trajektorie zusammen.*

![Entwicklung der relativen Verbesserung des gefundenen Minimums für das beste Modell jedes Benchmarks. Der Wert *step*=0 entspricht dem besten Punkt des initialen Designs, bevor Punkte mittels EI hinzugefügt werden. Die Kurven zeigen den Mittelwert und die schattierten Bänder die Standardabweichung.](/assets/articles/surrogate_models/evolution_incumbent_best_model_by_benchmark.png)

Bei Borehole liegen die finale und die kumulierte Verbesserung sehr nahe beieinander, was darauf hindeutet, dass der Prozess bereits in frühen Phasen günstige Regionen findet. Bei Branin ist die finale Verbesserung sehr hoch, wobei ein Teil des Fortschritts über den Verlauf der Trajektorie auftritt. Forrester weist eine starke, aber weniger unmittelbare finale Verbesserung auf. Bei Hartmann6 ist der Fortschritt gradueller, was bestätigt, dass das verfügbare Budget bei Problemen höherer Dimension restriktiver ist.

Abbildung 3.1 bestätigt diese zeitliche Auswertung. Bei Borehole und Branin tritt ein Großteil der Verbesserung während der ersten Iterationen auf, und danach tendieren die Kurven zur Stabilisierung. Bei Forrester hängt die Trajektorie stärker vom initialen Design ab, und die Verbesserung tritt weniger unmittelbar auf. Bei Hartmann6 ist der Fortschritt progressiver, was die Bedeutung eines informativeren initialen Designs bei Problemen höherer Dimension unterstreicht. Als ergänzendes Material zeigt Abbildung 5.2 die je Benchmark aggregierte finale relative Verbesserung des gefundenen Minimums.

Insgesamt bestätigt die Analyse der relativen Verbesserung des gefundenen Minimums, dass EI seine zentrale Funktion im Optimierungszyklus erfüllt: die Auswahl neuer Auswertungen, die den besten verfügbaren Wert gegenüber dem initialen Design verbessern.

##### Entwicklung der Vorhersagefähigkeit des GP

Über den Erfolg in der Optimierung hinaus wird analysiert, ob der *Infill*-Prozess die Vorhersagefähigkeit des GP verbessert. Dazu wird hauptsächlich der relative MAE gegenüber dem Ausgangszustand verwendet. Diese Auswertung muss von der relativen Verbesserung des gefundenen Minimums getrennt werden: Eine Konfiguration kann im Durchschnitt auf der Testmenge besser vorhersagen, ohne notwendigerweise am nützlichsten für das Finden von Minima zu sein.

Tabelle 3.5 fasst das beste prädiktive Modell jedes Benchmarks und dessen Vergleich mit Dummy zusammen. Dummy nimmt nicht am *Infill*-Prozess teil, dient aber als Referenz, um zu prüfen, ob der GP gegenüber einer trivialen Vorhersage einen Mehrwert bietet.

| Benchmark | Bestes MAE-Modell | MAE-Verbesserung | Gewinn vs. Dummy | \% besser als Dummy |
|:----------|:-----------------|:----------:|:-----------------:|:------------------:|
| Borehole  | $GP\_RBF\_ARD$   |  $0.574$   |      $0.847$      |     $100.0\%$      |
| Branin    | $GP\_RBF$        |  $0.175$   |      $0.280$      |      $72.2\%$      |
| Forrester | $GP\_RBF$        |  $0.281$   |      $0.291$      |      $66.7\%$      |
| Hartmann6 | $GP\_Linear$     |  $0.466$   |      $0.306$      |      $94.4\%$      |

*Zusammenfassung der prädiktiven Verbesserung des GP und Vergleich mit Dummy.*

Der GP bietet gegenüber Dummy in allen Benchmarks einen prädiktiven Mehrwert, wenn auch mit unterschiedlicher Intensität. Die Verbesserung ist besonders deutlich bei Borehole und Hartmann6, während sie bei Branin und Forrester moderater und unregelmäßiger ausfällt. Allerdings stimmen die besten prädiktiven Modelle nicht immer mit den besten Optimierungsmodellen überein. Bei Hartmann6 etwa ist das nach MAE beste Modell $GP\_Linear$, während das beste Modell zur Verbesserung des gefundenen Minimums $GP\_RBF$ ist. Ähnlich sticht $GP\_RBF$ beim MAE für Branin und Forrester hervor, während die besten Optimierungsmodelle $GP\_Matern52\_ARD$ bzw. $GP\_Matern52$ sind.

Las métricas probabilísticas se interpretan como diagnóstico complementario. En conjunto, el proceso de *infill* tiende a reducir simultáneamente el MAE y el NLPD respecto al estado inicial, aunque la calibración de la incertidumbre no es uniforme entre benchmarks. En Borehole y Branin se observa subcobertura en todos los modelos, lo que sugiere intervalos predictivos demasiado estrechos. En Forrester y Hartmann6 la cobertura se aproxima más al valor nominal en algunas configuraciones. Por tanto, la incertidumbre del GP resulta útil para guiar EI y diagnosticar la fiabilidad de las predicciones, pero no debe confundirse con una garantía directa de éxito en optimización.

Diese Ergebnisse bestätigen, dass die globale Genauigkeit, die probabilistische Kalibrierung und der Nutzen zur Steuerung der Suche nach dem Optimum verwandte, aber nicht äquivalente Aspekte sind. Die ergänzenden Abbildungen in Anhang 5 zeigen diese Trennung detaillierter: RBF und Matérn vereinen das stabilste Verhalten in mehreren Benchmarks, während der lineare Kernel weniger robust ist und nur in konkreten Fällen hervorsticht; siehe Abbildungen 5.3 und 5.4.

##### Einfluss der experimentellen Bedingungen

Abschließend wird der aggregierte Effekt der bewerteten experimentellen Bedingungen zusammengefasst. Angesichts der hohen Anzahl an Kombinationen besteht das Ziel nicht darin, jede Konfiguration isoliert zu beschreiben, sondern zu identifizieren, welche Faktoren die Leistung des GP + EI-Prozesses stärker zu bedingen scheinen.

| **Rang** | **Faktor** | **Beobachtetes Ergebnis** | **Schlussfolgerung** |
|:---|:---|:---|:---|
| 1 | Benchmark und Dimensionalität | Die beste finale Verbesserung des *Incumbent* variiert stark zwischen den Funktionen: Branin erreicht $0.898$, während Hartmann6 bei $0.434$ bleibt. | Die Schwierigkeit der Zielfunktionslandschaft dominiert das Ergebnis. Die Dimensionalität und die Struktur der Funktion bedingen mehr als jede isolierte Entscheidung der Pipeline. |
| 2 | $n_{\text{train}}$ initial | Der Effekt der Anfangsgröße ist innerhalb desselben Benchmarks sichtbar: Die Verbesserungsspanne erreicht $0.457$ bei Forrester und $0.365$ bei Branin. | Die Anfangsinformation bedingt die Trajektorie von EI. Mit wenigen Punkten besteht mehr Spielraum für relative Verbesserung, aber auch mehr Instabilität; deshalb muss dies zusammen mit dem erreichten Endwert gelesen werden. |
| 3 | Kernel | Die Unterschiede zwischen den Kerneln sind nicht immer eindeutig. Bei Borehole liegen mehrere Modelle praktisch gleichauf, während bei Branin, Forrester und Hartmann6 vor allem RBF- und Matérn-Varianten hervorstechen. | Der Kernel spielt eine Rolle, sollte aber nicht als Wettbewerb mit einem einzigen Gewinner gelesen werden. RBF und Matérn bieten das konsistenteste Verhalten, wobei die endgültige Wahl von der Geometrie des Benchmarks und dem analysierten Kriterium abhängt. |
| 4 | ARD | Bei der Verbesserung des *Incumbent* ergibt ARD gegenüber dem Basis-Kernel $\Delta=0.002$, mit $46\%$ günstigen Paaren ($n=108$). Beim MAE ist der Effekt größer: $\Delta=0.109$, $63\%$ günstig. | ARD verbessert die Optimierung nicht systematisch, kann aber der Vorhersagequalität helfen. Sein Nutzen hängt vom Benchmark ab und sollte nicht als allgemeiner Vorteil dargestellt werden. |
| 5 | Sampler | Sobol gegenüber Random ergibt $\Delta=0.035$ beim *Incumbent*, aber nur $38\%$ günstige Paare ($n=186$). Beim MAE ist das Delta negativ. | Der Sampler hat einen sekundären und unregelmäßigen Effekt. Er kann konkrete Trajektorien verändern, tritt aber nicht als dominanter Faktor gegenüber dem Benchmark, der Anfangsgröße oder dem Kernel auf. |
| 6 | Rauschen | Beim Vergleich ohne Rauschen gegenüber Gaußschem Rauschen verbessert sich der *Incumbent* im Mittel: $\Delta=0.110$ für $\sigma=0.5$ und $\Delta=0.125$ für $\sigma=1.0$. Die günstigen Prozentsätze sind jedoch moderat. | Rauschen verringert die Stabilität des Prozesses und kann die Suche beeinträchtigen, sein Effekt ist jedoch nicht einheitlich. Er hängt vom Benchmark ab und davon, ob Optimierung oder Vorhersagequalität analysiert wird. |

*Qualitative Lesart der Faktoren, die die Leistung bei Benchmarks bedingen.*

Tabelle 3.6 sollte nicht als strenges statistisches Ranking interpretiert werden, sondern als empirische Synthese der beobachteten Muster. Die Hauptmetrik zur Reihung der Faktoren ist die relative finale Verbesserung des beobachteten Minimums, das heißt, wie sehr sich der beste gefundene Wert nach dem *Infill*-Prozess verbessert. Wenn Deltas angegeben werden, stammen diese aus gepaarten Vergleichen: Zum Beispiel bedeutet $\Delta>0$ bei „Sobol -- Random", dass Sobol unter denselben übrigen Bedingungen eine höhere durchschnittliche Verbesserung erzielt als Random. Die Hauptschlussfolgerung ist, dass die Leistung des GP + EI-Zyklus vor allem durch den Benchmark selbst bedingt wird. In einer praktischen Lesart sollte der Fokus zunächst auf der geometrischen Schwierigkeit des Problems, der Größe des Anfangsdesigns und der Kernel-Familie liegen; ARD, der Sampler und das Rauschen werden als sekundäre Faktoren interpretiert, die konkrete Trajektorien verändern können, aber nicht allein ein schlecht konditioniertes Problem oder eine geringe Anfangsinformation ausgleichen können. Die Analyse der Effekte dieser Bedingungen ist als ergänzendes Material in Abbildung 5.5 enthalten.

### Experiment 2: Realfall mit Insektendiäten

Das zweite Experiment überträgt den Rahmen der Surrogatmodelle auf einen realen Fall, der auf experimentellen Daten der Insektenzucht basiert. In diesem Kontext liegt das Interesse auf der Larvenphase, da diese anschließend als Futterquelle genutzte Phase ist. Das experimentelle Problem besteht darin, zu untersuchen, wie verschiedene Diätformulierungen das Wachstum der Larven und ihre finale Zusammensetzung beeinflussen.

Das experimentelle Design geht von einer Kontrolldiät aus, die ausschließlich aus Weizenkleie besteht. Diese Diät stellt die übliche Zuchtreferenz dar und ermöglicht den Vergleich mit der Einführung unkonventioneller Diäten. Die unkonventionellen Diäten behalten die Weizenkleie als Basis bei, ersetzen jedoch einen Teil davon durch ein Nebenprodukt mit Nährwert. In der Hauptstudie werden drei Nebenprodukte berücksichtigt: Olivenblätter, Quinoaschalen und Olivenpresskuchen.

Aus modellierungstechnischer Sicht besteht das Ziel nicht nur darin, ein Modell an eine Datentabelle anzupassen, sondern zu bewerten, ob ein GP nützliche Beziehungen zwischen der Diätzusammensetzung und der beobachteten Reaktion der Larven erfassen kann. Diese Beziehungen sind relevant, weil die physischen Versuche begrenzt sind und jede neue Formulierung eine experimentelle Validierung erfordert. Daher wird das Surrogatmodell als vorbereitendes Werkzeug konzipiert, um Reaktionen zu schätzen und Unsicherheit zu quantifizieren, bevor möglicherweise eine prospektive Phase zur Auswahl neuer Diäten folgt.

Die interessierenden Reaktionen kombinieren produktive Parameter und Zusammensetzungsvariablen. Auf der produktiven Ebene ist `FCR` eine relevante Variable, die das Verhältnis zwischen der verzehrten Futtermenge und der erzielten Gewichtszunahme misst. Niedrigere `FCR`-Werte weisen auf eine effizientere Umwandlung des Futters in Biomasse hin. Auf der kompositionellen Ebene werden Variablen wie `PROTEINA (%)` und `QUITINA (%)` berücksichtigt. In den ersten Tests wurde auch `TPC` berücksichtigt, das die Konzentration der gesamten phenolischen Verbindungen in den Larven nach der Zucht erfasst und als zusätzlicher Indikator für den Diäteffekt genutzt wird, dessen Ergebnisse jedoch aufgrund der geringen Stichprobenzahl nicht analysiert werden.

Folglich verfolgt der Realfall zwei methodologische Ziele. Erstens soll geprüft werden, ob der GP in der Lage ist, anhand von Zusammensetzungs-, Inklusions- und Nebenprodukttyp-Variablen auf ungesehene Diäten zu generalisieren. Zweitens soll analysiert werden, ob die prädiktive Unsicherheit als Zuverlässigkeitssignal des Modells dienen kann. Im Unterschied zum Experiment mit synthetischen *Benchmarks* wird hier kein sequenzieller *Infill*-Zyklus ausgeführt, da die Beobachtungen aus bereits durchgeführten experimentellen Versuchen stammen.

#### Umfang des Realfalls und Auswahl von *Hermetia*

Die ursprüngliche experimentelle Datei enthält Informationen zu zwei Arten: *Hermetia illucens* und *Tenebrio molitor*. Zudem umfasst sie mehrere Informationsblöcke: Nährwertzusammensetzung und TPC der Diäten, produktive Zuchtparameter sowie Nährwertzusammensetzung und TPC der erhaltenen Larven. In diesem Experiment wird ausschließlich mit dem experimentellen Block von *Hermetia illucens* gearbeitet.

Die Daten zu *Tenebrio molitor* wurden hingegen nicht verwendet, da sie mehrere experimentelle Blöcke vermischen, zusätzliche Diättypen einbeziehen und Behandlungsunterschiede aufweisen, wie etwa die Wasserversorgung bei einigen Kontrollen. Um einen geschlossenen und vergleichbaren Fall beizubehalten, beschränkt sich die Analyse auf *Hermetia illucens*: 33 Beobachtungen, organisiert in 11 Diäten mit 3 Replikaten pro Diät. In diesem Block sind die Eingangsvariablen vollständig, und die Zielgrößen weisen ausreichende Verfügbarkeit auf: FCR verfügt über 31 gültige Werte, während Chitin und Protein über 33 verfügen.

#### Experimentelles Design und berücksichtigte Diäten

Der analysierte Datensatz enthält eine Kontrolldiät auf Basis von Weizenkleie sowie mehrere unkonventionelle Diäten, bei denen ein Teil dieser Kleie durch Olivenblätter, Olivenpresskuchen oder Quinoaschalen ersetzt wird. Zur Kennzeichnung der Formulierungen werden kurze Bezeichnungen verwendet, die das Nebenprodukt und den Inklusionsprozentsatz kombinieren: Zum Beispiel bezeichnet `Hoja15` eine Diät mit 15 % Olivenblatt. Dieselbe Regel gilt für die Diäten mit Presskuchen und Quinoa.

Jede Formulierung wurde in drei unabhängigen Larvenchargen getestet. Daher stellt jede Beobachtung ein Replikat einer Diät dar, keinen Durchschnitt pro Diät. Bei der Validierung bleiben die drei Replikate einer selben Formulierung zusammen: Wird eine Diät ausgeschlossen, nimmt keines ihrer Replikate am Training teil. Tabelle 3.7 fasst den Umfang des Realfalls zusammen.

| **Element** | **Beschreibung** |
|:---|:---|
| Analysierter Block | Versuche mit *Hermetia illucens* |
| Beobachtungen | 33 experimentelle Replikate |
| Bewertete Diäten | 11 Formulierungen |
| Replikate pro Diät | 3 unabhängige Chargen |
| Referenzdiät | Weizenkleie ohne Nebenprodukteinschluss |
| Hauptnebenprodukte | Olivenblatt, Olivenpresskuchen und Quinoaschale |
| Gruppierung für Validierung | Replikate derselben Diät |
| Modellierte Zielgrößen | FCR, Larvenchitin und -protein |
| Arbeitsebene | Experimentelles Replikat |

*Zusammenfassung der im Realfall verwendeten Daten.*

Die Eingangsrepräsentation beschreibt jede Diät anhand von Formulierungs- und Zusammensetzungsvariablen. Es werden zwei Sätze verglichen: ein reduzierter, mit Inklusionsprozentsatz, Protein, Faser, Fett und durchschnittlichem TPC der Diät; und ein vollständiger, der Asche, Kohlenhydrate und abgeleitete Nährwertverhältnisse hinzufügt. In beiden Fällen wird der Nebenprodukttyp einbezogen, sodass das Modell zwischen Kontrolle, Olivenblatt, Olivenpresskuchen und Quinoa unterscheidet, ohne sich nur auf den Diätnamen zu verlassen.

Auf diese Weise verwendet das Modell nicht nur den Diätnamen als Bezeichner, sondern eine strukturierte Beschreibung jeder Formulierung. Diese Repräsentation ermöglicht es zu untersuchen, ob die verfügbaren Variablen ausreichend Information enthalten, um die Reaktion vollständig ungesehener Diäten während des Trainings vorherzusagen.

#### Bereinigung und Vorverarbeitung

Der im Realfall verwendete Datensatz wird aus der ursprünglichen experimentellen Datei durch einen Prozess der Bereinigung, Kombination und Transformation von Tabellen gewonnen. Das Ziel dieser Phase ist es, von mehreren Excel-Blättern mit nach Analysetyp verteilten Informationen zu einer einzigen Tabelle auf Ebene des experimentellen Replikats zu gelangen. In dieser Tabelle muss jede Zeile sowohl die beobachtete Reaktion als auch die notwendigen Variablen zur Beschreibung der bewerteten Diät enthalten.

Der befolgte Prozess lässt sich in folgenden Schritten zusammenfassen.

****Schritt 1. Bereinigung der ursprünglichen Blätter.****

Zunächst werden die Blätter zu Diätzusammensetzung, produktiven Parametern, Larvenzusammensetzung und TPC normalisiert. In dieser Phase werden doppelte Spaltennamen korrigiert, Behandlungen vereinheitlicht und die Spalten für Mittelwerte und Abweichungen strukturiert. Zudem werden die Namen der Variablen normalisiert, die später bei der Modellierung verwendet werden.

****Schritt 2. Nummerierung der Replikate.****

Um die verschiedenen Informationsquellen korrekt verbinden zu können, wird innerhalb jeder Diät ein Replikat-Bezeichner vergeben. Dieser Schritt ist notwendig, da die produktiven Parameter und die Larvenzusammensetzung auf Chargen- bzw. Zuchtreplikat-Ebene erfasst werden. Auf diese Weise kann jede finale Beobachtung einer konkreten Diät und einem konkreten experimentellen Replikat zugeordnet werden.

****Schritt 3. Erstellung des Produktivitätsdatensatzes.****

Nach der Bereinigung der ursprünglichen Tabellen wird der Produktivitätsdatensatz von *Hermetia* erstellt. Dazu werden die produktiven Parameter mit der Larvenzusammensetzung kombiniert, wobei Diät, Behandlung, Replikat und Art als Schlüssel dienen. Anschließend werden die Nährwertzusammensetzung der Diät und die TPC-Information hinzugefügt. Das Ergebnis ist eine Tabelle auf Replikatebene, die produktive Information, Larvenzusammensetzung und Diätbeschreibung integriert.

****Schritt 4. Extraktion von Diät-Metadaten.****

Ausgehend vom textuellen Namen jeder Diät werden strukturierte Variablen erzeugt, wie `diet_name`, `byproduct_type` und `inclusion_pct`. Diese Transformation ermöglicht es, dass das Modell nicht nur vom Diätnamen abhängt, sondern von interpretierbaren Variablen: Nebenprodukttyp, Inklusionsprozentsatz und Nährwertzusammensetzung. Zudem werden abgeleitete Nährwertverhältnisse berechnet, die im vollständigen Variablenmodus verwendet werden.

****Schritt 5. Auswahl der Zielgröße und Filterung fehlender Werte.****

Die Modellierung erfolgt unabhängig für jede Zielvariable. Für jede Zielgröße werden ausschließlich die Zeilen entfernt, deren Reaktionswert fehlt. Diese Entscheidung vermeidet, für andere Zielgrößen gültige Beobachtungen zu verwerfen. So wird `FCR` mit 31 gültigen Beobachtungen bewertet, während `QUITINA (%)` und `PROTEINA (%)` mit den 33 verfügbaren Beobachtungen bewertet werden.

****Schritt 6. Erstellung von $X$, $\bm{y}$ und Gruppen.****

Für jede Zielgröße werden die Eingangsmatrix $X$, der Antwortvektor $\bm{y}$ und der Gruppenvektor erstellt. Der Vektor $\bm{y}$ enthält die beobachteten Werte der ausgewählten Zielgröße. Der Gruppenvektor wird aus `diet_name` gebildet, sodass alle Replikate derselben Diät denselben Gruppenbezeichner teilen. Diese Gruppierung wird die Grundlage des in Abschnitt 3.3.4 beschriebenen LODO-Protokolls sein.

****Schritt 7. Kodierung kategorialer Variablen.****

Die Eingangsvariablen werden durch Kombination der ausgewählten numerischen Variablen mit einer binären Kodierung des Nebenprodukttyps gebildet. Konkret wird `byproduct_type` in Indikatorvariablen umgewandelt und mit den numerischen Variablen des reduzierten oder vollständigen Modus verkettet. Alle resultierenden Spalten werden vor der Modellanpassung in numerisches Format umgewandelt.

****Schritt 8. Skalierung innerhalb der `Pipeline`.****

El escalado de variables se realiza dentro del `Pipeline` de cada modelo. Por tanto, los parámetros de normalización se estiman únicamente con los datos de entrenamiento de cada partición y se aplican después sobre la dieta retenida en test.

Das Skalieren der Variablen erfolgt innerhalb der `Pipeline` jedes Modells. Die Normalisierungsparameter werden daher ausschließlich mit den Trainingsdaten jeder Partition geschätzt und anschließend auf die im Test zurückgehaltene Diät angewendet.

Im final verwendeten CSV weisen die ausgewählten Eingabevariablen keine fehlenden Werte auf. Insgesamt transformiert diese Vorverarbeitung die ursprünglichen experimentellen Daten in eine überwachte, nach Diät gruppierte Struktur. Diese Organisation ermöglicht es, das Modell unter einer anspruchsvolleren Bedingung als einer zufälligen zeilenweisen Partitionierung zu bewerten: die Vorhersage der Reaktion vollständiger, während des Trainings nicht beobachteter Diäten.

#### Leave-One-Diet-Out-Bewertungsprotokoll

Die Bewertung des Praxisfalls erfolgt mittels eines *Leave-One-Diet-Out*-Protokolls (LODO), motiviert durch die gruppierte Struktur des Datensatzes. Die Beobachtungen sind keine unabhängigen, unzusammenhängenden Stichproben, sondern experimentelle Replikate, die derselben Diät zugeordnet sind. Eine zufällige zeilenweise Partitionierung könnte daher Replikate derselben Diät gleichzeitig in Training und Test platzieren, was zu einer zu optimistischen Schätzung der Vorhersagefähigkeit des Modells führen würde.

Sei $\mathcal{G}=\{g_1,\dots,g_K\}$ die Menge der verfügbaren Diäten. Im äußeren Fold $k$ besteht die Testmenge aus allen Beobachtungen, die zur Diät $g_k$ gehören:

$$
\mathcal{D}_{\text{test}}^{(k)}
    =
    \{(\bm{x}_i,y_i): g_i = g_k\}.
$$

Die Trainingsmenge enthält die übrigen Diäten:

$$
\mathcal{D}_{\text{train}}^{(k)}
    =
    \{(\bm{x}_i,y_i): g_i \in \mathcal{G}\setminus\{g_k\}\}.
$$

Auf diese Weise werden alle Replikate der zurückgehaltenen Diät von der Anpassung ausgeschlossen. Im *Hermetia*-Datensatz, der über 11 Diäten verfügt, erzeugt das Protokoll 11 äußere Folds.

*Schema des LODO-Protokolls: In jedem äußeren Fold wird eine vollständige Diät zurückgehalten, um die endgültige Leistung zu messen; die Hyperparameter-Auswahl erfolgt mittels interner Validierung auf den verbleibenden Diäten.*

Die Auswahl der Hyperparameter erfolgt verschachtelt. Für jeden äußeren Fold wird die zurückgehaltene Diät ausschließlich für die abschließende Bewertung reserviert. Auf den verbleibenden Diäten wird eine ebenfalls gruppenbasierte interne Validierung durchgeführt, und es wird die Konfiguration mit dem niedrigsten MAE ausgewählt. Anschließend wird das Modell mit allen Beobachtungen des äußeren Trainings erneut trainiert und auf der zurückgehaltenen Diät bewertet.

Dieses Design bewertet eine anspruchsvollere Aufgabe als die Interpolation zwischen Replikaten bereits beobachteter Diäten. Das Modell muss die Reaktion einer vollständigen Diät vorhersagen, die weder an der endgültigen Anpassung noch an der Hyperparameter-Auswahl beteiligt war. Daher werden die mit LODO erzielten Ergebnisse als Schätzung der Generalisierungsfähigkeit auf nicht gesehene Diäten innerhalb des verfügbaren experimentellen Raums interpretiert.

#### Modelle, Anpassung und Metriken

In dieser Phase werden die Modellfamilien aus Tabelle 3.1 wiederverwendet, die dem Praxisfall entsprechen: Dummy, GP linear, GP RBF, GP Matérn $3/2$, GP Matérn $5/2$ und zusammengesetztes GP. Alle Modelle werden nach dem in Abschnitt 3.3.4 beschriebenen verschachtelten LODO-Protokoll bewertet, unabhängig für jedes Ziel und jede Eingaberepräsentation. Um eine übermäßige Erkundung von Varianten zu vermeiden, wird ARD nicht von Anfang an auf alle Familien angewendet, sondern nur auf den zuvor besten Kernel jeder Kombination aus Ziel und Eingaberepräsentation; im endgültigen Protokoll beschränkt sich diese Variante auf RBF oder Matérn, nicht auf den zusammengesetzten Kernel.

Die zur Reproduktion beider Experimente notwendige Konfiguration wird gemeinsam in Anhang 8 zusammengefasst.

Im Praxisfall wird das zusammengesetzte GP durch eine Summe von Kernels konkretisiert:

$$
k(\bm{x},\bm{x}')
=
C_1\,k_{\text{Lineal}}(\bm{x},\bm{x}')
+
C_2\,k_{\text{Matern }5/2}(\bm{x},\bm{x}')
+
k_{\text{White}}(\bm{x},\bm{x}'),
$$

wobei $C_1$ und $C_2$ positive Skalierungskonstanten sind. Die lineare Komponente erfasst einen globalen Trend der Reaktion in Bezug auf die Eingabevariablen; die Matérn-$5/2$-Komponente ermöglicht sanfte lokale Abweichungen; und der weiße Rauschterm repräsentiert experimentelle Variabilität, die durch die verfügbaren Kovariaten nicht erklärt wird.

Die Anpassung ist in drei Ebenen organisiert:

****Interne Auswahl.****

In jedem äußeren Fold werden die Parameter mittels eines internen LODO ausgewählt, das ausschließlich auf den Trainingsdiäten angewendet wird. Die Auswahlmetrik ist der MAE, weshalb die Konfiguration mit dem niedrigsten MAE bei der internen Validierung gewählt wird.

****Externe Bewertung.****

Nach der Auswahl der Parameter wird das Modell mit allen Diäten des äußeren Trainings erneut trainiert und auf der zurückgehaltenen Diät bewertet. Die endgültigen Metriken stammen aus dieser externen Bewertung, nicht aus dem Training. Bei den GP-Modellen werden sowohl der Vorhersagemittelwert als auch die Standardabweichung ermittelt. Diese zweite Größe ermöglicht die Berechnung von Metriken im Zusammenhang mit Unsicherheit.

Die Interpretation der Metriken stützt sich auf die in Abschnitt 2.5 eingeführten prädiktiven und probabilistischen Definitionen, doch in diesem Fall muss präzisiert werden, wie diese unter LODO-Validierung aggregiert werden. Der MAE wird als Hauptmetrik zur Auswahl der Hyperparameter und zum Vergleich der Modelle verwendet, da er die ursprünglichen Einheiten jedes Ziels beibehält und eine direkte Interpretation des mittleren Fehlers erlaubt.

Bei LODO entspricht jeder äußere Fold einer zurückgehaltenen Diät. Daher wird der Makro-MAE als mittlerer Fehler pro Diät interpretiert: Er wird berechnet, indem der MAE der äußeren Folds gemittelt wird, sodass jede Diät das gleiche Gewicht erhält. Dies ist die Hauptinterpretation im Praxisfall, da das Ziel darin besteht, die Generalisierung auf vollständige, nicht gesehene Diäten zu bewerten, und nicht darin, Diäten mit mehr gültigen Replikaten zu bevorzugen.

Um Modelle mit der Baseline zu vergleichen, wird der relative MAE gegenüber Dummy verwendet:

$$
\rho_{\mathrm{Dummy}} = \frac{\mathrm{MAE}_{\mathrm{modelo}}}{\mathrm{MAE}_{\mathrm{Dummy}}}.
$$

Werte von $\rho_{\mathrm{Dummy}}<1$ zeigen an, dass das Modell den Fehler gegenüber der trivialen Vorhersage reduziert. Wenn zudem die Schwierigkeit jeder zurückgehaltenen Diät analysiert wird, wird der Fehler durch den Bereich des entsprechenden Ziels normalisiert. Diese Normalisierung ermöglicht den visuellen Vergleich von Fehlern bei Variablen mit unterschiedlichen Skalen, wie FCR, Chitin und Protein.

Bei den GP-Modellen wird zudem die prädiktive Unsicherheit bewertet. Die Abdeckung des ungefähren 95%-Intervalls misst den Anteil der Beobachtungen, die innerhalb von $\mu \pm 1.96\sigma$ liegen. Zudem wird die prädiktive Standardabweichung $\sigma(\bm{x})$ mit dem beobachteten absoluten Fehler $|y-\mu(\bm{x})|$ verglichen, um zu überprüfen, ob das Modell den Fällen, in denen es tatsächlich größere Fehler macht, eine höhere Unsicherheit zuweist. Diese Unsicherheitsmetriken werden nicht zur Parameterauswahl verwendet, sondern dienen als Diagnose der Zuverlässigkeit des GP.

Schließlich wird im prospektiven Teil des Praxisfalls EI als Akquisitions-Score interpretiert und nicht als Validierungsmetrik. Der beobachtete *Incumbent* repräsentiert die bereits experimentell gemessene beste Diät für jedes Ziel, während der Kandidat mit dem höchsten EI eine nicht getestete Formulierung ist, die das Modell aufgrund seiner Kombination aus Vorhersagemittelwert und Unsicherheit als informativ einschätzt. Diese Unterscheidung verhindert, dass eine prospektive Empfehlung als bestätigtes experimentelles Ergebnis dargestellt wird.

#### Ergebnisse des Praxisfalls

Die Ergebnisse des Praxisfalls werden mit einer anderen Lesart als beim Benchmark-Experiment analysiert. In diesem Szenario steht nur eine begrenzte Anzahl bereits durchgeführter experimenteller Versuche zur Verfügung. Das Hauptinteresse liegt daher nicht allein darin, den mittleren Fehler des Modells zu messen, sondern zu überprüfen, ob das GP übertragbare Beziehungen für vollständige, während des Trainings nicht beobachtete Diäten lernt.

##### Strukturelle Generalisierung versus Memorierung

Die erste Frage ist, ob das Ersatzmodell ein allgemeines Signal aus den Formulierungs- und Zusammensetzungsvariablen extrahiert oder ob es nur Diäten erkennt, die den beobachteten ähneln. Hierfür wird das in Abschnitt 3.3.4 beschriebene LODO-Protokoll verwendet, das das Modell dazu zwingt, eine vollständig unbekannte Formulierung vorherzusagen.

![Relative Leistung der Modelle gegenüber der Dummy-Baseline in der LODO-Validierung. Werte unter $1$ zeigen einen niedrigeren MAE als Dummy an und damit eine bessere Vorhersagefähigkeit bei nicht gesehenen Diäten.](/assets/articles/surrogate_models/fig_01_lodo_generalization_vs_dummy.png)

![Generalisierungsschwierigkeit pro zurückgehaltener Diät im LODO-Protokoll. Die Farbe stellt den durch den Zielbereich normalisierten Fehler dar, und der Text zeigt den mittleren MAE in den ursprünglichen Einheiten jeder Variable.](/assets/articles/surrogate_models/fig_02_error_by_left_out_diet.png)

Abbildung 3.3 fasst die relative Leistung gegenüber dem Dummy-Modell zusammen. Die vertikale Achse stellt den auf Dummy normalisierten MAE dar, sodass Werte unter $1$ eine Verbesserung gegenüber der trivialen Vorhersage anzeigen. Unter dieser Lesart verbessert das zusammengesetzte GP das Dummy-Modell bei allen drei aktiven Zielen: Die relative Fehlerreduktion beträgt $15.8\%$ bei FCR, $15.4\%$ bei Chitin und $36.6\%$ bei Protein. Dieses Ergebnis zeigt, dass sich das Modell nicht darauf beschränkt, einen globalen Mittelwert zu reproduzieren, sondern in den Eingabevariablen enthaltene Informationen nutzt, um die Reaktion nicht gesehener Diäten zumindest teilweise vorherzusagen.

Diese aggregierte Verbesserung darf jedoch nicht als gleichmäßige Generalisierung interpretiert werden. Abbildung 3.4 gliedert den Fehler nach zurückgehaltener Diät auf und zeigt, dass die Schwierigkeit des Problems von der bewerteten Formulierung abhängt. Bei FCR sind die Fehler bei mehreren Diäten relativ gering, obwohl schwierigere Diäten auftreten, wie `Hoja50` und einige Formulierungen mit Trester oder Quinoa. Bei Chitin konzentriert sich die Schwierigkeit besonders auf Diäten wie `Orujo70`, `Orujo90` und `Control`, während bei Protein hohe Fehler bei Diäten wie `Orujo50` und einigen Blattformulierungen auffallen.

Die gemeinsame Betrachtung beider Abbildungen ist wichtig. Die Verbesserung gegenüber Dummy zeigt, dass ein lernbares Signal in den Ernährungs- und Formulierungsvariablen vorhanden ist, aber die Fehlerkarte macht deutlich, dass sich dieses Signal nicht gleichermaßen auf alle Diäten überträgt. Dies steht im Einklang mit der Natur des Praxisfalls: Die Reaktionen der Larven hängen nicht nur von einem globalen Trend ab, der mit dem Einschlussprozentsatz oder der durchschnittlichen Zusammensetzung verbunden ist, sondern auch von spezifischen Effekten des Nebenprodukts, experimenteller Variabilität und möglichen Wechselwirkungen, die in den verfügbaren Variablen nicht vollständig abgebildet sind.

Folglich erlaubt das Ergebnis nicht die Behauptung, dass das GP das Problem der Vorhersage neuer Diäten vollständig löst. Es erlaubt jedoch die Aussage, dass das zusammengesetzte Modell nützliche Regelmäßigkeiten erfasst, die über das bloße Memorieren von Replikaten hinausgehen. Die LODO-Validierung fungiert hier als Test der strukturellen Generalisierung: Wenn das Modell Dummy übertrifft, wenn eine vollständige Diät entfernt wird, dann stammt ein Teil der gelernten Information aus gemeinsamen Beziehungen zwischen den Diäten.

##### Geometrische Ausdrucksstärke des zusammengesetzten GP

Nachdem überprüft wurde, dass das GP im Vergleich zur Baseline eine Vorhersagefähigkeit bietet, stellt sich als nächste Frage, welche Art von Kovarianzstruktur für diesen Praxisfall am besten geeignet ist. Wie in Abschnitt 2.2.3 erläutert, ist der Kernel nicht nur eine technische Komponente des GP, sondern das Element, das die a-priori-Hypothesen über die Form der zu approximierenden Funktion festlegt. In diesem Fall kann die biologische Reaktion der Larven einen globalen Trend enthalten, der mit der Ernährungszusammensetzung der Diät verbunden ist, aber auch lokale Abweichungen aufgrund der Art des Nebenprodukts, des Einschlussprozentsatzes und der experimentellen Variabilität.

Abbildung 3.5 vergleicht verschiedene Kernel-Familien anhand des mittleren MAE pro Diät, also des Makro-MAE des LODO-Protokolls. Der Vergleich zeigt, dass der zusammengesetzte Kernel das wettbewerbsfähigste Modell bei allen drei betrachteten Zielen ist. Dieser Unterschied ist bei Protein besonders deutlich, wo das zusammengesetzte GP den Fehler gegenüber Dummy und gegenüber den einfachen Kernels merklich reduziert. Auch bei Chitin ist eine konsistente Verbesserung gegenüber den einzelnen Alternativen zu beobachten, während bei FCR die Unterschiede zwischen mehreren nichtlinearen Kernels geringer ausfallen, obwohl der zusammengesetzte Kernel unter den besten Optionen bleibt.

![Vergleich der Kernel-Familien im *Hermetia*-Praxisfall. Die vertikale Achse zeigt den mittleren MAE pro Diät, äquivalent zum durch LODO-Validierung ermittelten Makro-MAE; niedrigere Werte deuten auf eine bessere Generalisierung auf nicht gesehene Diäten hin.](/assets/articles/surrogate_models/fig_04_kernel_family_mae.png)

Diese Lesart steht im Einklang mit der zuvor gegebenen Definition des zusammengesetzten GP: einer Kovarianzstruktur, die globalen Trend, lokale Flexibilität und experimentelles Rauschen kombiniert. Der Vergleich in Abbildung 3.5 stellt daher nicht nur Modellnamen gegenüber, sondern unterschiedliche Hypothesen über die Form der Beziehung zwischen Diät und Reaktion.

Dieses Ergebnis muss jedoch mit Vorsicht interpretiert werden. Die Überlegenheit des zusammengesetzten GP in Abbildung 3.5 bedeutet nicht, dass das Modell die biologische Dynamik des Systems vollständig erfasst, noch dass es außerhalb des beobachteten experimentellen Bereichs sicher extrapolieren kann. Was gezeigt wird, ist, dass innerhalb des von den verfügbaren Diäten abgedeckten Raums und unter LODO-Validierung eine Kovarianzstruktur, die globalen Trend, lokale Variation und experimentelles Rauschen kombiniert, die Daten besser beschreibt als die einzeln betrachteten einfachen Familien.

Daher wird das zusammengesetzte GP als Hauptmodell übernommen, nicht weil es die Unsicherheit des Praxisfalls beseitigt, sondern weil es die beste beobachtete Kombination aus Flexibilität und Robustheit bietet.

##### Praktische Kalibrierung der Unsicherheit

Neben der Bewertung der Genauigkeit des Vorhersagemittelwerts ist in diesem Praxisfall von Interesse, zu überprüfen, ob die Unsicherheit des GP nützliche Informationen über die Zuverlässigkeit seiner Vorhersagen liefert. Diese Frage ist besonders relevant, da das Ziel nicht nur darin besteht, bereits getestete Diäten zu interpolieren, sondern zu beurteilen, ob das Modell Entscheidungen über nicht beobachtete Formulierungen leiten kann. In diesem Zusammenhang kann eine fehlerbehaftete Punktvorhersage weiterhin nützlich sein, wenn das Modell in den Regionen, in denen es wahrscheinlich versagt, eine hohe Unsicherheit ausdrückt. Umgekehrt wäre ein Modell, das große Fehler mit zu engen Intervallen macht, als prospektives Werkzeug wenig zuverlässig.

Die Abbildung 3.6 vergleicht für jedes Ziel die prädiktive Standardabweichung des GP mit dem in den durch LODO zurückgehaltenen Diäten begangenen MAE. Diese Darstellung ermöglicht eine praktische Analyse, ob die Bereiche, in denen das Modell eine höhere Unsicherheit angibt, mit jenen übereinstimmen, in denen der tatsächliche Fehler größer ist. Der Zusammenhang ist bei Chitin am deutlichsten, wo eine positive Korrelation zwischen Unsicherheit und Fehler festzustellen ist. In diesem Fall tendiert der GP dazu, Beobachtungen, die sich tatsächlich als schwieriger vorherzusagen erweisen, eine höhere Standardabweichung zuzuweisen, sodass die Unsicherheit als nützliches Vorsichtssignal fungiert.

Bei FCR und Protein ist die Kalibrierung jedoch schwächer. Bei FCR ist der Zusammenhang zwischen Standardabweichung und Fehler praktisch null, was darauf hindeutet, dass das Modell einfache und schwierige Fälle anhand ihrer Unsicherheit nicht gut ordnet. Bei Protein ist der Zusammenhang ebenfalls begrenzt, und es treten sogar relevante Fehler in Bereichen auf, in denen das Modell nicht immer eine proportional hohe Unsicherheit zuweist. Die Unsicherheit des GP darf daher nicht als ein in allen Zielen perfekt kalibriertes Maß interpretiert werden. Insgesamt zeigt Abbildung 3.6, dass die prädiktive Unsicherheit nützliche, aber nicht einheitliche Informationen über die Zuverlässigkeit des Modells liefert. Eine ergänzende Visualisierung der Vorhersageintervalle je Diät findet sich in Anhang 7.5.

![Beziehung zwischen der prädiktiven Standardabweichung des GP und dem in der LODO-Validierung beobachteten absoluten Fehler. Jeder Punkt entspricht einer in einer zurückgehaltenen Diät bewerteten Beobachtung; die Farbe zeigt die Art des Nebenprodukts an.](/assets/articles/surrogate_models/fig_05_uncertainty_vs_error.png)

Folglich wird die Unsicherheit des GP in dieser Arbeit als unterstützendes Signal für die Entscheidungsfindung interpretiert, nicht als absolute Zuverlässigkeitsgarantie. Ihre Rolle ist besonders nützlich, um zukünftige Versuche zu priorisieren und Regionen zu erkennen, in denen das Modell eine größere Unkenntnis anerkennt. Jeder aus dem Surrogat abgeleitete Kandidat muss jedoch als experimentelle Hypothese verstanden werden, die noch der Validierung bedarf, und nicht als abgeschlossene Schlussfolgerung über die tatsächliche Leistung einer neuen Diät.

##### Prospektives Screening mittels EI

Zum Abschluss des Realfalls wird untersucht, ob das trainierte GP als Screening-Werkzeug für eine mögliche zukünftige Experimentierphase eingesetzt werden kann. Im Gegensatz zum Benchmark-Experiment wird hier kein physischer *Infill*-Zyklus ausgeführt: Es werden keine neuen Diäten getestet noch der Datensatz aktualisiert. Das Ziel besteht lediglich darin zu prüfen, ob das Modell eine Priorisierung von Kandidatenformulierungen innerhalb eines machbaren Raums erlaubt.

Dazu wird EI verwendet, das in Abschnitt 2.4.3 eingeführt und zuvor formuliert wurde, wobei die Ausrichtung jedes Ziels angepasst wird: FCR wird minimiert, während Chitin und Protein maximiert werden. Der prospektive Raum wird aus den im Realfall betrachteten Nebenprodukten Blatt, Trester und Quinoa sowie aus Einschlussprozentsätzen innerhalb des beobachteten Bereichs konstruiert. Über diesem Bereich wird ein Raster mit Schritten von $1\%$ erzeugt, wobei bereits getestete Formulierungen ausgeschlossen werden.

Tabelle 3.8 vergleicht für jedes Ziel den experimentell besten beobachteten Punkt mit dem von EI priorisierten nicht getesteten Kandidaten.

| **Ziel** | **Beobachteter Incumbent** | **EI-Kandidat** | **Vorhersage des Kandidaten** | **EI** | **Lesart** |
|:---|:---|:---|:---|:---|:---|
| FCR | `Quinoa30`: $1.543$ | `Quinoa16` | $1.726 \pm 0.065$ | $2.7{\cdot}10^{-5}$ | Der vorhergesagte Mittelwert verbessert den Incumbent nicht, weshalb der Kandidat als explorativ zu interpretieren ist. |
| Chitin | `Orujo70`: $15.318\%$ | `Orujo89` | $12.658 \pm 1.061\%$ | $0.002$ | EI priorisiert eine Trester-Region mit hoher Inklusion, ohne jedoch im Mittel den besten beobachteten Wert zu übertreffen. |
| Protein | `Orujo50`: $32.147\%$ | `Orujo31` | $30.938 \pm 0.912\%$ | $0.038$ | Der Kandidat liegt in einer nicht getesteten Trester-Zone, aber der vorhergesagte Mittelwert bleibt weiterhin unter dem Incumbent. |

*Beobachtete Incumbents und von EI priorisierte prospektive Kandidaten.*

Die Ergebnisse zeigen, dass die EI-Kandidaten die besten beobachteten Werte nicht ersetzen: Für FCR bleibt der Incumbent `Quinoa30`, für Chitin `Orujo70` und für Protein `Orujo50`. Zudem übertreffen die vorhergesagten Mittelwerte der Kandidaten diese Incumbents nicht eindeutig. Der Nutzen von EI liegt in diesem Fall daher nicht darin, neue optimale Diäten nachzuweisen, sondern darin, die Mittelwertvorhersage und die Unsicherheit des GP in konkrete Vorschläge für eine nächste Versuchsiteration umzuwandeln.

Insgesamt ist diese Analyse als eine Pre-*Infill*-Phase zu interpretieren. Das Surrogatmodell reduziert den Suchraum und weist auf Formulierungen mit einem gewissen Informationswert hin, aber die endgültige Entscheidung würde eine experimentelle Validierung erfordern. Eine ergänzende Visualisierung der EI-Landschaft über das machbare Raster findet sich in Anhang 7.6.

#### Zusammenfassung der Ergebnisse

Insgesamt zeigen die Ergebnisse des Kapitels ein kohärentes Bild zwischen den beiden experimentellen Szenarien. Bei den synthetischen Benchmarks ermöglicht der GP + EI-Zyklus eine Verbesserung des besten gefundenen Werts gegenüber dem Ausgangsdesign, insbesondere wenn die Unsicherheit des Modells dabei hilft, neue Auswertungen zu lenken. Im Realfall zeigt die LODO-Validierung, dass das zusammengesetzte GP gegenüber dem Dummy-Modell bei mehreren Zielen eine prädiktive Fähigkeit bietet, obwohl seine Unsicherheit nicht in allen Fällen perfekt kalibriert erscheint. Das Surrogatmodell darf daher nicht als Werkzeug interpretiert werden, das die experimentelle Validierung ersetzen kann, sondern als Mechanismus, um den Suchraum zu reduzieren, die Unsicherheit teilweise zu quantifizieren und vernünftige Kandidaten für eine nächste Versuchsiteration vorzuschlagen.

## Schlussfolgerungen und zukünftige Arbeit

Das Ziel dieser Abschlussarbeit (TFG) bestand darin, ein Entscheidungsproblem anzugehen: zu bestimmen, welche neuen Konfigurationen es wert sind, in einem realen Experiment mit hohen Kosten evaluiert zu werden. Als Ausgangspunkt wurde eine umfassende Untersuchung des Problems und der geeignetsten Werkzeuge zu seiner Lösung durchgeführt. Das gesamte entwickelte Wissen wurde rigoros und fundiert dargestellt, ausgehend von den Grundlagen des maschinellen Lernens und mit Vertiefung in die theoretischen Grundlagen bayesianischer Prozesse, der bayesianischen Optimierung und der Surrogatmodelle. Der Zweck dieses Ansatzes besteht nicht darin, das reale Experiment zu ersetzen, sondern die aus bereits beobachteten Daten gewonnenen Informationen zu nutzen, um Entscheidungen über die nächsten zu bewertenden Konfigurationen effizienter zu lenken.

In einer Vorphase wurden synthetische Experimente durchgeführt, die es uns ermöglichten, in einer kontrollierten Umgebung zu forschen, in der die reale Funktion bekannt ist und geprüft werden kann, ob das Suchverfahren das Ausgangsdesign verbessert. In diesem Szenario hat der Zyklus aus Anpassung des Gaußprozesses, Auswahl neuer Punkte mittels *Expected Improvement* und Aktualisierung des beobachteten Datensatzes den besten gefundenen Wert in allen untersuchten Benchmarks verbessert. Dies bestätigt, dass das Modell, sofern es über ein ausreichendes Signal verfügt, die Suche in vielversprechendere Regionen lenken kann als die, die allein mit dem Ausgangsdesign erzielt werden.

Diese vorherigen Experimente zeigen auch eine wichtige Schlussfolgerung: Ein Modell, das im Durchschnitt gut vorhersagt, ist nicht immer dasjenige, das am besten bei der Optimierung hilft. Bei dieser Art von Problemen kommt es nicht nur darauf an, den mittleren Fehler zu reduzieren, sondern auch darauf, die Unsicherheit zu nutzen, um zu entscheiden, wo es sich lohnt zu explorieren. Deshalb hängt der Nutzen eines Gaußprozesses nicht nur von seiner prädiktiven Genauigkeit ab, sondern von seiner Fähigkeit, Vorhersage und Unsicherheit in einer Strategie zur Auswahl neuer Auswertungen zu kombinieren.

Im Realfall der Diäten für *Hermetia illucens* wurde das Problem als eine Lernaufgabe auf Basis bereits verfügbarer experimenteller Daten angegangen. Um das Modell auf eine Weise zu bewerten und zu validieren, die dem praktischen Einsatz nahekommt, wurde ein Leave-One-Diet-Out-Protokoll verwendet, bei dem vollständige Diäten während des Trainings ausgeschlossen und geprüft wurde, ob das Modell in der Lage ist, deren Ergebnisse abzuschätzen. Dieser Ansatz ist anspruchsvoller als die Vorhersage isolierter Wiederholungen, da er dem realen Fall ähnelt, eine noch nicht getestete Diät bewerten zu wollen.

Unter dieser Bewertung verbessert der Gaußprozess das einfache Referenzmodell bei FCR, Chitin und Protein. Dies zeigt somit, dass die Formulierungs- und Zusammensetzungsvariablen der Diät nützliche Informationen enthalten und sich das Modell nicht darauf beschränkt, lediglich die verfügbaren Daten zu memorieren. Dennoch müssen die Ergebnisse mit Vorsicht interpretiert werden: Der Datensatz ist klein, die Generalisierung ist nicht für alle Diäten gleich gut, und die prädiktive Unsicherheit erscheint nur teilweise kalibriert. Die Vorhersagen des Modells sind daher als Entscheidungshilfe zu verstehen, nicht als endgültige biologische Schlussfolgerung.

Eine wichtige Erkenntnis des Realfalls ist, dass sich die Arbeit in einer ersten Phase des experimentellen Zyklus befindet. Mit den aktuellen Daten kann das Modell Kandidatendiäten vorschlagen und Regionen aufzeigen, die interessant erscheinen, kann aber noch nicht allein bestätigen, welche die beste Formulierung ist. Gerade deshalb ist der *Infill*-Prozess relevant: Durch die Auswahl neuer Diäten mittels eines Kriteriums wie Expected Improvement, deren experimentelle Bewertung und das erneute Training des Modells mit diesen neuen Daten könnte der Gaußprozess zunehmend robuster, nützlicher für die Priorisierung von Kandidaten und potenziell besser kalibriert werden.

Aus angewandter Sicht wäre die Empfehlung für ein wissenschaftliches Team, das mit Futtermitteln oder Diäten arbeitet, diese Methodik als Unterstützungswerkzeug zu nutzen, bevor die nächste Versuchsrunde geplant wird. Anstatt neue Formulierungen wenig gezielt zu testen, ermöglicht das Modell, die bereits durchgeführten Experimente zu nutzen, um die verfügbaren Optionen besser zu ordnen, vielversprechende Kandidaten zu identifizieren und zu entscheiden, wo es sich mehr lohnt, Zeit und Ressourcen zu investieren. Insgesamt zeigt die Arbeit, dass Surrogatmodelle die experimentelle Validierung nicht ersetzen, aber den Suchraum reduzieren, Unsicherheit quantifizieren und dabei helfen können, die nächsten zu testenden Diäten mit mehr Kriterium auszuwählen.

### Zukünftige Arbeit

Die erste Richtung zukünftiger Arbeit besteht darin, den experimentellen Zyklus des Realfalls abzuschließen. In dieser Abschlussarbeit reicht der Realfall bis zu einer Pre-*Infill*-Phase: Das Modell wird mit den verfügbaren Daten trainiert, mittels LODO validiert und zur Vorschlagsermittlung prospektiver Kandidaten genutzt. Der natürliche nächste Schritt wäre, eine oder mehrere Kandidatenformulierungen auszuwählen, sie experimentell zu bewerten und das GP mit den neuen Beobachtungen neu zu trainieren. Auf diese Weise würde der Realfall von einer retrospektiven Validierung zu einem echten sequenziellen Prozess der Optimierung mittels Surrogatmodellen übergehen.

Eine zweite Fortsetzungslinie wäre die Formalisierung einer multizielorientierten Entscheidungsfunktion. In der aktuellen Analyse werden FCR, Chitin und Protein getrennt untersucht, was es ermöglicht, jedes Ziel individuell zu interpretieren. Aus produktiver Sicht hängt die Wahl einer Diät jedoch normalerweise von mehreren gleichzeitigen Kriterien ab, wie Umwandlungseffizienz, Nährstoffzusammensetzung, Kosten, Verfügbarkeit des Nebenprodukts oder experimentelle Durchführbarkeit. Die Definition einer Entscheidungsregel zusammen mit Expertenkriterien würde es ermöglichen, Kompromisse zwischen Zielen und nicht nur individuelle Optima zu untersuchen.

Ebenso wäre es wichtig, den experimentellen Datensatz zu erweitern. Der verwendete Realfall enthält 11 Diäten mit jeweils 3 Wiederholungen, ausreichend für eine erste methodische Validierung, aber noch begrenzt, um starke biologische Schlussfolgerungen zu ziehen. Die Aufnahme neuer Diäten, weiterer Einschlussstufen und weiterer Wiederholungen würde die Stabilität des Modells verbessern, die Kalibrierung der Unsicherheit besser untersuchen und prüfen lassen, ob die gelernten Zusammenhänge auch außerhalb des derzeit beobachteten Bereichs bestehen bleiben.

Aus methodischer Sicht könnten zukünftige Erweiterungen Modelle untersuchen, die mehrere Ziele gemeinsam behandeln können, wie Multi-Output-Gaußprozesse oder hierarchische Modelle. Dies würde es ermöglichen, mögliche Zusammenhänge zwischen FCR, Protein und Chitin zu nutzen, insbesondere in Szenarien mit wenigen Daten. Ebenfalls interessant wäre die Untersuchung von Modellen mit heteroskedastischem Rauschen oder Techniken zur probabilistischen Rekalibrierung, da sich die Unsicherheit des GP bei den verschiedenen Zielen nicht einheitlich verhält.

Schließlich könnte der Akquisitionsprozess über EI hinaus erweitert werden. In zukünftigen Versionen könnten multizielorientierte Kriterien, Kriterien mit Restriktionen oder *Batch-Infill*-Strategien verwendet werden, die mehrere Diäten für den Test in derselben Versuchsrunde vorschlagen. Diese Erweiterungen würden es ermöglichen, den entwickelten Rahmen in ein vollständigeres Entscheidungsunterstützungswerkzeug umzuwandeln, das in der Lage ist, aufeinanderfolgende Versuchskampagnen mit einer effizienteren Nutzung des verfügbaren Budgets zu leiten.

## Zusätzliches Material zum Benchmark-Experiment

Dieser Anhang enthält das methodische und grafische Material des Benchmark-Experiments, das aus dem Haupttext entfernt wurde, um die Arbeit innerhalb der Umfangsgrenze zu halten. In Kapitel 3 bleiben die zentralen Elemente der Analyse erhalten: die Benchmark-Tabelle, die allgemeine Konfiguration des Experiments, die *Incumbent*-Zusammenfassung, die Entwicklung des *Incumbent*, die prädiktive Zusammenfassung und die Hierarchie der experimentellen Faktoren.

### Detaillierter Definitionsbereich des Borehole-Benchmarks

| **Variable** | **Beschreibung**                      | **Definitionsbereich**       |
|:-------------|:-------------------------------------|:------------------|
| $r_w$        | Radius des Bohrlochs                       | $[0.05, 0.15]$    |
| $r$          | Einflussradius                  | $[100, 50000]$    |
| $T_u$        | Transmissivität des oberen Grundwasserleiters | $[63070, 115600]$ |
| $H_u$        | Piezometrische Höhe (oben)          | $[990, 1110]$     |
| $T_l$        | Transmissivität des unteren Grundwasserleiters | $[63.1, 116]$     |
| $H_l$        | Piezometrische Höhe (unten)          | $[700, 820]$      |
| $L$          | Länge des Bohrlochs                    | $[1120, 1680]$    |
| $K_w$        | Hydraulische Leitfähigkeit des Bohrlochs    | $[9855, 12045]$   |

*Definitionsbereich des Borehole-Benchmarks.*

### Ablauf des *Infill*-Prozesses

*Ablauf des Infill-Prozesses: Anfangsdesign, Training des GP, Vorhersage von Mittelwert und Unsicherheit, Berechnung und Maximierung von EI, Auswertung des Benchmarks, Aktualisierung der Daten und Messung der Ergebnisse. Der Zyklus wiederholt sich, bis das Budget aufgebraucht ist.*

#### Detaillierte Beschreibung des *Infill*-Prozesses

Der *Infill*-Prozess bildet ein Optimierungsszenario mit kostspieligen Auswertungen nach, bei dem jeder neue Punkt fundiert ausgewählt werden muss. In diesem Experiment wird die Zielfunktion als Minimierungsproblem formuliert, weshalb die erwartete Verbesserung in Bezug auf den bisher besten beobachteten Wert berechnet wird.

In jeder Iteration wird der GP mit der verfügbaren Beobachtungsmenge $\mathcal{D}_n$ angepasst, und das EI-Kriterium wird über den Suchraum berechnet. Obwohl die Zielfunktion minimiert wird, wird EI maximiert, da der Punkt mit der größten erwarteten Verbesserung gesucht wird:

$$
\bm{x}_{\mathrm{next}}
=
\operatorname*{arg\,max}_{\bm{x}\in\mathcal{X}} (EI(\bm{x})).
$$

Die Maximierung von EI erfolgt mittels *Differential Evolution*, da es sich um eine robuste Strategie für potenziell multimodale Akquisitionsfunktionen handelt, die keine Gradienten benötigt. Zusätzlich wird `polish=True` verwendet, wodurch nach der globalen Suche eine lokale Verfeinerung mit L-BFGS-B angewendet wird. Da die Optimierer von SciPy als Minimierungsprobleme formuliert sind, wird in der Implementierung $-EI(\bm{x})$ optimiert, was der Maximierung von $EI(\bm{x})$ entspricht.

### Seed-Konfiguration und Reproduzierbarkeit

Dieser Abschnitt enthält ausschließlich die in den Benchmark-Trajektorien verwendete Seed-Zuweisung. Die vollständige Rechenkonfiguration und das Reproduktionsverfahren sind in Anhang 8 dokumentiert.

| **Element** | **Verwendeter Seed** |
|:---|:---|
| Anfangstraining | 42 |
| Testmenge | 1042 |
| Rauschen beim Training und Test | 42 |
| Rauschen während des *Infill* | 42 |
| EI, Differential Evolution und zufälliger Rückgriff | In jeder Iteration wird ein schrittabhängiger Seed verwendet, $S + \text{step} \times 1009$, wobei $S$ der Basis-Seed der Konfiguration ist. |
| Kreuzvalidierungs-Audit im aktiven Modus | Wird das Audit mittels KFold aktiviert, hängt der Seed vom Schritt ab, um die Reproduzierbarkeit jeder Iteration zu gewährleisten. |

*In dem Benchmark-Experiment verwendete Seed-Konfiguration.*

### Endgültige Verbesserung des gefundenen Minimums pro Benchmark

![Relative Endverbesserung des während des Infill-Prozesses gefundenen Minimums. Bei Forrester, Branin und Hartmann6 wird die relative Reduktion des Gap zum Optimum dargestellt. Bei Borehole, wo in der Implementierung kein definiertes Optimum verfügbar ist, wird die relative Reduktion des besten bereinigten Werts gegenüber dem Anfangsdesign gezeigt. Die Balken geben den Mittelwert und die vertikalen Linien die Standardabweichung über die aggregierten Trajektorien an.](/assets/articles/surrogate_models/evolution_incumbent_best_model_by_benchmark.png)

Abbildung 5.2 fasst den Optimierungserfolg des *Infill*-Prozesses zusammen, nicht die globale prädiktive Qualität des Ersatzmodells. Bei Branin wird die größte mittlere Reduktion des Gap zum Optimum erzielt, mit GP_Matern52_ARD, gefolgt von Forrester mit GP_Matern52. Borehole zeigt ebenfalls eine starke Verbesserung, wobei hier jedoch die Reduktion des besten bereinigten Werts gemessen wird, da in der Implementierung kein Referenzoptimum vorliegt. Hartmann6 weist eine moderatere Verbesserung mit größerer Variabilität auf.

Insgesamt bestätigt die Abbildung, dass EI es ermöglicht, in allen Benchmarks bessere Werte der Zielfunktion zu finden, zeigt aber auch, dass das Ausmaß der Verbesserung stark vom Problem abhängt. Daher sollte der *Incumbent* als zentrale Kennzahl des Sucherfolgs interpretiert werden, ergänzend zum MAE und den übrigen prädiktiven Metriken.

### Relative Entwicklung des MAE

![Relative Entwicklung des MAE für das beste prädiktive Modell jedes Benchmarks. Die gestrichelte Linie markiert den Anfangswert. Werte unter 1 zeigen eine Fehlerreduktion gegenüber dem nur mit dem Anfangsdesign trainierten Modell an.](/assets/articles/surrogate_models/evolution_relative_mae_best_model_by_benchmark.png)

Abbildung 5.3 zeigt die Entwicklung des relativen MAE während des *Infill*-Prozesses für das beste prädiktive Modell jedes Benchmarks. Die gestrichelte Linie bei $1$ stellt den Fehler des nur mit dem Anfangsdesign trainierten Modells dar; Werte unter $1$ zeigen daher eine Reduktion des MAE an, während Werte darüber eine vorübergehende Verschlechterung widerspiegeln.

Das klarste Muster zeigt sich bei Borehole, wo GP_RBF_ARD den Fehler durchgehend reduziert, besonders wenn das Anfangsdesign klein ist. Auch bei Hartmann6 zeigt sich eine rasche Verbesserung, insbesondere bei $n_{\mathrm{train}}=6$, wo GP_Linear eine deutliche Reduktion des MAE erreicht. Branin und Forrester dagegen weisen weniger monotone Trajektorien auf: Manche Konfigurationen verschlechtern sich zunächst, bevor sie sich stabilisieren oder verbessern. Dies deutet darauf hin, dass das Hinzufügen von Punkten mittels EI nicht immer eine sofortige Reduktion des Gesamtfehlers bewirkt.

Insgesamt zeigt die Abbildung, dass der *Infill* die prädiktive Qualität des Ersatzmodells verbessern kann, diese Verbesserung jedoch vom Benchmark, der Anfangsgröße und dem Modell abhängt. Zudem darf der relative MAE nicht mit dem Optimierungserfolg verwechselt werden: EI wählt Punkte aus, um die Suche nach dem Optimum zu verbessern, nicht notwendigerweise, um den prädiktiven Fehler im gesamten Bereich monoton zu minimieren.

### Prädiktive und probabilistische Diagnose

![Beziehung zwischen punktueller Genauigkeit und probabilistischer Diagnose während des Infill-Prozesses. Die horizontale Achse stellt den relativen MAE gegenüber dem Anfangswert dar, die vertikale Achse den relativen NLPD. Der untere linke Quadrant entspricht dem wünschenswerten Fall, mit gleichzeitiger Verbesserung von punktuellem Fehler und probabilistischer Diagnose. Die Farbe zeigt den Kalibrierungsfehler der 95%-Abdeckung an, definiert als $\left|\mathrm{coverage}_{95}-0.95\right|$.](/assets/articles/surrogate_models/mae_vs_probabilistic_diagnostics_by_step.png)

Abbildung 5.4 setzt die Verbesserung des punktuellen Fehlers in Beziehung zur Verbesserung der probabilistischen Diagnose während des *Infill*. Die horizontale Achse zeigt den relativen MAE, die vertikale Achse den relativen NLPD, beide gegenüber dem Anfangswert. Der untere linke Quadrant stellt somit den wünschenswerten Fall dar: geringerer mittlerer Fehler und bessere probabilistische Qualität als beim Anfangsdesign. Die Farbe fügt eine dritte Ablesung hinzu, den Kalibrierungsfehler der Abdeckung bei $95\,\%$, wobei dunklere Farbtöne besser kalibrierte Intervalle anzeigen.

Alle vier Benchmarks enden in der Zone gleichzeitiger Verbesserung, wenn auch mit Nuancen. Borehole zeigt die deutlichste Verbesserung bei MAE und NLPD, mit einem Endpunkt nahe $0.47$ beim relativen MAE und $0.05$ beim relativen NLPD; dennoch bleibt eine Unterdeckung bestehen, sodass die Unsicherheit nicht vollständig kalibriert ist. Forrester verbessert sich ebenfalls deutlich beim NLPD und moderat beim MAE. Branin zeigt eine sanftere Verbesserung und eine weniger direkte Trajektorie. Hartmann6 bietet das ausgewogenste Verhalten: Es reduziert den MAE und hält eine Kalibrierung nahe $95\,\%$ aufrecht.

Die Hauptschlussfolgerung ist, dass die prädiktive Verbesserung nicht allein anhand des MAE bewertet werden sollte. Ein Modell kann den mittleren Fehler reduzieren und dennoch schlecht kalibrierte Unsicherheitsintervalle liefern. Diese Abbildung ergänzt daher die MAE-Entwicklung und rechtfertigt eine gemeinsame Analyse von punktueller Genauigkeit, NLPD und prädiktiver Abdeckung.

### Effekt der experimentellen Bedingungen

![Gepaarte Effekte verschiedener experimenteller Bedingungen auf die Verbesserung des gefundenen Minimums und die relative Verbesserung des MAE. Ein positives Delta zeigt an, dass die erste Bedingung des Vergleichs eine größere mittlere Verbesserung erzielt. Die horizontalen Balken stellen die Standardabweichung der gepaarten Deltas dar.](/assets/articles/surrogate_models/factor_effects_paired_for_tfg.png)

Abbildung 5.5 vergleicht den mittleren Effekt verschiedener experimenteller Bedingungen, wobei die übrigen Konfigurationen jeweils paarweise konstant gehalten werden. Die Deltas sind im Allgemeinen klein im Verhältnis zu ihrer Variabilität, was darauf hindeutet, dass kein einzelner Faktor das Ergebnis stabil dominiert. Bei der Verbesserung des *Incumbent* zeigt das Fehlen von Rauschen einen moderat positiven Effekt gegenüber gaußschem Rauschen, während Sobol gegenüber Zufall und ARD gegenüber dem Basis-Kernel schwache Effekte aufweisen.

Beim MAE tendiert ARD dazu, das mittlere Ergebnis leicht zu verbessern, jedoch mit hoher Streuung. Das Sobol-Sampling zeigt keinen klaren Vorteil gegenüber dem zufälligen, und der Effekt des Rauschens auf den MAE ist gering. Die Abbildung legt daher nahe, dass das Endverhalten stärker vom Benchmark und dem Zusammenspiel der Bedingungen abhängt als von einer einzelnen isolierten experimentellen Entscheidung.

## Ergänzende Analyse des *Infill*-Prozesses

Dieser Anhang ergänzt das Benchmark-Material aus Anhang 5 um eine visuelle und statistische Betrachtung des *Infill*-Prozesses. Ziel ist nicht, neue Hauptergebnisse einzuführen, sondern detaillierter zu zeigen, wie das Surrogatmodell an der sequenziellen Auswahl von Punkten beteiligt ist und warum prädiktive Metriken und Optimierungsmetriken nicht immer dasselbe Modell auswählen.

### Zusammenhang zwischen prädiktiver Verbesserung und Verbesserung des gefundenen Minimums

Tabelle 6.1 vergleicht für jeden Benchmark das Modell mit der größten Verbesserung des gefundenen Minimums mit dem Modell mit der größten relativen Verbesserung des MAE. Dieser Vergleich ist relevant, weil das Ziel eines Optimierungszyklus per *Infill* nicht notwendigerweise darin besteht, den globalen mittleren Fehler des Surrogatmodells zu minimieren, sondern bessere Werte der Zielfunktion mit einem begrenzten Auswertungsbudget zu finden.

| **Benchmark** | **Bestes Modell nach gefundenem Minimum** | **Verbesserung *inc.*** | **Bestes Modell nach MAE** | **Verbesserung MAE** |
|:---|:---|:---|:---|:---|
| Borehole | GP_Linear | 0,720 | GP_RBF_ARD | 0,574 |
| Branin | GP_Matern52_ARD | 0,898 | GP_RBF | 0,175 |
| Forrester | GP_Matern52 | 0,737 | GP_RBF | 0,281 |
| Hartmann6 | GP_RBF | 0,434 | GP_Linear | 0,466 |

*Vergleich zwischen dem besten Modell nach Verbesserung des gefundenen Minimums und dem besten Modell nach relativer Verbesserung des MAE.*

Bei allen Benchmarks in Tabelle 6.1 stimmt das Modell, das den mittleren Fehler am stärksten reduziert, nicht zwangsläufig mit dem Modell überein, das die größte Verbesserung des gefundenen Minimums erzielt. Dieses Ergebnis rechtfertigt es, die prädiktive Qualität des Surrogatmodells und die Wirksamkeit des Suchprozesses getrennt zu betrachten.

### Forrester als visueller Fall des *Infill*-Mechanismus

Forrester wird als visueller Fall verwendet, weil es sich um einen eindimensionalen Benchmark handelt. Diese Eigenschaft erlaubt es, gleichzeitig die reale Funktion, den posterioren Mittelwert des GP, die prädiktive Unsicherheit und die beobachteten Punkte darzustellen. Er eignet sich daher gut, um die Rolle von EI bei der Auswahl neuer Auswertungen grafisch zu erklären.

![Entwicklung des Infill-Prozesses bei Forrester ohne Rauschen. Jedes Panel zeigt den Zustand des GP nach Einbeziehung einer wachsenden Anzahl von Punkten. Die blaue Linie stellt die reale Funktion dar, die orangefarbene Linie den prädiktiven Mittelwert des GP, die Bänder zeigen die Unsicherheit und die Punkte kennzeichnen die verfügbaren Beobachtungen.](/assets/articles/surrogate_models/app_forrester_gp_evolution_no_noise.png)

Abbildung 6.1 zeigt, wie sich das Surrogatmodell verändert, während neue Beobachtungen einbezogen werden. In den ersten Iterationen dominiert die Unsicherheit große Bereiche des Definitionsbereichs. Im Verlauf des Prozesses passt sich das Modell besser an die Region nahe dem Minimum an und reduziert die Unsicherheit rund um die ausgewerteten Punkte. Diese Visualisierung hilft zu verstehen, warum der Prozess nicht allein anhand des MAE bewertet werden sollte: Eine global unvollkommene Vorhersage kann dennoch nützlich sein, wenn sie die Suche in vielversprechende Regionen lenkt.

### Punktuelle Entscheidung mittels Expected Improvement

Abbildung 6.2 zeigt eine konkrete EI-Entscheidung bei Forrester. Der GP wird mit vier anfänglichen Sobol-Punkten und mehreren vorherigen *Infill*-Auswertungen trainiert. Im dargestellten Schritt schlägt EI den nächsten Punkt bei $x_{\mathrm{next}}=0.763$ vor, sehr nahe am bekannten Optimum des Benchmarks.

![Visuelle Synthese einer Expected-Improvement-Entscheidung bei Forrester. Das obere Panel zeigt den posterioren Mittelwert, die Unsicherheit, die vorherigen Beobachtungen, den *Incumbent* und den vorgeschlagenen Punkt. Das untere linke Panel stellt die EI-Akquisitionsfunktion dar, und das untere rechte Panel quantifiziert die Veränderung des Gaps zum Optimum nach Auswertung des neuen Punktes.](/assets/articles/surrogate_models/app_forrester_ei_decision_synthesis_step5.png)

| **Element** | **Wert** |
|:---|:---|
| Benchmark | Forrester |
| Modell | GP_Matern52 |
| Anfangsdesign | Sobol, $n_{\mathrm{train}}=4$, ohne Rauschen |
| *Infill*-Schritt | 5 |
| Vorgeschlagener Punkt | $x_{\mathrm{next}}=0.763$ |
| EI am vorgeschlagenen Punkt | 1,519 |
| Bester zuvor beobachteter Wert | -4,364 |
| Beobachteter Wert bei $x_{\mathrm{next}}$ | -6,004 |
| Bekanntes Optimum | $f(x^\star)=-6.021$ |
| Nach der Auswertung geschlossenes Gap | 99,0 % |

*Numerische Zusammenfassung der in Abbildung 6.2 dargestellten EI-Entscheidung.*

Dieses Beispiel zeigt die praktische Rolle der Unsicherheit in einem GP. EI wählt einen neuen Punkt nicht nur aufgrund des vorhergesagten Mittelwerts aus, sondern aufgrund der Kombination aus erwarteter Verbesserung und prädiktiver Streuung. In diesem Fall liegt der gewählte Punkt sehr nahe am tatsächlichen Optimum und reduziert das vor der Auswertung bestehende Gap fast vollständig.

### Ergänzende statistische Kontraste

Neben den beschreibenden Tabellen erlauben die durch das Experiment erzeugten Aufzeichnungen nichtparametrische Kontraste zu den Benchmark-Ergebnissen. Diese Kontraste sind nicht als formaler Nachweis universeller Überlegenheit gedacht, sondern als ergänzende Überprüfung zweier in der Hauptanalyse beobachteter Effekte: der Verbesserung des *Incumbent* während des *Infill*-Prozesses und der Existenz von Unterschieden zwischen GP-Kernel-Familien.

Um die Verbesserung des *Incumbent* zu bewerten, wurde der beste beobachtete Wert am Anfang und am Ende jeder aktiven Trajektorie verglichen. Jede Trajektorie ist definiert durch Benchmark, Sampler, Anfangsgröße, Rauschkonfiguration, Validierungsmodus und Modell. Da die Zielfunktion der Benchmarks als Minimierung formuliert ist, zeigt eine Verringerung des *Incumbent* eine Verbesserung an. Tabelle 6.3 fasst einen einseitigen Wilcoxon-Vorzeichen-Rang-Test mit der Alternativhypothese einer Verbesserung nach dem *Infill*-Prozess zusammen.

| **Modell** | **Trajektorien** | **Verbessern sich** | **Median der Verbesserung** | **$p$-Wert** |
|:---|---:|---:|---:|---:|
| Alle GP | 372 | 316 | 2,812 | $7,35\times10^{-54}$ |
| GP_Linear | 66 | 30 | 0,000 | $8,67\times10^{-7}$ |
| GP_Matern32 | 66 | 62 | 2,970 | $3,79\times10^{-12}$ |
| GP_Matern52 | 66 | 60 | 2,519 | $8,15\times10^{-12}$ |
| GP_Matern52_ARD | 54 | 54 | 8,105 | $8,13\times10^{-11}$ |
| GP_RBF | 66 | 59 | 2,574 | $1,20\times10^{-11}$ |
| GP_RBF_ARD | 54 | 51 | 7,602 | $2,57\times10^{-10}$ |

*Kontrast der Verbesserung des Incumbent zwischen Anfang und Ende der Infill-Trajektorien.*

Die Ergebnisse zeigen, dass der *Infill*-Prozess den *Incumbent* in den betrachteten Trajektorien systematisch reduziert. Der Effekt ist besonders deutlich bei den Kerneln mit ARD und bei den Matérn-/RBF-Kerneln, während das lineare Modell einen medianen Verbesserungswert von null aufweist, ohne sich jedoch in vielen Konfigurationen zu verschlechtern und mit punktuellen Verbesserungen, die ausreichen, damit der aggregierte Kontrast signifikant ist.

Um die GP-Kernel hinsichtlich der Vorhersagequalität zu vergleichen, wurde der MAE der vollständigen Blöcke verwendet, die von allen Kerneln geteilt werden. Jeder Block entspricht einer Kombination aus Benchmark, Sampler, Anfangsgröße, Rauschkonfiguration und Validierungsmodus. Bei den 54 vollständigen Blöcken erkennt der Friedman-Test Unterschiede zwischen den Modellen $(p=1,88\times10^{-6})$. Tabelle 6.4 zeigt die mittleren Rangwerte und die paarweisen Wilcoxon-Kontraste gegenüber dem Kernel mit dem besten mittleren Rang.

| **Modell** | **Mittlerer Rang** | **$p$ vs. GP_Matern52_ARD** | **Interpretation** |
|:---|---:|---:|:---|
| GP_Matern52_ARD | 2,741 | -- | Bester mittlerer Rang. |
| GP_RBF_ARD | 2,815 | 0,677 | Praktisch nicht vom Besten zu unterscheiden. |
| GP_Matern52 | 3,370 | $4,51\times10^{-4}$ | Schlechter als die ARD-Version im paarweisen Kontrast. |
| GP_RBF | 3,593 | 0,138 | Nicht eindeutiger Unterschied zum Besten. |
| GP_Matern32 | 4,056 | $2,04\times10^{-5}$ | Schlechter als der beste Kernel in den geteilten Blöcken. |
| GP_Linear | 4,426 | $5,28\times10^{-8}$ | Geringere aggregierte prädiktive Leistung. |

*Nichtparametrischer Vergleich zwischen GP-Kerneln anhand des MAE in vollständigen Blöcken. Niedrigere Rangwerte zeigen besseres Verhalten an.*

Diese Kontraste stützen zwei in der Arbeit verwendete Schlussfolgerungen: erstens, dass der *Infill*-Zyklus eine tatsächliche Verbesserung gegenüber dem besten beobachteten Wert bringt; zweitens, dass Kernel mit ARD tendenziell die besten Positionen beim MAE einnehmen. Es zeigt sich jedoch keine klare statistische Trennung zwischen GP_Matern52_ARD und GP_RBF_ARD, weshalb die Wahl eines konkreten Kernels zusammen mit der Stabilität, den Rechenkosten und dem Verhalten bei jedem einzelnen Problem interpretiert werden sollte.

## Ergänzendes Material zum Praxisfall

Dieser Anhang enthält Zusatzmaterial zum Praxisfall *Hermetia illucens*. Ziel ist es, die Rückverfolgbarkeit des modellierten Datensatzes zu dokumentieren und ergänzende Abbildungen bereitzustellen, die zur Interpretation der Ergebnisse aus Kapitel 3 beitragen, ohne die Hauptabbildungen zu LODO-Generalisierung, Kernel-Vergleich, Unsicherheit und Pre-*Infill* zu ersetzen.

### Inventar der erzeugten Datensätze

Der Aufbereitungsworkflow erzeugt mehrere Datensätze aus den ursprünglichen experimentellen Dateien. Die Modellierung des Praxisfalls beschränkt sich jedoch auf den Produktivitätsdatensatz von *Hermetia*, aus den in Kapitel 3 beschriebenen Gründen des Umfangs und der Homogenität. Tabelle 7.1 fasst die erzeugten Datensätze und ihre Rolle in der Arbeit zusammen.

| **Datensatz** | **Zeilen** | **Diäten** | **Art** | **Verwendung** |
|:---|:---|:---|:---|:---|
| `productivity_hermetia_lote.csv` | 33 | 11 | *Hermetia* | Im Aufbereitungsworkflow des Praxisfalls modellierter Datensatz. |
| `productivity_tenebrio_lote.csv` | 57 | 19 | *Tenebrio* | Erzeugter, verfügbarer Datensatz, in den aktuellen Ergebnissen nicht modelliert. |
| `productivity_all_lote.csv` | 90 | 19 | Beide | Kombinierter Datensatz, verfügbar für spätere Analysen. |
| `quality_hermetia_dieta.csv` | 11 | 11 | *Hermetia* | Als Unterstützung erzeugter Qualitätsdatensatz nach Diät. |
| `quality_tenebrio_dieta.csv` | 19 | 19 | *Tenebrio* | Als Unterstützung erzeugter Qualitätsdatensatz nach Diät. |
| `quality_all_dieta.csv` | 30 | 19 | Beide | Verfügbarer kombinierter Qualitätsdatensatz. |

*Inventar der im Fall Entomotive erzeugten Datensätze.*

Im modellierten Datensatz entsprechen die 33 Beobachtungen 11 Diäten mit drei Replikaten pro Diät. Die verwendeten Zielgrößen sind `FCR`, `QUITINA (%)` und `PROTEINA (%)`. Die Datenverfügbarkeit ist vollständig für Chitin und Protein, mit 33 gültigen Werten, und etwas geringer für FCR, mit 31 gültigen Werten.

### Beobachtete Werte nach Diät

Abbildung 7.1 fasst die beobachteten Mittelwerte nach Diät für die drei Zielgrößen des Praxisfalls zusammen. Diese Abbildung wird nicht als Hauptergebnis des Modells verwendet, sondern als deskriptive Vorabprüfung: Sie zeigt, dass die besten beobachteten Werte nicht immer zur selben Diät gehören und dass das Problem eine multiobjektive Natur hat.

![Beobachtete Mittelwerte nach Diät für FCR, Chitin und Protein im *Hermetia*-Datensatz. Bei FCR sind niedrigere Werte besser; bei Chitin und Protein sind höhere Werte besser.](/assets/articles/surrogate_models/app_real_dataset_targets_by_diet.png)

| **Zielgröße** | **Kriterium** | **Diät** | **Nebenprodukt** | **Beobachteter Mittelwert** |
|:---|:---|:---|:---|:---|
| FCR | Minimieren | Quinoa30 | Quinoa | 1,543 |
| Chitin | Maximieren | Orujo70 | Trester | 15,318 |
| Protein | Maximieren | Orujo50 | Trester | 32,147 |

*Beste beobachtete Diäten je Einzelzielgröße.*

### Paritätsdiagramm des besten GP je Zielgröße

Abbildung 7.2 ergänzt die aggregierten Metriken aus Kapitel 3. Statt den Fehler in einem einzigen Wert zusammenzufassen, stellt sie beobachtete gegenüber vorhergesagten Werten in der LODO-Validierung für das beste GP je Zielgröße dar. Die Diagonale kennzeichnet eine perfekte Vorhersage.

![LODO-Vorhersagen des besten GP je Zielgröße gegenüber den beobachteten Werten. Jeder Punkt entspricht einem experimentellen Replikat, die Farbe zeigt die Art des Nebenprodukts an.](/assets/articles/surrogate_models/app_real_best_gp_parity_by_target.png)

Diese Abbildung erlaubt es, Muster zu erkennen, die im makro-MAE verborgen bleiben. Bei FCR neigen die Vorhersagen dazu, sich in einem engen Intervall zu konzentrieren, was auf Schwierigkeiten bei der Extrapolation zwischen Diäten hinweist. Bei Chitin und Protein zeigt sich ein klareres Signal, wenngleich mit relevanten Fehlern bei einigen Nebenprodukten. Das Modell liefert also nützliche Informationen, sollte aber nicht als Ersatz für den experimentellen Test interpretiert werden.

### LODO-Kontraste gegenüber dem Dummy-Basismodell

Die LODO-Ergebnisse des Praxisfalls erlauben zudem eine ergänzende statistische Betrachtung, die jedoch aufgrund der Stichprobengröße notwendigerweise vorsichtig ausfallen muss. In dieser Analyse dient jede in der LODO-Validierung zurückgehaltene Diät als gepaarte Einheit, sodass die Modelle auf denselben Trainings- und Testpartitionen verglichen werden. Als Metrik wird der MAE je Diät verwendet, da dies die Hauptmetrik der Generalisierungsanalyse ist.

Zunächst wurde je Zielgröße ein Friedman-Test angewendet, um alle verfügbaren Modelle zu vergleichen. Anschließend wurde als gezielter Kontrast gegenüber dem Basismodell ein einseitiger Wilcoxon-Vorzeichenrangtest zwischen dem Dummy und dem GP mit dem besten mittleren Rang je Zielgröße durchgeführt. Die $p$-Werte sind als explorative Evidenz zu interpretieren, nicht als schlüssiger Beweis, da jede Zielgröße nur über 11 Diäten verfügt.

| **Zielgröße** | **Referenz-GP** | **Rang** | **$p_F$** | **Verbesserung** | **$\Delta$MAE Median** | **Lesart** |
|:---|:---|---:|---:|---:|---:|:---|
| FCR | GP_Matern32_NoARD | 3,455 | 0,521 | 8/11 | 0,017 | Tendenz zugunsten des GP, jedoch ohne schlüssige Evidenz gegenüber dem Dummy $(p=0,062)$. |
| Protein | GP_Matern32_ARD | 3,273 | 0,209 | 9/11 | 0,827 | Günstige Evidenz für den GP gegenüber dem Dummy $(p=0,034)$; auch andere GP-Kernel zeigen explorativ signifikante Verbesserungen. |
| Chitin | GP_Linear | 2,636 | 0,194 | 8/11 | 0,380 | Positive mediane Verbesserung, jedoch nicht schlüssiger Kontrast $(p=0,160)$. |

*Ergänzende Kontraste zum LODO-MAE je Diät gegenüber dem Dummy-Basismodell. \DeltaMAE steht für MAE\_{\mathrm{Dummy}}-MAE\_{\mathrm{GP}}, sodass positive Werte den GP begünstigen.*

Die Hauptlesart dieser Kontraste steht im Einklang mit den Paritätsdiagrammen und den aggregierten Metriken. Bei Protein liefert das GP-Modell eine konsistentere Verbesserung gegenüber dem Basismodell. Bei FCR zeigt sich ein schwaches Signal, was mit der Schwierigkeit der Extrapolation zwischen Diäten und der Konzentration der Vorhersagen in einem engen Bereich vereinbar ist. Bei Chitin begünstigen zwar einige Folds das GP, die Evidenz reicht jedoch nicht aus, um mit den verfügbaren Daten eine statistisch robuste Verbesserung zu behaupten.

### Vorhersageintervalle je Diät

Abbildung 7.3 zeigt für jede in der LODO-Validierung zurückgehaltene Diät den Vorhersagemittelwert des GP zusammen mit näherungsweisen $95\,\%$-Intervallen. Diese Visualisierung ergänzt die in Abschnitt 3.3.6.3 vorgestellte Kalibrierungsanalyse, da sie es erlaubt, disaggregiert zu betrachten, bei welchen Diäten die Intervalle die beobachteten Werte angemessen enthalten und bei welchen konkrete Abdeckungsfehler auftreten.

![Vorhersageintervalle je Diät unter LODO-Validierung. Die schwarzen Punkte stellen den Vorhersagemittelwert des GP dar, die Balken zeigen näherungsweise $95\,\%$-Intervalle, berechnet als $\mu \pm 1,96\sigma$. Die farbigen Punkte entsprechen den beobachteten Werten nach Art des Nebenprodukts.](/assets/articles/surrogate_models/fig_06_prediction_intervals_by_diet.png)

### Prospektive EI-Landschaft

Abbildung 7.4 zeigt die prospektive EI-Landschaft über dem im Praxisfall betrachteten machbaren Raster von Formulierungen. Diese Visualisierung ergänzt das in Abschnitt 3.3.6.4 vorgestellte Screening, bei dem die beobachteten Incumbents mit den durch EI priorisierten, nicht getesteten Kandidaten verglichen werden.

Ziel dieser Abbildung ist nicht zu zeigen, dass die vorgeschlagenen Kandidaten optimal sind, sondern zu veranschaulichen, wie das Akquisitionskriterium den prospektiven Wert über den Suchraum verteilt. Die Bereiche mit höherem EI entsprechen Kombinationen, bei denen das Modell eine mögliche Verbesserung gegenüber dem beobachteten Stand oder eine hinreichend relevante Unsicherheit identifiziert, die eine künftige Auswertung rechtfertigt.

![Prospektive EI-Landschaft über dem machbaren Raster von Formulierungen. Die EI-Maxima kennzeichnen Kandidatenpunkte für eine mögliche künftige Auswertung, indem sie die mittlere Vorhersage und die Unsicherheit des GP kombinieren.](/assets/articles/surrogate_models/fig_08_ei_candidate_landscape.png)

Diese Abbildung ist daher als Hilfsmittel für die nachfolgende Versuchsplanung zu verstehen. Im aktuellen Stand der Arbeit stellen die von EI hervorgehobenen Punkte Testhypothesen dar und keine validierten experimentellen Ergebnisse.

## Rechnerkonfiguration und Reproduzierbarkeit

Dieser Anhang dokumentiert die rechnerische Konfiguration, die zur Erzeugung der Ergebnisse dieser Arbeit verwendet wurde. Sein Ziel ist es nicht, neue Methodik einzuführen, sondern nachvollziehbar festzuhalten, wie das experimentelle Protokoll, der Code und die in der Arbeit verwendeten Ausgabedateien miteinander verknüpft sind. Im Unterschied zu den Anhängen A--C, die sich auf Ergebnisse und ergänzende Abbildungen konzentrieren, wird hier nur die zur Reproduktion notwendige Konfiguration erfasst.

### Konfiguration des Benchmark-Experiments

Das Benchmark-Experiment reproduziert ein Optimierungsszenario mit begrenzten Auswertungen. Für jede Funktion wird ein initiales Design erzeugt, ein GP angepasst und es werden neue Beobachtungen mittels EI hinzugefügt. Die relevante Konfiguration stammt aus `src/configs/benchmark_tuning_specs.py` sowie aus der in `src/analysis/active_learning.py` implementierten Logik des aktiven Lernens. Tabelle 8.1 fasst die verwendeten Werte zusammen.

| **Element** | **Konfiguration** |
|:---|:---|
| In der Arbeit beibehaltene Benchmarks | Forrester, Branin, Hartmann6 und Borehole. |
| Initiales Sampling | Sobol und uniformes Zufalls-Sampling. |
| Initiale Größen | $n_{\mathrm{train}}\in\{1,d,4d\}$, wobei $d$ die Dimension des Benchmarks ist. |
| Testmenge | 200 innerhalb der Domäne jedes Benchmarks generierte Punkte. |
| Synthetisches Rauschen | Ohne Rauschen, Gauß-Rauschen mit $\sigma=0.5$ und Gauß-Rauschen mit $\sigma=1.0$. |
| Modellfamilien | Dummy, lineares GP, RBF-GP, Matérn-$3/2$-GP, Matérn-$5/2$-GP; für Dimensionen größer als eins werden ARD-Varianten bei RBF und Matérn $5/2$ einbezogen. |
| *Infill*-Budget | $5d$ neue Auswertungen, mit einem Maximum von 50 Gesamtbeobachtungen während der Trajektorie. |
| Akquisitionsfunktion | Expected Improvement mit Explorationsparameter $\xi=0.01$. |
| Optimierung von EI | Differential Evolution über die kontinuierliche Domäne des Benchmarks, mit anschließender lokaler Verfeinerung. In der Implementierung wird $-EI(\bm{x})$ minimiert, was der Maximierung von $EI(\bm{x})$ entspricht. |
| Budget des EI-Optimierers | $\max(2000,500d)$ äquivalente Kandidatenauswertungen zur Dimensionierung der Akquisitionssuche. |
| Basis-Seed | 42\. Bei jedem *Infill*-Schritt verschiebt sich der EI-Seed als $42 + 1009\cdot \mathrm{step}$. |

*Rechnerische Konfiguration des Benchmark-Experiments.*

Die verwendeten Rauschpegel sind absolut und stellen daher nicht dieselbe relative Schwere in allen Benchmarks dar. Die Rauschanalyse ist daher als Robustheitstest der Pipeline unter festen Störungen zu verstehen, nicht als normalisierter Vergleich der Rauschwirkung zwischen den Funktionen.

Die internen Hyperparameter des GP, wie charakteristische Längen, Signalamplitude und Weißrauschpegel, werden während der Anpassung des `GaussianProcessRegressor` durch Maximierung der marginalen Log-Likelihood neu geschätzt. Daher sollte in den Benchmarks jede Kernel-Familie nicht als fester Satz numerischer Parameter interpretiert werden, sondern als strukturelle Kovarianz-Hypothese, die an die in jeder Iteration verfügbaren Daten angepasst wird.

Die Implementierung enthält zudem benchmarkspezifische Raster in `benchmark_grids.py`. Diese Raster dienen als Unterstützung für Audits oder Anpassungsmodi, wobei sich die Hauptlesart der Arbeit auf den aggregierten Vergleich der GP-Familien, den Effekt des initialen Designs, das Rauschen, die Sampling-Methode und die Entwicklung des *Incumbent* während des *Infill*-Prozesses konzentriert.

### Konfiguration des Realfalls

Der Realfall wird auf Grundlage des Datensatzes `productivity_hermetia_lote.csv` ausgeführt. Für jedes Ziel werden eine Eingabematrix, ein Antwortvektor und ein Gruppenvektor nach Diät erstellt. Die Validierungsgruppe ist nicht die Replik, sondern die vollständige Diät, sodass die drei Repliken einer Formulierung gemeinsam im Training oder im Test verbleiben.

Es werden drei aktive Ziele bewertet: FCR, Chitin und Protein. FCR wird als zu minimierende Variable interpretiert, während Chitin und Protein in der prospektiven Phase als zu maximierende Variablen interpretiert werden. Die endgültige Verfügbarkeit beträgt 31 Beobachtungen für FCR und 33 Beobachtungen für Chitin und Protein.

#### Eingaberepräsentationen

Es werden zwei Eingaberepräsentationen verglichen. Beide umfassen numerische Formulierungs- und Diätzusammensetzungsvariablen zusammen mit binären Indikatoren des Nebenprodukttyps. Der Diätname wird zur Definition der LODO-Gruppen verwendet, jedoch nicht als direkte Prädiktorvariable eingesetzt.

| **Modus** | **Dimension** | **Enthaltene Variablen** |
|:---|:---|:---|
| Reduziert | 9 Variablen | Fünf numerische Variablen: Einschlussprozentsatz, mittleres Protein der Diät, mittlere Faser, mittleres Fett und mittlerer TPC der Diät. Hinzu kommen vier binäre Nebenproduktindikatoren: Kontrolle, Blatt, Trester und Quinoa. |
| Vollständig | 14 Variablen | Die Variablen des reduzierten Modus, plus mittlere Asche, mittlere Kohlenhydrate und drei abgeleitete Verhältnisse: Protein/Kohlenhydrate, Protein/Faser und Faser/Fett. Enthält ebenfalls die vier Nebenproduktindikatoren. |

*Im Realfall verwendete Eingaberepräsentationen.*

Die ausgewählten Eingabevariablen weisen im verwendeten endgültigen Datensatz keine fehlenden Werte auf. Die Median-Imputation existiert im Aufbereitungsablauf als Sicherheitsmechanismus, verändert jedoch die Eingaberepräsentation in den Hauptergebnissen nicht. Die Skalierung der Variablen erfolgt innerhalb der `Pipeline` des GP-Modells, sodass die Standardisierungsparameter ausschließlich mit den Trainingsdaten jedes Folds angepasst werden.

#### Modelle und Raster des Realfalls

Die Anpassung des Realfalls verwendet verschachtelte LODO-Validierung. In jedem äußeren Fold wird eine vollständige Diät als Test zurückgehalten. Auf den verbleibenden Diäten wird eine interne LODO-Validierung durchgeführt, um Hyperparameter auszuwählen. Anschließend wird das Modell mit allen Diäten des äußeren Trainings neu trainiert und auf der zurückgehaltenen Diät evaluiert. Die interne Auswahlmetrik ist MAE.

| **Modell** | **Beschreibung** |
|:---|:---|
| Dummy | Konstante Basislinie. Das Raster erlaubt die Auswahl zwischen Mittelwert und Median der internen Trainingsmenge. |
| Lineares GP | Kernel `DotProduct` plus Weißrauschen. Repräsentiert einen globalen linearen Trend. |
| RBF-GP | RBF-Kernel plus Weißrauschen. Wird isotrop und, wo zutreffend, mit ARD ausgewertet. |
| Matérn-$3/2$-GP | Matérn-Kernel mit $\nu=3/2$ plus Weißrauschen. Wird isotrop und, wo zutreffend, mit ARD ausgewertet. |
| Matérn-$5/2$-GP | Matérn-Kernel mit $\nu=5/2$ plus Weißrauschen. Wird isotrop und, wo zutreffend, mit ARD ausgewertet. |
| Zusammengesetztes GP | Summe aus linearer Komponente, Matérn-$5/2$-Komponente und Weißrauschen. Im endgültigen Protokoll wird es ohne ARD ausgewertet, um eine übermäßige Erkundung von Varianten bei einem kleinen Datensatz zu vermeiden. |

*Im Realfall bewertete Modellfamilien.*

Das gemeinsame Raster der GPs ist in Tabelle 8.4 zusammengefasst. Jede Kernel-Familie wird mit der in Tabelle 8.3 angegebenen Struktur bewertet. Innerhalb jedes Folds optimiert `GaussianProcessRegressor` die kontinuierlichen Kernel-Parameter mittels der marginalen Log-Likelihood.

| **Parameter** | **Berücksichtigte Werte** |
|:---|:---|
| `alpha` | $\{10^{-10},10^{-5},10^{-2},1.0\}$. Dieser Term wird während der Anpassung zur Diagonalen der Kernel-Matrix hinzugefügt und hilft, numerische Stabilität und effektive Regularisierung zu kontrollieren. |
| `kernel` | Eine Kernel-Struktur pro Modellfamilie: linear, RBF, Matérn $3/2$, Matérn $5/2$ oder zusammengesetzt. |
| `normalize_y` | `True`. Die Zielvariable wird während der Anpassung des GP intern normalisiert. |
| `n_restarts_optimizer` | 15 Neustarts des Optimierers der marginalen Log-Likelihood. |
| Weißrauschen | `WhiteKernel` mit initialem Pegel $10^{-4}$ und Schranken $[10^{-7},0.8]$. |
| Charakteristische Längen | Bei isotropen Kernels wird eine gemeinsame initiale Skala verwendet. Bei ARD wird ein initialer Einsvektor verwendet, mit einer Skala pro Eingabevariable. Die Schranken sind $[10^{-2},10^{5}]$. |

*Im Realfall verwendetes Hyperparameter-Raster.*

Um die Anzahl der Modelle bei einem kleinen Datensatz zu begrenzen, wird ARD nicht wahllos auf alle Familien angewendet. Zunächst werden die Basisfamilien verglichen, danach wird eine ARD-Variante ausgewertet, die je nach Kombination aus Ziel und Eingaberepräsentation ausgewählt wird. Tabelle 8.5 fasst diese Varianten zusammen.

| **Repräsentation** | **Ziel** | **Bewertete ARD-Variante** |
|:-------------------|:-------------|:--------------------------|
| Reduziert           | FCR          | GP RBF ARD                |
| Reduziert           | Chitin      | GP RBF ARD                |
| Reduziert           | Protein     | GP Matérn $3/2$ ARD       |
| Vollständig           | FCR          | GP Matérn $5/2$ ARD       |
| Vollständig           | Chitin      | GP Matérn $5/2$ ARD       |
| Vollständig           | Protein     | GP Matérn $3/2$ ARD       |

*Je Kombination aus Ziel und Repräsentation bewertete ARD-Varianten.*

### Vermeidung von Informationsleckage

Die Vermeidung von Informationsleckage ist im Realfall besonders relevant, da die Beobachtungen Repliken von Diäten und keine unabhängigen Stichproben ohne Struktur sind. Das Protokoll vermeidet die Leckage durch vier Entscheidungen:

1.  **Gruppierung nach Diät.** Alle Repliken derselben Formulierung teilen dieselbe Gruppe. In jedem äußeren Fold wird eine vollständige Diät ausgelassen.

2.  **Verschachtelte Auswahl.** Die im äußeren Fold zurückgehaltene Diät nimmt nicht an der Auswahl der Hyperparameter teil. Die Auswahl erfolgt mit internem LODO über die Trainingsdiäten.

3.  **Skalierung innerhalb des Modells.** Die Standardisierung von $X$ wird innerhalb der `Pipeline` unter ausschließlicher Verwendung der Trainingsdaten des jeweiligen Folds angepasst.

4.  **Strukturelle Nutzung des Nebenprodukttyps.** Das Modell erhält Nebenproduktindikatoren und Zusammensetzungsvariablen, verwendet jedoch nicht den Diätnamen als direkten prädiktiven Identifikator. Der Diätname wird zur Konstruktion der LODO-Gruppen reserviert.

Die binäre Kodierung des Nebenprodukttyps wird anhand einer im Voraus bekannten experimentellen Variable erstellt. Diese Kodierung verwendet nicht die Antwort des zurückgehaltenen Folds. Folglich besteht das vermiedene Hauptleckagerisiko nicht darin, zu wissen, welche Nebenprodukte existieren, sondern darin, zuzulassen, dass Repliken einer Diät gleichzeitig im Training und im Test erscheinen.

### Reproduktion der Ergebnisse des Realfalls

Der Realfall kann mit dem Skript `reproduce_tfg_outputs.py` reproduziert werden. Standardmäßig verwendet dieses Skript die bereits berechneten Anpassungsprotokolle wieder, führt das Audit durch, regeneriert die endgültigen Abbildungen und kopiert die notwendigen grafischen Ressourcen in den Ressourcenordner der Arbeit. Falls die vollständige Neuberechnung der LODO-Anpassung gewünscht wird, kann es mit der Option `--rerun-tuning` ausgeführt werden.

    python reproduce_tfg_outputs.py
    python reproduce_tfg_outputs.py --rerun-tuning

Das Skript überprüft, ob die erwarteten Ergebnisse für die beiden Eingaberepräsentationen und die drei aktiven Ziele vorliegen. In der protokollierten Ausführung erfasst das Reproduzierbarkeits-Manifest 6 Anpassungszusammenfassungen und 42 Fold-Dateien, entsprechend den Kombinationen aus Repräsentation, Ziel und aktiven Modellen. Die Manifest-Datei wird gespeichert unter:

    outputs/reproducibility_real_case_manifest.json

Die in diesem Anhang dokumentierte Reproduzierbarkeit hat zwei Ebenen. Zum einen kann der Realfall aus den vorhandenen Anpassungsprotokollen regeneriert werden, was die Rekonstruktion der in der Arbeit verwendeten Tabellen, Audits, Abbildungen und grafischen Ressourcen ermöglicht. Zum anderen werden, falls die vollständige Anpassung ausgeführt wird, die verschachtelten LODO-Modelle ausgehend vom *Hermetia*-Datensatz neu berechnet. Diese zweite Option ist rechnerisch aufwendiger, da sie die interne Hyperparameterauswahl für jeden äußeren Fold, jedes Ziel, jede Repräsentation und jede Modellfamilie wiederholt.

Bei den Benchmarks hängen die Ergebnisse von den Seeds, der Sampling-Methode, der initialen Größe, dem Rauschen und dem *Infill*-Budget ab. Daher interpretiert die Arbeit einzelne Trajektorien nicht als endgültige Schlussfolgerungen, sondern als aggregierte Muster über die bewerteten Konfigurationen. Diese Unterscheidung ist wichtig: Das Ziel des Benchmark-Experiments besteht darin, das Verhalten des GP+EI-Zyklus unter kontrollierten Bedingungen zu untersuchen, während das Ziel des Realfalls darin besteht, zu prüfen, ob derselbe Rahmen ein nützliches Signal liefern kann, bevor eine neue experimentelle Kampagne geplant wird.

## Abkürzungen

|  |  |
|:---|:---|
| GP | Gaussian Process (Gaußprozess) |
| BO | Bayesian Optimization (Bayes'sche Optimierung) |
| SBO | Surrogate-Based Optimization (Surrogatmodell-basierte Optimierung) |
| DoE | Design of Experiments (Versuchsplanung) |
| BB | Black-Box (Blackbox) |
| RMSE | Root Mean Squared Error |
| MAE | Mean Absolute Error |
| NLL | Negative Log-Likelihood |
| LHS | Latin Hypercube Sampling (Lateinisches Hyperwürfel-Sampling) |
| CV | Cross-Validation (Kreuzvalidierung) |
| ARD | Automatic Relevance Determination (Automatische Relevanzbestimmung) |
| RBF | Radial Basis Function (Radiale Basisfunktion) |
| SE | Squared Exponential (Quadratisch-exponentiell; auch RBF) |
| LML | Log Marginal Likelihood (Marginale Log-Likelihood / Evidenz) |
| EI | Expected Improvement |
| DE | Differential Evolution |

## Referenzen

Bishop, Christopher M. 2006. *Pattern Recognition and Machine Learning*. Information Science and Statistics. Springer.

Dixon, L. C. W., and G. P. Szegö. 1978. "The Global Optimization Problem: An Introduction." In *Towards Global Optimisation 2*, edited by L. C. W. Dixon and G. P. Szegö. North-Holland.

Forrester, Alexander I. J., András Sóbester, and Andy J. Keane. 2008. *Engineering Design via Surrogate Modelling: A Practical Guide*. Wiley. <https://doi.org/10.1002/9780470770801>.

Gneiting, Tilmann, and Adrian E. Raftery. 2007. "Strictly Proper Scoring Rules, Prediction, and Estimation." *Journal of the American Statistical Association* 102 (477): 359--78. <https://doi.org/10.1198/016214506000001437>.

Hastie, Trevor, Robert Tibshirani, and Jerome Friedman. 2009. *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*. 2nd ed. Springer Series in Statistics. Springer. <https://doi.org/10.1007/978-0-387-84858-7>.

Jiang, Ping, Qi Zhou, and Xinyu Shao. 2020. *Surrogate Model-Based Engineering Design and Optimization*. Springer Tracts in Mechanical Engineering. Springer Singapore. <https://doi.org/10.1007/978-981-15-0731-1>.

Jones, Donald R., Matthias Schonlau, and William J. Welch. 1998. "Efficient Global Optimization of Expensive Black-Box Functions." *Journal of Global Optimization* 13 (4): 455--92. <https://doi.org/10.1023/A:1008306431147>.

Kuleshov, Volodymyr, Nathan Fenner, and Stefano Ermon. 2018. "Accurate Uncertainties for Deep Learning Using Calibrated Regression." *Proceedings of the 35th International Conference on Machine Learning*, Proceedings of machine learning research, vol. 80: 2796--804. <https://proceedings.mlr.press/v80/kuleshov18a.html>.

Morris, Max D., Toby J. Mitchell, and Donald Ylvisaker. 1993. "Bayesian Design and Analysis of Computer Experiments: Use of Derivatives in Surface Prediction." *Technometrics* 35 (3): 243--55. <https://doi.org/10.1080/00401706.1993.10485320>.

NIST Digital Library of Mathematical Functions. 2010. "Chapter 10: Bessel Functions." Edited by Frank W. J. Olver, Daniel W. Lozier, Ronald F. Boisvert, and Charles W. Clark. <https://dlmf.nist.gov/10>.

Rasmussen, Carl Edward, and Christopher K. I. Williams. 2006. *Gaussian Processes for Machine Learning*. Adaptive Computation and Machine Learning. MIT Press. <http://www.GaussianProcess.org/gpml>.

Schölkopf, Bernhard, Ralf Herbrich, and Alexander J. Smola. 2001. "A Generalized Representer Theorem." *Computational Learning Theory*, Lecture notes in computer science, vol. 2111: 416--26. <https://doi.org/10.1007/3-540-44581-1_27>.

Settles, Burr. 2009. *Active Learning Literature Survey*. Computer Sciences Technical Report No. 1648. University of Wisconsin--Madison.

Sobol', Ilya M. 1967. "On the Distribution of Points in a Cube and the Approximate Evaluation of Integrals." *USSR Computational Mathematics and Mathematical Physics* 7 (4): 86--112. <https://doi.org/10.1016/0041-5553(67)90144-9>.

Surjanovic, Sonja, and Derek Bingham. 2013. "Virtual Library of Simulation Experiments: Test Functions and Datasets." <https://www.sfu.ca/~ssurjano/optimization.html>.
