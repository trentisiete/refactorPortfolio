---
title: "Bilderzeugung mittels stochastischer Differentialgleichungen und Diffusionsprozessen"
date: 2025-05-27
excerpt: "Eine umfassende Analyse von Diffusionsmodellen und ihrer Anwendung bei der Bilderzeugung."
# tags: ["python", "sklearn", "mlstudio", "gemini"]
draft: false
kind: "article"
# readingTime: 7
lang: "de"
translated: true
sourceHash: "0003347d2502cc4d"
---

# Vom Rauschen zum Bild: eine mathematische Lektüre der Diffusionsmodelle

Diffusionsmodelle lassen sich anhand einer scheinbar einfachen Idee verstehen: Wenn wir wissen, wie man ein Bild Schritt für Schritt zerstört, bis es zu Rauschen wird, können wir vielleicht den umgekehrten Weg lernen. Der interessante Teil ist, dass dieser umgekehrte Weg weder darin besteht, Bilder auswendig zu lernen, noch einen klassischen Rauschfilter anzuwenden. Er besteht darin, die lokale Geometrie einer Wahrscheinlichkeitsverteilung zu lernen: wohin sich eine verrauschte Probe bewegen muss, um einem echten Bild immer ähnlicher zu werden.

In diesem Artikel stelle ich eine Implementierung von Diffusionsmodellen vor, die mittels stochastischer Differentialgleichungen (*Stochastic Differential Equations*, SDEs) formuliert sind, nach dem Rahmenwerk von Song et al. [^song2021scorebased_sde]. Die Idee ist nicht nur, generierte Bilder zu zeigen, sondern zu erklären, warum die Methode funktioniert, welche Rolle der *Score* $\nabla_x \log p_t(x)$ spielt und wie die inverse SDE, das Training durch *Denoising Score Matching*, die *Sampler* und die *Probability Flow ODE* auf natürliche Weise entstehen.

Der zugehörige Code ist auf GitHub verfügbar:

<https://github.com/trentisiete/image_generation_difussion_project>

---

## 1. Das Problem: eine Verteilung in einem riesigen Raum lernen

Ein Bild ist nicht nur eine Matrix aus Pixeln. Aus probabilistischer Sicht ist ein Bild ein Punkt in einem Raum sehr hoher Dimension. Ein RGB-Bild von $32 \times 32$, wie die von CIFAR-10, lebt in einem Raum der Dimension $32 \cdot 32 \cdot 3 = 3072$. Allerdings sind nicht alle Punkte dieses Raums natürliche Bilder. Die meisten sind Rauschen ohne Struktur. Echte Bilder besetzen einen sehr speziellen Bereich: Sie enthalten Kanten, Texturen, Symmetrien, Objekte, Hintergründe, kompatible Farben und Muster, die nicht zufällig entstehen.

Das Ziel eines generativen Modells ist es, die Verteilung der echten Daten anzunähern, die wir mit $p_{\text{data}}(x)$ bezeichnen, um neue Proben generieren zu können:

$$
x \sim p_{\text{model}}(x)
$$

wobei $p_{\text{model}}$ versucht, $p_{\text{data}}$ zu ähneln.

Die Herausforderung besteht darin, dass $p_{\text{data}}(x)$ nicht explizit bekannt ist. Wir haben Beispiele, aber keine geschlossene Formel der Verteilung. Diffusionsmodelle greifen dieses Problem auf elegante Weise an: Anstatt direkt zu versuchen zu lernen, wie man ein sauberes Bild von Grund auf erzeugt, konstruieren sie zunächst einen Prozess, der die Daten kontrolliert zerstört, und lernen anschließend, ihn umzukehren.

---

## 2. Die zentrale Intuition: Zerstören ist einfach, Rekonstruieren ist Lernen

Der Vorwärtsprozess beginnt mit einem echten Bild $x(0) \sim p_{\text{data}}$ und fügt schrittweise Rauschen hinzu, bis eine Variable $x(T)$ entsteht, die sich einer einfachen Verteilung annähert, normalerweise einer Gauß-Verteilung. Konzeptuell:

$$
x(0) \longrightarrow x(t) \longrightarrow x(T) \approx \mathcal{N}(0,I)
$$

Dieser Prozess ist einfach, weil wir ihn selbst definieren. Wir können wählen, wie viel Rauschen hinzugefügt wird und wie sich dieses Rauschen mit der Zeit entwickelt.

Der umgekehrte Prozess ist der eigentlich generative:

$$
x(T) \longrightarrow x(t) \longrightarrow x(0)
$$

Wir gehen von reinem Rauschen aus und entfernen Rauschen, bis wir ein plausibles Bild erhalten. Doch hier taucht die entscheidende Frage auf: Woher weiß das Modell, wohin sich eine verrauschte Probe bewegen soll?

Die Antwort ist der *Score*:

$$
\nabla_x \log p_t(x)
$$

Dieser Vektor zeigt in die Richtung, in der die logarithmische Dichte der zum Zeitpunkt $t$ gestörten Daten am schnellsten zunimmt. Intuitiv gesagt: Wenn $x(t)$ ein verrauschtes Bild ist, zeigt der Score an, in welchen Bereich des Raums es sich verschieben sollte, um unter der Verteilung von Bildern mit diesem Rauschniveau wahrscheinlicher zu sein.

**Erste wichtige Erkenntnis:** Das Modell muss nicht direkt die vollständige Verteilung $p_t(x)$ lernen. Es genügt, ihr Gradientenfeld zu lernen. Statt sich zu fragen „Wie hoch ist die genaue Wahrscheinlichkeit dieses Bildes?“, lernt es „Wohin sollte ich mich bewegen, um in einer wahrscheinlicheren Region zu sein?“.

---

## 3. Der Vorwärtsprozess als SDE

In kontinuierlicher Zeit wird der Vorwärts-Diffusionsprozess durch eine stochastische Differentialgleichung ausgedrückt:

$$
dx = f(x,t)\,dt + g(t)\,dW(t)
$$

wobei:

- $x(t)$ die Probe zum Zeitpunkt $t$ darstellt.
- $f(x,t)$ der Drift-Term ist, der eine deterministische Entwicklung einführt.
- $g(t)$ die Intensität des Rauschens steuert.
- $W(t)$ ein Standard-Wiener-Prozess ist, das heißt eine Brownsche Bewegung.

Die SDE kombiniert zwei Effekte. Der Term $f(x,t)\,dt$ treibt die Probe auf deterministische Weise, während $g(t)\,dW(t)$ Zufälligkeit einführt. Im Kontext der Diffusion ist genau diese Zufälligkeit der Mechanismus, der strukturierte Daten in Rauschen verwandelt.

### 3.1. Variance Exploding SDE

Die VE-SDE wird definiert als:

$$
dx = \sqrt{\frac{d[\sigma^2(t)]}{dt}}\,dW(t)
$$

Hier gibt es keinen deterministischen Drift, da $f(x,t)=0$ ist. Das Datum behält sein Zentrum, aber seine Varianz nimmt mit der Zeit zu. Deshalb heißt sie *Variance Exploding*: Die Varianz wächst fortschreitend, bis das ursprüngliche Signal vom Rauschen dominiert wird.

Der Übergang des Vorwärtsprozesses hat Gaußsche Form:

$$
p_{0t}(x(t)\mid x(0)) =
\mathcal{N}\left(x(t);\,x(0),\, [\sigma^2(t)-\sigma^2(0)]I\right)
$$

In dieser Familie wird das Datum gestört, indem zunehmend intensiveres Rauschen um das ursprüngliche Bild hinzugefügt wird.

### 3.2. Variance Preserving SDE

Die VP-SDE hat die Form:

$$
dx = -\frac{1}{2}\beta(t)x(t)\,dt + \sqrt{\beta(t)}\,dW(t)
$$

Im Unterschied zur VE-SDE gibt es hier tatsächlich einen Drift. Der Term

$$
-\frac{1}{2}\beta(t)x(t)
$$

zieht das Signal fortschreitend zusammen, während der stochastische Term Rauschen hinzufügt. Die Kombination ist so konzipiert, dass sie die globale Varianz erhält, wenn die Daten normalisiert sind.

Ihr Übergang ist ebenfalls Gaußsch:

$$
p_{0t}(x(t)\mid x(0)) =
\mathcal{N}\left(
x(t);
x(0)e^{-\frac{1}{2}\int_0^t \beta(s)\,ds},
\left[1-e^{-\int_0^t \beta(s)\,ds}\right]I
\right)
$$

Diese Gleichung ist sehr aufschlussreich: Der Mittelwert des Bildes verblasst mit dem Exponentialfaktor, während die Varianz des Rauschens komplementär zunimmt. Das Signal verschwindet, aber nicht abrupt; es löst sich entlang einer kontrollierten Trajektorie auf.

### 3.3. Sub-VP SDE

Die sub-VP-SDE modifiziert den Diffusionsterm:

$$
dx =
-\frac{1}{2}\beta(t)x(t)\,dt
+
\sqrt{\beta(t)\left(1-e^{-2\int_0^t \beta(s)\,ds}\right)}\,dW(t)
$$

Diese Variante behält eine ähnliche Struktur wie die VP-SDE bei, passt aber die Menge des zu jedem Zeitpunkt eingeführten Rauschens an. In der Praxis kann sie bessere Ergebnisse bezüglich der Likelihood liefern, ohne dass dies zwangsläufig mit der besten wahrgenommenen Qualität übereinstimmt.

---

## 4. Die mathematische Wende: eine Diffusion umkehren

Bis hierher haben wir nur definiert, wie man ein Bild zerstört. Der generative Teil erscheint, wenn wir das Ergebnis von Anderson über zeitlich umgekehrte Diffusionsprozesse verwenden [^anderson1982reverse]. Wenn der Vorwärtsprozess ist:

$$
dx = f(x,t)\,dt + g(t)\,dW(t)
$$

dann existiert ein umgekehrter Prozess, der es erlaubt, dieselben Randverteilungen in entgegengesetzter Richtung zu durchlaufen:

$$
dx =
\left[
f(x,t) - g(t)^2 \nabla_x \log p_t(x)
\right]dt
+
g(t)\,d\bar{W}(t)
$$

wobei $d\bar{W}(t)$ die Brownsche Bewegung in umgekehrter Zeit darstellt.

Diese Gleichung enthält den Kern der score-basierten Diffusionsmodelle. Alles Bekannte des Vorwärtsprozesses erscheint in $f$ und $g$. Das Unbekannte konzentriert sich in einem einzigen Term:

$$
\nabla_x \log p_t(x)
$$

Das heißt, um die Diffusion umzukehren, müssen wir nicht die gesamte Dynamik von Grund auf lernen. Wir müssen den Score der gestörten Verteilung zu jedem Zeitpunkt schätzen.

**Zweite wichtige Erkenntnis:** Die Generierung reduziert sich darauf, ein Vektorfeld zu lernen. Wenn das Modell weiß, wie es bei jedem Rauschniveau schätzen kann, wohin die Dichte der Bilder zunimmt, kann es eine Probe vom Rauschen zu einer Region führen, in der realistische Bilder existieren.

In der Praxis wird ein neuronales Netz $s_\theta(x,t)$ trainiert, um Folgendes anzunähern:

$$
s_\theta(x,t) \approx \nabla_x \log p_t(x)
$$

Ersetzt man den echten Score durch den gelernten Score, bleibt die inverse SDE:

$$
dx =
\left[
f(x,t) - g(t)^2 s_\theta(x,t)
\right]dt
+
g(t)\,d\bar{W}(t)
$$

Dies ist die Gleichung, die während des Samplings diskretisiert wird.

---

## 5. Wie man den Score trainiert, ohne die Dichte zu kennen

Auf den ersten Blick scheint es, als hätten wir ein zirkuläres Problem. Wir wollen ein Netz trainieren, um $\nabla_x \log p_t(x)$ anzunähern, aber wir kennen $p_t(x)$ nicht. Wenn wir die Verteilung nicht kennen, wie erhalten wir dann das Trainingslabel?

Der Schlüssel liegt darin, nicht direkt den unbekannten Randscore zu verwenden, sondern den bedingten Score des Störungsprozesses, den wir tatsächlich kennen, weil wir den Vorwärtsprozess selbst definiert haben.

Für viele in der Diffusion verwendete SDEs lässt sich der Vorwärtsübergang als Gauß-Verteilung schreiben:

$$
p_{0t}(x(t)\mid x(0)) =
\mathcal{N}\left(x(t);\mu_t(x(0)),\sigma_t^2 I\right)
$$

wobei $\mu_t(x(0))$ der Mittelwert zum Zeitpunkt $t$ ist und $\sigma_t^2$ die Varianz des hinzugefügten Rauschens.

Nun berechnen wir den Score dieser Gauß-Verteilung bezüglich $x(t)$. Die logarithmische Dichte, ohne die Konstanten, die nicht von $x(t)$ abhängen, ist:

$$
\log p_{0t}(x(t)\mid x(0))
=
-\frac{1}{2\sigma_t^2}
\left\|x(t)-\mu_t(x(0))\right\|_2^2
+
C
$$

Ableiten nach $x(t)$ ergibt:

$$
\nabla_{x(t)} \log p_{0t}(x(t)\mid x(0))
=
-\frac{x(t)-\mu_t(x(0))}{\sigma_t^2}
$$

Diese Gleichung ist einer der elegantesten Punkte der Methode. Wenn wir eine verrauschte Probe $x(t)$ aus einem sauberen Bild $x(0)$ erzeugen, kennen wir genau, wie der bedingte Score aussehen sollte, der zurück auf den Mittelwert der Störung zeigt.

**Dritte wichtige Erkenntnis:** Der Prozess des Rauschhinzufügens selbst erzeugt die Trainingslabels. Wir müssen keine Bilder annotieren, $p_{\text{data}}(x)$ nicht kennen, noch eine unlösbare Dichte berechnen. Es genügt, ein Bild zu nehmen, es zu stören und dem Netz beizubringen, den Vektor vorherzusagen, der diese Störung rückgängig macht.

Die Verlustfunktion des *Denoising Score Matching* in kontinuierlicher Zeit lautet:

$$
\theta^*
=
\arg\min_\theta
\mathbb{E}_{t \sim \mathcal{U}(0,T)}
\left[
\lambda(t)
\mathbb{E}_{x(0)\sim p_{\text{data}}}
\mathbb{E}_{x(t)\mid x(0)}
\left[
\left\|
s_\theta(x(t),t)
-
\nabla_{x(t)}
\log p_{0t}(x(t)\mid x(0))
\right\|_2^2
\right]
\right]
$$

Setzt man den gaußschen bedingten Score ein:

$$
\left\|
s_\theta(x(t),t)
+
\frac{x(t)-\mu_t(x(0))}{\sigma_t^2}
\right\|_2^2
$$

Das Netz lernt für jedes Rauschniveau, wie eine gestörte Probe korrigiert werden muss, um zu einer Zone hoher Wahrscheinlichkeit zurückzukehren.

---

## 6. Vom Score zum Sampling: wie ein Bild entsteht

Sobald $s_\theta(x,t)$ trainiert ist, besteht die Generierung darin, die inverse SDE von $t=T$ bis $t=0$ numerisch zu lösen. Man initialisiert:

$$
x(T) \sim p_T(x)
$$

normalerweise eine Gauß-Verteilung, und wendet einen Integrator in umgekehrter Zeit an.

### 6.1. Euler-Maruyama

Euler-Maruyama ist der grundlegende Integrator für SDEs. Diskretisieren wir die Zeit als $t_N=T,\dots,t_0=0$, kann ein umgekehrter Schritt interpretiert werden als:

$$
x_{i-1}
=
x_i
+
\left[
f(x_i,t_i)
-
g(t_i)^2 s_\theta(x_i,t_i)
\right]\Delta t
+
g(t_i)\sqrt{|\Delta t|}\,z_i
$$

mit:

$$
z_i \sim \mathcal{N}(0,I)
$$

Diese Methode kombiniert eine deterministische, vom Score geleitete Korrektur mit einem zufälligen Term, der die stochastische Natur des Prozesses beibehält.

### 6.2. Predictor-Corrector

Die Predictor-Corrector-*Sampler* fügen eine zweite Idee hinzu. Der Predictor bewegt sich in umgekehrter Zeit vorwärts, zum Beispiel mit Euler-Maruyama. Der Corrector hingegen verfeinert die Probe auf demselben Rauschniveau durch Schritte, die von der Langevin-Dynamik inspiriert sind.

Der Corrector kann als eine Art zu sagen betrachtet werden: „Bevor wir zum nächsten Rauschniveau übergehen, passen wir die Probe noch etwas an, damit sie unter $p_t$ wahrscheinlicher wird“. Dies verbessert meist die visuelle Qualität, weil nicht nur in Richtung weniger Rauschen fortgeschritten wird, sondern auch die Position innerhalb jeder Zwischenverteilung korrigiert wird.

### 6.3. Probability Flow ODE

Der SDE-Rahmen hat eine besonders interessante Eigenschaft: Es existiert eine deterministische ODE, deren Trajektorien dieselben Randverteilungen teilen wie die SDE. Diese *Probability Flow ODE* wird geschrieben als:

$$
dx =
\left[
f(x,t)
-
\frac{1}{2}g(t)^2\nabla_x \log p_t(x)
\right]dt
$$

Ersetzt man den echten Score durch den gelernten:

$$
dx =
\left[
f(x,t)
-
\frac{1}{2}g(t)^2s_\theta(x,t)
\right]dt
$$

**Vierte wichtige Erkenntnis:** Dasselbe Score-Modell ermöglicht zwei Arten der Generierung. Eine stochastische, mittels der inversen SDE, und eine deterministische, mittels der Probability-Flow-ODE. Erstere behält Rauschen während des Samplings bei; letztere verwandelt eine Anfangsbedingung in eine einzige Trajektorie.

Die ODE erlaubt auch die Schätzung von Likelihoods mittels der instantanen Formel des Variablenwechsels [^chen2018neuralode]. Wenn $p_t(x(t))$ sich gemäß einer ODE $dx/dt = v_t(x)$ entwickelt, dann gilt:

$$
\frac{d \log p_t(x(t))}{dt}
=
-\nabla_x \cdot v_t(x(t))
$$

Bei SDE-basierter Diffusion erlaubt dies die Berechnung von Metriken wie der NLL oder den *Bits Per Dimension* (BPD), etwas, das nicht alle generativen Modelle auf natürliche Weise bieten.

---

## 7. Bedingte Generierung: den Prozess auf eine Klasse ausrichten

Die unbedingte Generierung erzeugt Bilder, ohne eine Kategorie festzulegen. Um den Prozess auf eine Klasse $y$ zu bedingen, kann ein zeitabhängiger Hilfsklassifikator $p_\phi(y\mid x(t),t)$ verwendet werden.

Die Herleitung geht von Bayes aus:

$$
p_t(x\mid y)
=
\frac{p_t(y\mid x)p_t(x)}{p_t(y)}
$$

Wir nehmen Logarithmen:

$$
\log p_t(x\mid y)
=
\log p_t(y\mid x)
+
\log p_t(x)
-
\log p_t(y)
$$

Wir leiten nach $x$ ab:

$$
\nabla_x \log p_t(x\mid y)
=
\nabla_x \log p_t(x)
+
\nabla_x \log p_t(y\mid x)
$$

da $p_t(y)$ nicht von $x$ abhängt.

Diese Gleichung hat eine sehr klare Interpretation: Der bedingte Score ist der unbedingte Score plus eine zusätzliche Kraft, die das Bild in Richtung der gewünschten Klasse drängt.

In der Implementierung wird er angenähert als:

$$
\nabla_x \log p_t(x\mid y)
\approx
s_\theta(x,t)
+
w \nabla_x \log p_\phi(y\mid x,t)
$$

wobei $w$ die Intensität der Führung steuert. So lautet die bedingte inverse SDE:

$$
dx =
\left[
f(x,t)
-
g(t)^2
\left(
s_\theta(x,t)
+
w\nabla_x \log p_\phi(y\mid x,t)
\right)
\right]dt
+
g(t)d\bar{W}(t)
$$

Praktisch gesehen liefert das Score-Modell allgemeines Wissen darüber, wie Bilder aussehen, während der Klassifikator eine semantische Präferenz einführt: „Ich möchte, dass dieses Bild am Ende einem Flugzeug, einem Hund, einem Schiff oder einem Frosch ähnelt.“

---

## 8. Imputation: nur erzeugen, was fehlt

Dieselbe Idee lässt sich auch nutzen, um verborgene Regionen eines Bildes zu imputieren. Sei $m$ eine binäre Maske, wobei $m=1$ bekannte Pixel anzeigt und $m=0$ zu rekonstruierende Bereiche.

Während des inversen Samplings schlägt das Modell ein vollständiges Bild vor. In jedem Schritt wird jedoch der bekannte Teil durch eine gestörte Version des Originalbildes auf demselben Rauschniveau ersetzt. Die Aktualisierung lässt sich ausdrücken als:

$$
x'_{t_{i-1}}
=
m \odot \tilde{x}_{t_{i-1}}
+
(1-m)\odot x_{t_{i-1}}
$$

wobei:

- $\tilde{x}_{t_{i-1}}$ das Originalbild ist, gestört durch den direkten Prozess bis zum Zeitpunkt $t_{i-1}$.
- $x_{t_{i-1}}$ die vom Modell erzeugte Stichprobe ist.
- $\odot$ das elementweise Produkt bezeichnet.

Die Interpretation ist einfach: Das Modell hat Freiheit, die unbekannten Regionen zu erfinden, muss aber die beobachtete Information respektieren. Diese Strategie verwandelt ein unbedingtes generatives Modell in ein durch Kontext bedingtes Rekonstruktionswerkzeug.

---

## 9. Implementierung: Theorie übersetzt in Komponenten

Die Implementierung wurde so organisiert, dass jedes mathematische Konzept eine direkte Entsprechung im Code hatte. Die SDEs wurden als Objekte implementiert, die in der Lage sind, ihre Drift- und Diffusionskoeffizienten, ihre Übergangsverteilungen und die für das Training des Scores notwendigen Operationen bereitzustellen.

Die wichtigsten Elemente waren:

- SDE-Familien: VE-SDE, VP-SDE und Sub-VP SDE.
- Lineare und kosinusförmige Noise-Schedules.
- Score-Modelle basierend auf U-Net.
- Sampler: Euler-Maruyama, Predictor-Corrector, Probability Flow ODE und exponentielle Integratoren.
- Metriken: FID, Inception Score, NLL und BPD.
- Erweiterungen für bedingte Generierung und Imputation.

Die U-Net-Architektur erweist sich als besonders geeignet, weil sie lokale und globale Information kombiniert. In Bildern müssen kleine Strukturen — Kanten, Texturen, Konturen — mit Beziehungen größerer Skala wie Form, Objekt und Hintergrund koexistieren. Die *Skip*-Verbindungen ermöglichen es, räumliches Detail wiederherzustellen, während die tiefen Schichten Kontext erfassen.

---

## 10. Ergebnisse: was man bei der Bilderzeugung beobachtet

Die Experimente wurden hauptsächlich mit CIFAR-10 und MNIST durchgeführt. Um das Verhalten des Systems zu veranschaulichen, werden repräsentative Ergebnisse mit einer linearen VP-SDE und verschiedenen Sampling-Methoden gezeigt.

<figure id="fig:evolucion_muestras_vp_lineal">
<img src="/assets/articles/difussion_models/evolucion_muestras_vp_lineal.png" />
<figcaption>Entwicklung der Generierung mit linearer VP-SDE und Predictor-Corrector-Sampler auf CIFAR-10. Von links nach rechts entwickelt sich die Stichprobe vom anfänglichen Rauschen zu einem finalen Bild.</figcaption>
</figure>

Die Sequenz zeigt visuell die zentrale Idee der Methode: Die Generierung entsteht nicht schlagartig. Zuerst bilden sich Farbmassen, danach Texturen, später erkennbare Strukturen und schließlich klarer definierte Details. Dieses Verhalten passt zur probabilistischen Interpretation: Am Anfang werden sehr verrauschte und glatte Verteilungen aufgelöst; am Ende wird auf niedrigen Rauschniveaus gearbeitet, wo feine Details wichtig sind.

<figure id="fig:comparativa_samplers_vp_lineal">
<img src="/assets/articles/difussion_models/comparativa_samplers_vp_lineal.png" />
<figcaption>Vergleich der finalen Bilder, die mit verschiedenen Samplern erzeugt wurden: Predictor-Corrector, Exponential Integrator, Probability Flow ODE und Euler-Maruyama.</figcaption>
</figure>

Bei den Vergleichen neigt Predictor-Corrector dazu, visuell stabilere Ergebnisse zu liefern, weil es zeitlichen Fortschritt und lokale Verfeinerung kombiniert. Euler-Maruyama ist einfacher, benötigt aber meist mehr Schritte. Die Probability Flow ODE bietet eine deterministische Trajektorie und ist besonders relevant, wenn es darum geht, Likelihoods zu schätzen.

---

## 11. Metriken: FID, IS und BPD erzählen nicht dieselbe Geschichte

Die quantitative Bewertung erfolgte mit gängigen Metriken für generative Modelle: Fréchet Inception Distance (FID), Inception Score (IS) und Bits Per Dimension (BPD).

<figure id="fig:metricas_fid_bpd">
<div class="minipage">
<img src="/assets/articles/difussion_models/fid.jpeg" />
</div>
<div class="minipage">
<img src="/assets/articles/difussion_models/bpd.jpeg" />
</div>
<figcaption>Verteilung der Ergebnisse für FID und BPD zwischen verschiedenen SDE-Konfigurationen.</figcaption>
</figure>

<figure id="fig:metricas_is">
<div class="minipage">
<img src="/assets/articles/difussion_models/is_media.jpeg" />
</div>
<div class="minipage">
<img src="/assets/articles/difussion_models/is_std.jpeg" />
</div>
<figcaption>Verteilung des Inception Score und seiner internen Standardabweichung.</figcaption>
</figure>

Die Ergebnisse zeigen einen wichtigen Punkt: Nicht alle Metriken belohnen dasselbe. In den durchgeführten Läufen erzielte lineare SubVP-SDE die besten relativen perzeptuellen Ergebnisse, mit niedrigerem FID und höherem IS. Dagegen zeichneten sich die Varianten mit kosinusförmigem Schedule bei BPD aus, insbesondere kosinusförmige SubVP-SDE.

Dieser Unterschied ist relevant, weil BPD die probabilistische Anpassung misst, während FID und IS näher an perzeptueller Qualität und semantischer Diversität liegen. Ein Modell kann den Daten eine gute Wahrscheinlichkeit zuweisen und trotzdem nicht die visuell überzeugendsten Stichproben erzeugen. Diese Entkopplung zwischen Likelihood und perzeptueller Qualität ist eine der wichtigsten Lektionen bei der Bewertung generativer Modelle.

**Fünfte wichtige Erkenntnis:** Das „beste“ Modell hängt davon ab, was „am besten“ bedeutet. Wenn wir visuell überzeugende Bilder suchen, können FID und IS aussagekräftiger sein. Wenn wir probabilistische Modellierung und Likelihood suchen, liefert BPD eine andere Lesart. Die Bewertung muss mehrere Metriken betrachten, nicht nur eine.

---

## 12. Bedingte Generierung

Für die bedingte Generierung wurde ein zeitabhängiger Klassifikator verwendet, basierend auf einer Wide-ResNet-Architektur, trainiert darauf, Klassen zu erkennen, auch wenn das Bild durch Rauschen gestört ist.

<figure id="fig:classifier_arch">
<img src="/assets/articles/difussion_models/classifier_architecture_image.png" style="width:70.0%" />
<figcaption>Schematische Architektur des zeitabhängigen Klassifikators, der zur Steuerung der bedingten Generierung verwendet wird.</figcaption>
</figure>

Der Klassifikator liefert den Term $\nabla_x \log p_\phi(y\mid x,t)$, der den effektiven Score während des Samplings modifiziert. So ist die Generierung nicht mehr vollständig frei, sondern auf eine konkrete Klasse ausgerichtet.

<figure id="fig:conditional_comparison">
<img src="/assets/articles/difussion_models/final_conditional_comparison.png" />
<figcaption>Vergleich der Ergebnisse bei der bedingten Generierung für verschiedene Konfigurationen.</figcaption>
</figure>

Die bedingte Generierung zeigte, dass der SDE-Rahmen nicht nur dazu dient, zufällige Bilder zu erzeugen, sondern auch das Ergebnis teilweise zu kontrollieren. Der Grad der Kontrolle hängt vom Klassifikator, der Guidance-Skala und der Qualität des Score-Modells ab. Eine zu niedrige Skala kann die Stichprobe nicht ausreichend lenken; eine zu hohe Skala kann Artefakte erzwingen oder die Diversität verringern.

---

## 13. Bildimputation

Bei der Imputationsaufgabe wurden von MNIST-Ziffern abgeleitete Masken auf CIFAR-10-Bilder angewendet. Das Modell musste die verborgenen Regionen rekonstruieren und dabei Kohärenz mit dem sichtbaren Kontext bewahren.

<figure id="fig:imputacion_cifar_mnist_3etapas">
<img src="/assets/articles/difussion_models/imputacion_cifar_mnist_3etapas.png" />
<figcaption>Imputation von Regionen in CIFAR-10 unter Verwendung von MNIST-Masken. Oben: maskierte Bilder. Mitte: imputiertes Ergebnis. Unten: originales Referenzbild.</figcaption>
</figure>

Die Imputation ist interessant, weil sie zeigt, dass das Modell nicht nur aus reinem Rauschen generiert. Es kann auch als Prior-Verteilung über natürliche Bilder fungieren: Wenn Information fehlt, schlägt es Inhalt vor, der mit dem Beobachteten kompatibel ist. Mit anderen Worten: Das Modell „gewinnt“ nicht die exakte verlorene Region zurück, sondern erzeugt eine plausible Rekonstruktion unter der gelernten Verteilung.

---

## 14. Konvergenz während des Trainings

Ebenfalls verglichen wurde die Entwicklung des Loss während der ersten 50 Epochen für verschiedene SDE-Konfigurationen.

<figure id="fig:loss_curves_violin_sdes_50epochs">
<div class="minipage">
<img src="/assets/articles/difussion_models/loss_curves_all_sdes.png" />
</div>
<div class="minipage">
<img src="/assets/articles/difussion_models/loss_violin_plot_all_sdes.png" />
</div>
<figcaption>Entwicklung und Verteilung des Loss während der ersten 50 Epochen für verschiedene SDE-Konfigurationen.</figcaption>
</figure>

<figure id="fig:loss_barplots_sdes_50epochs">
<div class="minipage">
<img src="/assets/articles/difussion_models/loss_final_barplot_all_sdes.png" />
</div>
<div class="minipage">
<img src="/assets/articles/difussion_models/loss_average_barplot_all_sdes.png" />
</div>
<figcaption>Vergleich des finalen Loss und des durchschnittlichen Loss während des anfänglichen Trainings.</figcaption>
</figure>

Die Kurven legen nahe, dass lineare SubVP-SDE und VE-SDE früher niedrige Loss-Werte erreichen. Diese Lesart sollte jedoch mit Vorsicht interpretiert werden: eine bessere Minimierung des Trainingsloss garantiert nicht automatisch bessere Bilder. Der Loss misst die Genauigkeit bei der Schätzung des Scores unter dem gewählten Störungsschema, während die generative Qualität auch vom Sampler, vom Noise-Schedule, von der Architektur und davon abhängt, wie sich Fehler während der inversen Integration fortpflanzen.

---

## 15. Einschränkungen

Das Projekt weist Einschränkungen auf, die man explizit benennen sollte. Das Training der Score-Modelle und der zeitabhängigen Klassifikatoren ist rechnerisch aufwendig, was die Anzahl der Epochen und die Erschöpfung einiger Vergleiche einschränkte. Die Bewertung von Metriken wie FID erfolgte auf Teilmengen der Daten, sodass die absoluten Werte variieren könnten. Hinzu kommt die inhärente Volatilität der Bewertungsmetriken für generative Modelle: Keine Metrik allein erfasst alle wünschenswerten Aspekte der Generierung.

---

## 16. Fazit: was der SDE-Rahmen klarstellt

Die Formulierung mittels SDEs bietet eine besonders klare Art, Diffusionsmodelle zu verstehen. Der direkte Prozess wandelt Daten mittels einer bekannten Dynamik in Rauschen um. Der inverse Prozess ermöglicht die Generierung von Daten, erfordert aber die Kenntnis des Scores der Zwischenverteilungen. Da dieser Score nicht verfügbar ist, wird er mit einem neuronalen Netz approximiert, das mittels *Denoising Score Matching* trainiert wird.

Das Mächtigste an diesem Ansatz ist, dass viele Teile auf natürliche Weise zusammenpassen:

- Die direkte SDE definiert, wie die Daten korrumpiert werden.
- Der Score gibt an, wie man zu Regionen hoher Wahrscheinlichkeit zurückkehrt.
- Die inverse SDE verwandelt Rauschen in Bilder.
- Die Probability Flow ODE bietet eine deterministische Alternative und ermöglicht die Schätzung von Likelihoods.
- Die Klassifikator-Guidance ermöglicht bedingte Generierung.
- Die Imputation ergibt sich aus der Kombination von inversem Sampling mit Beschränkungen auf bekannte Pixel.

Aus experimenteller Sicht zeigen die Ergebnisse, dass es keine einzige universell überlegene Konfiguration gibt. Lineare SubVP-SDE verhielt sich besser bei perzeptuellen Metriken wie FID und IS, während kosinusförmige Varianten bei BPD herausstachen. Dieser Unterschied verstärkt eine wichtige Idee: Die Bewertung generativer Modelle erfordert es, gleichzeitig visuelle Qualität, Diversität, Stabilität, Rechenkosten und probabilistische Anpassung zu betrachten.

Insgesamt ermöglichte diese Arbeit, Theorie, Implementierung und Experimentierung in einem einzigen Rahmen zu verbinden: von der mathematischen Herleitung des Scores bis zur Bilderzeugung, der bedingten Generierung und der Imputation.

---

## Referenzen

[^song2021scorebased_sde]: Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021). Score-Based Generative Modeling through Stochastic Differential Equations. In International Conference on Learning Representations (ICLR).

[^songermon2019ncsn]: Song, Y., & Ermon, S. (2019). Generative Modeling by Estimating Gradients of the Data Distribution. In Advances in Neural Information Processing Systems (NeurIPS).

[^sohl2015deep]: Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., & Ganguli, S. (2015). Deep Unsupervised Learning using Nonequilibrium Thermodynamics. In International Conference on Machine Learning (ICML).

[^ho2020denoising]: Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. In Advances in Neural Information Processing Systems (NeurIPS).

[^krizhevsky2009cifar]: Krizhevsky, A., & Hinton, G. (2009). Learning multiple layers of features from tiny images. Technical report, University of Toronto.

[^lecun1998gradient]: LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278-2324.

[^goodfellow2014generative]: Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., ... & Bengio, Y. (2014). Generative adversarial nets. In Advances in neural information processing systems (NeurIPS).

[^kingma2013auto]: Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114.

[^dinh2016density]: Dinh, L., Sohl-Dickstein, J., & Bengio, S. (2016). Density estimation using Real NVP. arXiv preprint arXiv:1605.08803.

[^kingma2018glow]: Kingma, D. P., & Dhariwal, P. (2018). Glow: Generative flow with invertible 1x1 convolutions. In Advances in neural information processing systems (NeurIPS).

[^anderson1982reverse]: Anderson, B. D. (1982). Reverse-time diffusion equation models. Stochastic Processes and their Applications, 13(3), 313-326.

[^vincent2011connection]: Vincent, P. (2011). A connection between score matching and denoising autoencoders. Neural computation, 23(7), 1661-1674.

[^chen2018neuralode]: Chen, R. T., Rubanova, Y., Bettencourt, J., & Duvenaud, D. K. (2018). Neural ordinary differential equations. In Advances in neural information processing systems (NeurIPS).

[^hyvarinen2005scorematching]: Hyvärinen, A. (2005). Estimation of non-normalized statistical models by score matching. Journal of Machine Learning Research, 6(Apr), 695-709.

[^nicholdhariwal2021improved]: Nichol, A. Q., & Dhariwal, P. (2021). Improved denoising diffusion probabilistic models. In International Conference on Machine Learning (ICML).

[^ronneberger2015unet]: Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention (MICCAI).

[^dhariwal2021diffusion]: Dhariwal, P., & Nichol, A. (2021). Diffusion models beat gans on image synthesis. In Advances in Neural Information Processing Systems (NeurIPS).

[^zagoruyko2016wide]: Zagoruyko, S., & Komodakis, N. (2016). Wide residual networks. arXiv preprint arXiv:1605.07146.
