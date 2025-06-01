# Peer Review

## Problem description (2/2)
**Score:** 2
**Motivatie:**
De probleemstelling is duidelijk beschreven in de README. Het project legt helder uit dat het een ML-model bouwt om tweedehands autoprijzen te voorspellen, en waarom dit nuttig is voor zowel kopers als verkopers. De context, het doel en de toepassing zijn goed uitgelegd.

---

## Experiment tracking & model registry (2/2)
**Score:** 2
**Motivatie:**
Het project gebruikt MLflow voor zowel experiment tracking als model registry. Dit blijkt uit de code (`mlflow.set_tracking_uri`, `mlflow.log_metric`, `mlflow.sklearn.log_model` met `registered_model_name`). In de README en documentatie wordt uitgelegd hoe MLflow wordt gebruikt en hoe je de UI kan openen. Er is een aparte MLflow service in Docker Compose.

---

## Workflow orchestration (2/2)
**Score:** 2
**Motivatie:**
De workflow is volledig georchestreerd met Prefect. Er is een Prefect flow (`training_pipeline`) met taken voor data loading, preprocessing, training, evaluatie, logging en monitoring. De flow is gedeployed en kan via de UI of CLI worden uitgevoerd. De documentatie beschrijft duidelijk hoe je de flow moet deployen en runnen.

---

## Model deployment (2/2)
**Score:** 2
**Motivatie:**
De model deployment is gecontaineriseerd. De webapp draait in een aparte Docker container en laadt het model uit de MLflow registry. De deployment is niet enkel lokaal, maar kan eenvoudig naar de cloud gebracht worden door de container te deployen. De instructies en Dockerfiles zijn aanwezig.

---

## Model monitoring (1/2)
**Score:** 1
**Motivatie:**
Er is monitoring met Evidently: er wordt een drift report gegenereerd en opgeslagen. Er is ook een Prefect task die drift detecteert en een retraining kan triggeren (al staat deze code nu uit-gecommentarieerd). Er worden nog geen automatische alerts verstuurd of volledig automatische retraining uitgevoerd, maar de basis voor conditionele workflows is aanwezig.

---

## Reproducibility (2/2)
**Score:** 2
**Motivatie:**
De README bevat duidelijke, stapsgewijze instructies om het project te runnen, inclusief dependency management via `requirements.txt` en Docker. Alle benodigde data (`car_prices.csv`) zit in de repo. De versies van alle dependencies zijn gespecificeerd.
**Opmerking:** Ik was niet in staat om het project te rebuilden en te openen in een container omdat mijn gebruikersnaam op mijn computer spaties bevat. Hierdoor kon ik geen diepgaande test doen van de reproduceerbaarheid.

---

## Opmerking
- **Sterk punt:** Zeer uitgebreide en duidelijke documentatie, goede code structuur, en alles is gecontaineriseerd en geautomatiseerd.

---

**Totaalscore:**
- Problem description: 2
- Experiment tracking & model registry: 2
- Workflow orchestration: 2
- Model deployment: 2
- Model monitoring: 1
- Reproducibility: 2
**Totaal:** 11/12

**Algemene feedback:**
Sterk project met goede MLOps-praktijken, duidelijke uitleg en reproduceerbare setup. Kleine verbeteringen in de README zouden het helemaal af maken.
Let op: door technische beperkingen (gebruikersnaam met spaties) kon ik de container niet zelf testen.
