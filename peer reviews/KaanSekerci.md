# Peer Review – Project Kaan Sekerci

## Problem description (1/2)
**Score:** 1
**Motivatie:**
De probleemstelling beschrijft de algemene context en de focus op de MLOps lifecycle, maar mist belangrijke details over de dataset en het exacte predictiedoel. Het blijft onduidelijk wat het model precies voorspelt en hoe de data is opgebouwd.

---

## Experiment tracking & model registry (2/2)
**Score:** 2
**Motivatie:**
MLflow wordt correct gebruikt voor zowel experiment tracking als model registry. De benodigde componenten zijn aanwezig en geïntegreerd in de pipeline.

---

## Workflow orchestration (2/2)
**Score:** 2
**Motivatie:**
Workflow orchestration is volledig uitgewerkt met Prefect. Wel is het verwarrend dat de README niet duidelijk maakt dat het uitvoeren van de flow uit twee aparte commando’s bestaat, en dat de bestandsnaam in de README niet overeenkomt met de werkelijke bestandsnaam. Dit kan tot verwarring leiden bij het opstarten van de workflow.

---

## Model deployment (2/2)
**Score:** 2
**Motivatie:**
Het model wordt containerized gedeployed als web API, wat cloud deployment mogelijk maakt. De deployment is technisch goed uitgewerkt.

---

## Model monitoring (1/2)
**Score:** 1
**Motivatie:**
Er is monitoring aanwezig via Evidently, maar deze is basic en automatiseert geen alerts of conditionele workflows. Daarnaast kunnen verkeerde line-endings in shell scripts de monitoring-service verstoren, wat niet in de documentatie staat.

---

## Reproducibility (1/2)
**Score:** 1
**Motivatie:**
De instructies zijn grotendeels aanwezig, maar er zijn meerdere punten die de reproduceerbaarheid beperken:
- Het hernoemen van `.env.example` naar `.env` wordt niet vermeld.
- De README verwijst naar een niet-bestaand bestand (`flow.py` i.p.v. `flows.py`).
- CMD-commando’s in Dockerfiles moeten soms aangepast worden.
- Shell scripts bevatten mogelijk Windows line-endings, wat tot fouten leidt.
- Er zijn dependency mismatch waarschuwingen en requirements zijn niet gepind.
- De betekenis van de API-output wordt nergens uitgelegd.

---

## Opmerking
Het project is technisch sterk en volgt MLOps best practices, maar de documentatie en kleine technische details beperken de gebruiksvriendelijkheid en reproduceerbaarheid. Duidelijkere instructies, consistente bestandsnamen, uitleg over modeloutput en aandacht voor platformonafhankelijke scripts zouden het project verder verbeteren.

---

## Totaalscore: **9/12**

---

### Samenvatting
Sterk project met een volledige MLOps pipeline en werkende monitoring, maar met tekortkomingen in probleemomschrijving, reproduceerbaarheid en documentatie. Kleine aanpassingen in de README en scripts kunnen de gebruikerservaring aanzienlijk verbeteren.
