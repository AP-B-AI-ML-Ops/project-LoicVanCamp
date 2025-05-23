# Monitoring

## Hoe werkt de monitoring?

- **Batch predictions** worden opgeslagen als CSV in `batch-data/report/students/`.
- **monitoring.py** vergelijkt referentie- en actuele data met Evidently en slaat de resultaten op als CSV én in de database (`mlflow_db`, tabel `evidently_metrics`).
- **Grafana** is gekoppeld aan de database en toont dashboards op basis van deze metrics.

## Resultaten vinden

- **CSV output:** `batch-data/report/evidently_metrics.csv`
- **Database:** tabel `evidently_metrics` in `mlflow_db`
- **Grafana dashboards:** [http://localhost:3400](http://localhost:3400)

## Monitoring draaien

```bash
python [monitoring.py](http://_vscodecontentref_/1)
```