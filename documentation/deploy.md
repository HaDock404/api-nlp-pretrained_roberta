## Activer les API nécessaires

```bash
gcloud services enable run.googleapis.com artifactregistry.googleapis.com
```

## Rendre le model sur le bucket public

```bash
cd ./models
zip -r model_roberta.zip model_roberta
zip -r tokenizer_roberta.zip tokenizer_roberta

gsutil cp model_roberta.zip gs://hadock404-models/
gsutil cp tokenizer_roberta.zip gs://hadock404-models/

gsutil acl ch -u AllUsers:R gs://hadock404-models/model_roberta.zip
gsutil acl ch -u AllUsers:R gs://hadock404-models/tokenizer_roberta.zip
```

## Construire et pousser l'image sur Cloud Run

```bash
gcloud builds submit --tag gcr.io/hadock404-project/api-emoxion
```

## Vérifier que l'image est bien disponible

```bash
gcloud container images list
```

## Déployer sur Cloud Run

```bash
gcloud run deploy api-emoxion \
  --image gcr.io/hadock404-project/api-emoxion \
  --platform managed \
  --region northamerica-northeast1 \
  --allow-unauthenticated \
  --memory 2Gi
```

## Adresse

Service URL: https://api-emoxion-782672784164.northamerica-northeast1.run.app/docs