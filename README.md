# Korisničko upustvo

## Podešavanja projekta
Od podešavanja je neophodno prvi put instalirati `conda` okruženje i opciono za potrebe testiranje prekopirati istreirane
modele. Alternativno, umesto preuzimanja modela oni se mogu istrenirati korišćenjem `train_workflow.py` na način opisan u 
nastavku.

### Podešavanja okruženja
Na alfa serveru u direktorijumu `shared/retina/environments/` se nalazi izvezeno okruženje za projekat. Instalirati ga
komandom `conda env create -f saunet-env.yml`.

### Postavljanje modela
Istranirani modeli se mogu preuzeti sa putanje: https://mega.nz/file/nu4SzSIL#i64RmWnhktaG4O8Am6lJF-YSn_rXlVQp1ZLsKoPHiqs

Modeli su trenirani 2-3 epohe i namenjeni su samo testiranju ispravnosti rešenja. Raspakovani direktorijum je potrebno
ubaciti direktno u koren projekta.

## Pokretanje

### Automatizovano testiranje
- Aktivirati instalirano okruženje
- U terminalu se pozicionirati se poddirektorijum `SA_UNet/mlflow`.
- Eksportovati putanju do projekta u promenljivu okruženja `PYTHONPATH` (`export PYTHONPATH=<deo_putanje>/SA_UNet:$PYTHONPATH`)
- Pokrenuti `predict_workflow.py` skript. Ovaj skript će nad svakim od modela u `models` poddirektorijumu projekta
pokrenuti predikciju na svakom od setova (trenutno su to DRIVE, STARE, CHASE i DROPSv1). Rezulati izvršavanja skripta će 
se naći u `results` poddirektorijumu.
- Opciono, pokrenuti `train_workflow.py` skript. Ovaj skript trenira SAUNet model sa podrazumevanim parametrima treninga
nad svakim od gore navedenih setova podataka i čuva modele u `models` poddirektorijum.

### [Opciono] Pokretanje `mlflow ui` alata za grafički pregled rezultata
- Nakon ispoštovanog uputstva za automatsko pokretanje, logovi koje `mlflow ui` alat prikazuje će se naći u 
`SA_UNet/mlflow/mlruns` direktorijumu.
- Pozicionirati se u terminalu u `SA_UNet/mlflow/` direktorijum i pokrenuti komandu `mlflow ui`. Podrazumevano se server
podiže na 127.0.0.1:5000. Proizvoljan port je moguće zadati pri pokretanju opcijom `--port <vrednost>`.

Dobije se nešto lepo poput:

![SA-UNet](mlflowui.png?raw=true "mlflow ui")
 