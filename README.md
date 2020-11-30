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

Potpuno trenirani modeli sa različitim okruženjima se nalaze na /home/shared/retina/models/saunet/trained putanji. 

### Podešavanje

Da bi sve radilo kao konfigurisano, potrebno je da se ulinkuju pravi direktorijumi. To radi bash skirpta 'make_ready' Mora se
pokrenuti samo jednom, i od tog trenutka do daljneg radi bez problema. Naravno, ovo radi samo na
klasteru. Ako nisi na klasteru, moraš postaviti odgovarajuću strukturu koja mu odgovara. 

## Pokretanje

### Automatizovano testiranje
- Aktivirati instalirano okruženje. mlrun ima opciju da se definišu potrebni paketi, pa da sam pravi i pokreće okruženje,
ali će onda svaki put svlačiti tensorflow i slične velike bliblioteke, što traje. Zbog toga sam se odlučila na manuelno
aktiviranje okruženja.
- U terminalu se pozicionirati se poddirektorijum `SA_UNet/mlflow`.
- Eksportovati putanju do projekta u promenljivu okruženja `PYTHONPATH` (`export PYTHONPATH=<deo_putanje>/SA_UNet:$PYTHONPATH`). Ovo se može
i uraditi sa 'make_pythonpath.sh' skriptom. Valja voditi računa da je ovo slučaj kada se skripta mora pokrenuti kao `source make_pythonpath.sh`
pošto inače nema izmene okruženja. 
- Pokrenuti `predict_workflow.py` skript. Ovaj skript će nad specificiranim setom dataset-ova poterati evaluaciju, učitavajući svaki put dogovarajući model, tj. DRIVE-model itd. Rezulati izvršavanja skripta će 
se naći u `results` poddirektorijumu.
- Opciono, pokrenuti `train_workflow.py` skript. Ovaj skript trenira SAUNet model sa podrazumevanim parametrima treninga
nad svakim od gore navedenih setova podataka i čuva modele u `models` poddirektorijum. (trenutno ne radi)

### [Opciono] Pokretanje `mlflow ui` alata za grafički pregled rezultata
- Nakon ispoštovanog uputstva za automatsko pokretanje, logovi koje `mlflow ui` alat prikazuje će se naći u 
`SA_UNet/mlflow/mlruns` direktorijumu na udaljenoj mašini.
- Skinuti direktorijum sa logovima na lokalnu mašinu (`SA_UNet/mlflow/mlruns`).
- Skinuti direktorijum sa rezultatima predikcije (`SA_UNet/results`) na lokalnu mašinu da bi se videli artefakti.
- Kako su neke putanje u logovima absolutne, logovani artefakti neće biti pronađeni na lokalnoj mašini. Pre pregleda je 
potrebno u svim meta.yml datotekama zameniti `</putanja/na/serveru/>SAUNet` zameniti sa `<putanja/na/lokalnoj/masini/>SAUNet`. 
U PyCharm-u se to lako odradi selektovanjem `mlruns` direktorijuma i prečicom `CTRL+SHIFT+R` koja otvara prozor za zamenu
stringa stringom za obeleženi direktorijum.
- Pozicionirati se u terminalu u `SA_UNet/mlflow/` direktorijum na lokalnoj mašini i pokrenuti komandu `mlflow ui`. 
Podrazumevano se server podiže na 127.0.0.1:5000. Proizvoljan port je moguće zadati pri pokretanju opcijom `--port <vrednost>`.

Dobije se nešto lepo poput:

![SA-UNet](mlflowui.png?raw=true "mlflow ui")

Ukoliko se logovi generišu na  