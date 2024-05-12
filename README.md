## Szakdolgozat:

### Fő részek:
* szakdolgozat.ipynb: az ágensek összehasonlítása, és az eredményük elemzése
* data_analysis.ipynb: a szimulációk bemeneti adatainak elemzése
* hyperparameter_tuning mappa: a hiperparaméter finomítás elemzésére
* wrappers mappa: a DQN algoritmushoz kellettek egyedi wrapperek, hogy kompatibilis legyen, azok vannak itt. Illetve a magyar árazáshoz használt Building class.
* reward functions mappa: a kipróbált reward function-ök, végül a rewardFunction3 volt használva
* generate python fájlok, amik a bemeneti adatok generálására voltak használva
* data_preprocessing.ipynb: az adatok előkészítéséért felelős metódust tartalmazza
* SAC_saved és DDPG_saved: az Amazon SageMaker Studio Lab-ről lementett modellek betöltését és elemzését tartalmazzák
* + lokálisan vannak elmentett modellek, környezetek és hiperparaméter optimalizálás eredmények

A használt package-k a requirements.txt-ben találhatóak
