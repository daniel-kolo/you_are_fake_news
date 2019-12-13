# Házi feladat dokumentáció 

Kezdetben 3 fős csapatként megajánlott jegyet célzó házi feladatként azt az feladatot tűztük ki célul, hogy egy olyan ágenst hozunk létre, ami képes egy adott szócikkról eldönteni, hogy fake news-e vagy sem. Tehát egy NLP (natural language processing) bináris klasszifikáló háló megtervezése, és impementálása volt a cél. Többféle megközelitéssel is próbálkoztunk, de végül az idő rövidsége, és egy csapattag elvesztése után Gyires-Tóth Bálint Tanár Úrral konzultálva a téma valamelyest módosult, azt a feladatot kaptuk, hogy járjuk körbe a fake news detektálás, és generálás témakörét, mind elméletben mind gyakorlatban. A mi saját munkánk, és a cutting edge megoldásokat is érdemes vizsgálnunk. 
A vizsgálat, és a próbák során Python 3-at, és a Keras frameworkot fogjuk használni.

## A file-ok és adatok bemutatása a projektben

A gihubra maximum 100mb méretű adat tölthető fel, így a használt adatokat shardoltuk.


| File neve | Leírás |
| ------------- | ------------- |
| fake1 | A fake cikkekből kinyert coprus első harmada  | 
| fake2  | A fake cikkekből kinyert coprus második harmada  |
| fake3  | A fake cikkekből kinyert coprus harmadik harmada  |
| rel1  | A hiteles cikkekből kinyert coprus első harmada  |
| rel2  | A hiteles cikkekből kinyert coprus második harmada  |
| rel3  | A hiteles cikkekből kinyert coprus harmadik harmada  |
| most_common_words_fake  | A fake corpus 2000 leggyakoribb szava, és előfordulási számosságuk  |
| most_common_words_reliable  | A hiteles corpus 2000 leggyakoribb szava, és előfordulási számosságuk  |
| unique_most_common_words_fake  | A most_common_words_fake szavaiból eltávolítottuk a mindkét halmazban szerplő szavakat  |
| unique_most_common_words_reliable  | A most_common_words_reliable szavaiból eltávolítottuk a mindkét halmazban szerplő szavakat  |


## Download_data.ipynb
Az adatok letöltése az MIT adatbázisából. Az MIT adaszettjében több féle címkével ellátott adat szerepel, nekünk csak a biztosan fake, és a biztosan hiteles adatok kellenek, mivel mi ilyen szempontok szerint klasszifikáló ágens-t vizsgálunk.

## data_slicing.ipynb

A corpusok importálására, és a különböző (már méretben github kompatibilis) file-ok elkészítését, file-ba szerializálását célzó függvények találhatóak ebben a notebook-ban.

## text_preprocess.py

Az cikkek szócikkből corpus-á alakítása során használjuk ezeket a függvényeket. Az URL-ek eltávolítását, a kis/nagy betűk egységes formátumra hozását, a lemmatizációt és a stop word-ök eltávolítását szolgáló scriptek találhatóak itt meg.

## training_preprocess.py

Itt találhatóak a tanítás előkészítéséért felelős scriptek. Előállítják a corpusból a training, és test adat szetteket. Itt készül el a beágyazási mátrix, a fasttext modellhez, valamint a vektorizáció.

## Preprocess_data.ipynb

Itt végezzük el az előző két előkészítő fileban definiált függvényeket a szócikkeken, így áll elő a két corpus, ami már készen áll a tanításra.

## Data analysis.ipynb

Az adatok vizsgálatára, analízisére használt függvényeket tartalmazza a file. További stop word-ök keresését segítheti, érdekes lehet megvizsgálni az első n leggyakoribb szót, majd ez után dönteni az eltávolításukról. A szócikkek hosszúságát is itt vizsgáljuk, mivel később szeretnénk a cikkek analízisében egyenlő hosszúságu mintákat vizsgálni. A vizsgálat szerint a legtöbb cikk az 1000-1200 szavas intervallumba esik, ezért ezt lehet optimális vizsgálni.

## BiLSTM network.ipynb

A tanítást végző neurális háló itt van implementálva, és tanítva.
Ez egy bilateral LSTM (Long short-term memory) neuráli háló, ami igen népszerű az NLP feladatok témakörében.
A tanítást elvégezve látjuk, hogy 10 epoch alatt 97%-os pontosságot értünk el a teszt adatainkon, ami gyanúra ad okot. Valami olyan információ lehet még az adatainkban, ami a neurális háló számára olyan indikátor lehet, ami alapján könnyebben klasszifikál. Hosszas vizsgálat után sem sikerült ezeket az információkat megállapítani, esetleg még arra tudtunk gondolni, hogy az MIT adatai annyira jók, hogy ezeken már elég könnyen tanult be a háló.

