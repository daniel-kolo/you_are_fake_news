# Házi feladat dokumentáció 

Kezdetben 3 fős csapatként megajánlott jegyet célzó házi feladatként azt az feladatot tűztük ki célul, hogy egy olyan ágenst hozunk létre, ami képes egy adott szócikkról eldönteni, hogy fake news-e vagy sem. Tehát egy NLP (natural language processing) bináris klasszifikáló háló megtervezése, és impementálása volt a cél. Többféle megközelitéssel is próbálkoztunk, de végül az idő rövidsége, és egy csapattag elvesztése után Gyires-Tóth Bálint Tanár Úrral konzultálva a téma valamelyest módosult, azt a feladatot kaptuk, hogy járjuk körbe a fake news detektálás, és generálás témakörét, mind elméletben mind gyakorlatban. A mi saját munkánk, és a cutting edge megoldásokat is érdemes vizsgálnunk. 

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