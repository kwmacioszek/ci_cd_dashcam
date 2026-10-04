**Ewaluacja bazowego DETR na zbiorze walidacyjnym**

Źródło wyników: `runs/base_val_gpu.json`. Raport opisuje istniejący wynik; nie wykonano ponownie ewaluacji.

Model: `facebook/detr-resnet-50`, bez dotrenowania na ocenianym zbiorze. Podział: `valid`. Oceniono **935 obrazów**. Ground truth pochodzi z etykiet zbioru.

Urządzenie inferencji zapisane w JSON: `cuda` (GPU; dla PyTorch ROCm także używa się nazwy `cuda`). Obliczenia metryk w sprawdzonym skrypcie odbywają się na CPU. Średni zapisany czas pętli inferencji, wraz z przygotowaniem obrazu i wyniku: 0.0984 s/obraz.

Próg pewności dla precision, recall i F1: **0.5**. Próg dopasowania IoU: **0,5**, zgodnie z domyślnym argumentem `iou_thr=0.5` funkcji `prf` w `scripts/eval_base.py`. JSON nie zapisuje tego progu; jego wartość ustalono z obecnego kodu.

AP i mAR oblicza TorchMetrics według procedury COCO. AP i mAP@[.5:.95] wykorzystują progi IoU od 0,50 do 0,95 co 0,05; AP50 wykorzystuje tylko IoU 0,50. mAR@100 uśrednia recall po klasach i tych progach IoU, przy limicie 100 najwyżej ocenionych detekcji na obraz i klasę. Szczegóły procedury: [implementacja COCO](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py).

W skrypcie detekcje o pewności poniżej 0,05 są odrzucane już podczas postprocessingu modelu. AP jest więc obliczane z rankingu pozostałych detekcji, bez dodatkowego odcięcia na 0,5. Wynik należy opisywać wraz z tym wstępnym progiem 0,05.

**Warianty i wyniki zbiorcze**

- A: mapowanie klas COCO na klasy zbioru; czerwone i zielone światła połączone w `Traffic-Light`.
- B: wariant A oraz filtr domenowy: odrzucenie zbyt dużych boksów i boksów w dolnej części obrazu oraz NMS między klasami z progiem IoU 0,7. Próg NMS pełni inną funkcję niż próg IoU 0,5 służący ocenie poprawności detekcji.
- C: wariant B oraz rozdzielenie czerwonych i zielonych świateł przez analizę koloru HSV wycinka obrazu.

Wartości metryk w tabelach podano w procentach, zaokrąglone do dwóch miejsc. mAP 11,33% odpowiada wartości 0,1133 i nie oznacza procentu poprawnie ocenionych obrazów.

| Wariant | mAP@[.5:.95] (%) | AP50 (%) | mAR@100 (%) |
|---|---:|---:|---:|
| A | 11,33 | 26,28 | 17,97 |
| B | 10,45 | 24,25 | 16,42 |
| C | 11,21 | 25,98 | 17,51 |

**Wyniki dla każdej klasy — wariant A**

AP oznacza AP@[.5:.95]. Precision, recall i F1 wykorzystują pewność ≥ 0,5 i IoU ≥ 0,5.

| Klasa | Liczba GT | AP (%) | Precision (%) | Recall (%) | F1 (%) | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Car | 3593 | 45,05 | 27,15 | 86,97 | 41,38 | 3125 | 8386 | 468 |
| Different-Traffic-Sign | 2479 | 0,00 | 0,00 | 0,00 | 0,00 | 0 | 0 | 2479 |
| Motorcycle | 28 | 8,33 | 48,39 | 53,57 | 50,85 | 15 | 16 | 13 |
| Pedestrian | 578 | 9,50 | 23,55 | 54,15 | 32,83 | 313 | 1016 | 265 |
| Pedestrian-Crossing | 271 | 0,00 | 0,00 | 0,00 | 0,00 | 0 | 0 | 271 |
| Prohibition-Sign | 280 | 2,54 | 26,44 | 8,21 | 12,53 | 23 | 64 | 257 |
| Speed-Limit-Sign | 149 | 0,00 | 0,00 | 0,00 | 0,00 | 0 | 0 | 149 |
| Truck | 359 | 22,57 | 16,19 | 59,33 | 25,43 | 213 | 1103 | 146 |
| Warning-Sign | 419 | 0,00 | 0,00 | 0,00 | 0,00 | 0 | 0 | 419 |
| Traffic-Light | 825 | 25,32 | 19,01 | 76,61 | 30,47 | 632 | 2692 | 193 |

**Wyniki dla każdej klasy — wariant B**

AP oznacza AP@[.5:.95]. Precision, recall i F1 wykorzystują pewność ≥ 0,5 i IoU ≥ 0,5.

| Klasa | Liczba GT | AP (%) | Precision (%) | Recall (%) | F1 (%) | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Car | 3593 | 38,44 | 25,99 | 76,82 | 38,83 | 2760 | 7861 | 833 |
| Different-Traffic-Sign | 2479 | 0,00 | 0,00 | 0,00 | 0,00 | 0 | 0 | 2479 |
| Motorcycle | 28 | 6,00 | 42,86 | 42,86 | 42,86 | 12 | 16 | 16 |
| Pedestrian | 578 | 9,14 | 24,40 | 52,60 | 33,33 | 304 | 942 | 274 |
| Pedestrian-Crossing | 271 | 0,00 | 0,00 | 0,00 | 0,00 | 0 | 0 | 271 |
| Prohibition-Sign | 280 | 2,53 | 26,74 | 8,21 | 12,57 | 23 | 63 | 257 |
| Speed-Limit-Sign | 149 | 0,00 | 0,00 | 0,00 | 0,00 | 0 | 0 | 149 |
| Truck | 359 | 23,09 | 27,04 | 55,43 | 36,35 | 199 | 537 | 160 |
| Warning-Sign | 419 | 0,00 | 0,00 | 0,00 | 0,00 | 0 | 0 | 419 |
| Traffic-Light | 825 | 25,33 | 19,42 | 76,61 | 30,99 | 632 | 2622 | 193 |

**Wyniki dla każdej klasy — wariant C**

AP oznacza AP@[.5:.95]. Precision, recall i F1 wykorzystują pewność ≥ 0,5 i IoU ≥ 0,5.

| Klasa | Liczba GT | AP (%) | Precision (%) | Recall (%) | F1 (%) | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Car | 3593 | 38,44 | 25,99 | 76,82 | 38,83 | 2760 | 7861 | 833 |
| Different-Traffic-Sign | 2479 | 0,00 | 0,00 | 0,00 | 0,00 | 0 | 0 | 2479 |
| Green-Traffic-Light | 219 | 17,04 | 36,81 | 48,40 | 41,81 | 106 | 182 | 113 |
| Motorcycle | 28 | 6,00 | 42,86 | 42,86 | 42,86 | 12 | 16 | 16 |
| Pedestrian | 578 | 9,14 | 24,40 | 52,60 | 33,33 | 304 | 942 | 274 |
| Pedestrian-Crossing | 271 | 0,00 | 0,00 | 0,00 | 0,00 | 0 | 0 | 271 |
| Prohibition-Sign | 280 | 2,53 | 26,74 | 8,21 | 12,57 | 23 | 63 | 257 |
| Red-Traffic-Light | 606 | 27,03 | 16,49 | 80,69 | 27,38 | 489 | 2477 | 117 |
| Speed-Limit-Sign | 149 | 0,00 | 0,00 | 0,00 | 0,00 | 0 | 0 | 149 |
| Truck | 359 | 23,09 | 27,04 | 55,43 | 36,35 | 199 | 537 | 160 |
| Warning-Sign | 419 | 0,00 | 0,00 | 0,00 | 0,00 | 0 | 0 | 419 |

**Dlaczego AP wykorzystuje ranking, a precision, recall i F1 ustalony próg?**

Model przypisuje każdej detekcji ocenę pewności. Dla AP detekcje danej klasy są uporządkowane od najwyższej do najniższej oceny. Kolejne detekcje wyznaczają punkty krzywej precision–recall. AP podsumowuje tę krzywą przez uśrednienie interpolowanej precyzji na poziomach recall. Poprawne detekcje na początku rankingu sprzyjają wysokiemu AP, a wczesne fałszywe alarmy je obniżają. AP ocenia więc jakość uporządkowania detekcji w zakresie zachowanych wyników. [Implementacja TorchMetrics](https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/detection/mean_ap.py).

Precision, recall i F1 opisują działanie dla konkretnego ustawienia systemu. Tutaj najpierw pozostają detekcje z pewnością co najmniej 0,5, a następnie ocenia się je przy IoU co najmniej 0,5. Zmiana progu pewności zmienia zestaw przyjętych detekcji, więc zmienia też te trzy metryki. Zwykle wyższy próg zmniejsza liczbę fałszywych alarmów, ale może zwiększać liczbę pominiętych obiektów. Nie gwarantuje to monotonicznej zmiany precision.

IoU to pole części wspólnej przewidzianego boksu i boksu referencyjnego podzielone przez pole ich sumy. W funkcji `prf` dopasowanie odbywa się osobno dla każdego obrazu i klasy, w kolejności malejącej pewności. Detekcja otrzymuje najlepszy dostępny, jeszcze niedopasowany obiekt GT, jeśli IoU ≥ 0,5. Każdy obiekt referencyjny może być dopasowany najwyżej raz. Kolejna detekcja tego samego obiektu jest FP, jeżeli nie pasuje do innego wolnego GT.

- TP: detekcja poprawnej klasy dopasowana do obiektu referencyjnego.
- FP: detekcja bez poprawnego dopasowania.
- FN: obiekt referencyjny bez dopasowanej detekcji.

`precision = TP / (TP + FP)`

`recall = TP / (TP + FN)`

`F1 = 2 × precision × recall / (precision + recall)`

Przykład dla samochodów w wariancie A: TP = 3125, FP = 8386, FN = 468. Model wykrył 3125 z 3593 samochodów, czyli recall 86,97%. Jednocześnie tylko 3125 z 11511 detekcji samochodów było poprawnych, czyli precision 27,15%. F1 wyniosło 41,38%. Oznacza to wysoką wykrywalność przy dużej liczbie fałszywych alarmów.

**Interpretacja otrzymanych wyników**

Wariant A ma najwyższe mAP: 11,33%, przy AP50 26,28%. Duży spadek między AP50 a mAP uśrednionym po bardziej wymagających progach IoU wskazuje, że część detekcji nie ma wystarczająco dokładnych boksów dla surowszej oceny. Jest to wniosek z porównania metryk, a nie pomiar przyczyn błędów.

Filtr domenowy w wariancie B obniżył mAP z 11,33% do 10,45%. Dla samochodów recall spadło z 86,97% do 76,82%. Dla ciężarówek filtr poprawił precision z 16,19% do 27,04% i F1 z 25,43% do 36,35%, przy spadku recall z 59,33% do 55,43%. Filtr pomaga części klas, ale usuwa też poprawne detekcje.

Wariant C pozwala osobno ocenić czerwone i zielone światła. Dla czerwonych świateł recall wynosi 80,69%, ale precision tylko 16,49%; dla zielonych odpowiednio 48,40% i 36,81%. Analiza HSV jest heurystyką rozdzielającą kolor detekcji modelu, a nie dotrenowanym klasyfikatorem sygnalizacji.

W A/B oceniane są 10 klas, a w C 11, ponieważ jedna klasa świateł zostaje rozdzielona na dwie. Zmienia się sposób uśredniania; wzrost mAP między B i C nie dowodzi poprawy wszystkich detekcji.

Klasy `Different-Traffic-Sign`, `Pedestrian-Crossing`, `Speed-Limit-Sign` i `Warning-Sign` nie mają mapowania z klas COCO w skrypcie. Żadna detekcja nie może więc dostać tych etykiet, co daje AP, precision, recall i F1 równe zero przy istniejących obiektach GT. Ich uwzględnienie obniża średnie. Mapowanie `Prohibition-Sign` na COCO `stop sign` jest przybliżone: model rozpoznający STOP nie obejmuje tym samym wszystkich znaków zakazu.

Wyniki stanowią punkt odniesienia dla modelu bez dotrenowania. Niskie metryki łączne wynikają między innymi z niedopasowania zestawu klas i licznych fałszywych detekcji. Dla motocykli jest tylko 28 obiektów GT, więc wynik tej klasy należy interpretować z uwzględnieniem małej liczby przykładów.

Zweryfikowano dla każdej klasy i wariantu: TP + FN = liczba GT, a precision, recall i F1 są zgodne z licznikami zapisanymi w JSON.
