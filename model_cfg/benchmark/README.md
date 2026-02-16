# Tiberius Benchmarks

> **Note:** Some benchmarks below were produced with parameter sets that have since been superseded. These results are retained for reference and reproducibility. The superseded YAML files can be found in [`model_cfg/superseded/`](../superseded/). For new analyses, always use the current parameter files in [`model_cfg/`](../).

---

## Mammals

| Species                 | Tool     | Exon Sn | Exon Pr |  Exon F1 | Gene Sn | Gene Pr | Gene F1 |
| :---------------------- | :------- | :-----: | :-----: | :------: | :------: | :------: | :------: |
| *Bos taurus*            | Tiberius |   88.2  |   87.2  |   87.7   |   51.2   |   50.2   |   50.7   |
|                         | ANNEVO   |   85.2  |   74.9  |   79.7   |   33.7   |   29.7   |   31.6   |
|                         | Helixer  |   81.0  |   66.3  |   72.8   |   18.1   |   18.9   |   18.5   |
|                         | AUGUSTUS |   67.6  |   65.8  |   66.7   |   11.6   |   11.4   |   11.5   |
|                         | BRAKER3  |   73.5  |   90.2  |   81.0   |   47.1   |   53.6   |   50.1   |
|                         | EGAPx    |   89.7  |   85.9  | 87.8 |   51.4   |   48.1   | 49.7 |
| *Delphinapterus leucas* |  Tiberius |   89.1  |   91.2  |   90.1   |   51.9   |   52.5   |   52.2    |
|                         | ANNEVO   |   85.2  |   78.2  |   81.6   |   29.0   |   26.1   |   27.5 |
|                         | Helixer  |   82.7  |   70.9  |   76.3   |   18.8   |   18.5   |   18.6   |
|                         | AUGUSTUS |   68.3  |   65.5  |   66.9   |   12.1   |   10.2   |   11.1   |
|                         | BRAKER3  |   74.7  |   92.2  |   82.5   |   51.3   |   52.7   |   52.0   |
|                         | EGAPx    |   92.2  |   90.4  | 91.3 |   59.8   |   54.2   | 56.9 |
| *Homo sapiens*          | Tiberius |   92.9  |   89.9  | 91.4 |   64.1   |   56.5   |   60.1   |
|                         | ANNEVO   |   88.3  |   76.1  |   81.7   |   36.1   |   28.6   |   32.0   |
|                         | Helixer  |   85.7  |   60.4  |   70.8   |   22.8   |   17.9   |   20.1   |
|                         | AUGUSTUS |   70.2  |   67.2  |   68.7   |   14.6   |   11.6   |   13.0   |
|                         | BRAKER3  |   82.9  |   90.9  |   86.7   |   62.3   |   51.8   |   56.6   |
|                         | EGAPx    |   91.6  |   85.5  |   88.4   |   57.8   |   44.5   | 50.3 |





## Diatoms

| Species                     | Tool     | Exon Sn | Exon Pr |  Exon F1 | Gene Sn | Gene Pr | Gene F1 |
| :-------------------------- | :------- | :-----: | :-----: | :------: | :------: | :------: | :------: |
| *Phaeodactylum tricornutum* | Tiberius |   61.0  |   67.3  |   64.0   |   59.9   |   60.8   | 60.3 |
|                             | ANNEVO   |   18.2  |   58.0  |   27.7   |   12.2   |   45.5   |   19.3   |
|                             | Helixer  |   42.8  |   48.0  |   45.2   |   39.5   |   47.2   |   43.0   |
|                             | AUGUSTUS |   47.0  |   59.7  |   52.6   |   42.4   |   50.5   |   46.1   |
|                             | BRAKER3  |   64.0  |   72.4  | 68.0 |   62.9   |   66.2   |   64.5   |
| *Thalassiosira pseudonana*  | Tiberius |   77.0  |   76.1  | 76.5 |   68.7   |   66.6   | 67.6 |
|                             | ANNEVO   |   12.7  |   61.4  |   21.0   |    8.4   |   41.6   |   14.0   |
|                             | Helixer  |   47.7  |   47.2  |   47.4   |   33.2   |   38.9   |   35.8   |
|                             | AUGUSTUS |   74.6  |   74.1  |   74.3   |   60.4   |   63.9   |   62.1   |
|                             | BRAKER3  |   65.7  |   83.1  |   73.8   |   65.2   |   73.7   |   69.2   |


## Lepidoptera *(superseded — see `insecta`)*

| Species            | Tool     | Exon Sn | Exon Pr |  Exon F1 | Gene Sn | Gene Pr | Gene F1 |
| :----------------- | :------- | :-----: | :-----: | :------: | :------: | :------: | :------: |
| *Bombyx mori*      | Tiberius |   74.4  |   81.3  |   77.8   |   39.8   |   32.5   |   35.8   |
|                    | ANNEVO   |   84.3  |   77.9  |   81.0   |   36.9   |   35.9   |   36.4   |
|                    | Helixer  |   84.2  |   71.0  |   77.0   |   31.2   |   27.5   |   29.3   |
|                    | AUGUSTUS |   59.0  |   63.2  |   61.0   |   21.4   |   13.0   |   16.1   |
|                    | BRAKER3  |   39.4  |   84.1  |   53.7   |   37.3   |   45.1   |   40.8   |
| *Colias croceus*   | Tiberius |   87.7  |   88.5  | 88.1 |   59.6   |   52.3   |   55.7   |
|                    | ANNEVO   |   86.3  |   81.6  |   83.9   |   43.7   |   43.7   |   43.7   |
|                    | Helixer  |   87.7  |   73.9  |   80.0   |   39.7   |   33.0   |   36.0   |
|                    | AUGUSTUS |   71.8  |   69.0  |   70.3   |   26.7   |   16.6   |   20.4   |
|                    | BRAKER3  |   73.6  |   90.7  |   81.3   |   63.0   |   58.8   | 60.8 |
| *Danaus plexippus* | Tiberius |   83.1  |   87.8  |   85.4   |   54.5   |   48.4   |   51.3   |
|                    | ANNEVO   |   86.9  |   82.5  |   84.7   |   47.4   |   46.7   |   47.0   |
|                    | Helixer  |   88.0  |   78.1  |   82.7   |   45.3   |   38.9   |   41.9   |
|                    | AUGUSTUS |   66.1  |   73.7  |   69.6   |   23.0   |   18.0   |   20.2   |
|                    | BRAKER3  |   82.7  |   91.8  | 87.0 |   69.2   |   59.6   | 64.0 |




## Eudicots 
| Species                    | Tool            | Exon Sn | Exon Pr |  Exon F1 | Gene Sn | Gene Pr | Gene F1 |
| :------------------------- | :-------------- | :-----: | :-----: | :------: | :------: | :------: | :------: |
| *Arabidopsis thaliana*     | Tiberius        |   87.3  |   92.9  |   90.0   |   67.4   |   76.4   |   71.6   |
|                            | ANNEVO          |   84.9  |   86.9  |   85.9   |   58.1   |   63.7   |   60.8   |
|                            | Helixer         |   88.8  |   87.5  |   88.1   |   67.2   |   68.3   |   67.7   |
|                            | AUGUSTUS        |   85.8  |   73.4  |   79.1   |   55.5   |   47.0   |   50.9   |
|                            | BRAKER3         |   81.2  |   94.9  |   87.5   |   65.6   |   78.0   | 71.2 |
| *Eschscholzia californica* | Tiberius        |   86.7  |   92.2  |   89.4   |   63.8   |   72.9   |   68.0   |
|                            | ANNEVO          |   84.1  |   83.5  |   83.8   |   51.4   |   57.4   |   54.2   |
|                            | Helixer         |   82.9  |   73.7  |   78.0   |   53.6   |   49.0   |   51.2   |
|                            | AUGUSTUS        |   74.8  |   63.7  |   68.8   |   28.5   |   24.6   |   26.4   |
|                            | BRAKER3         |   84.9  |   92.6  |   88.6   |   66.8   |   72.9   |   69.7   |
| *Mimulus guttatus*         | Tiberius        |   91.9  |   93.0  |   92.5   |   77.9   |   77.6   |   77.7   |
|                            | ANNEVO          |   88.1  |   84.1  |   86.0   |   61.9   |   60.0   |   60.9   |
|                            | Helixer         |   92.4  |   83.7  |   87.8   |   71.6   |   64.0   |   67.6   |
|                            | AUGUSTUS        |   80.6  |   76.4  |   78.5   |   45.3   |   39.2   |   42.0   |
|                            | BRAKER3         |   77.6  |   95.7  |   85.6   |   66.6   |   82.1   | 73.6 |


## Monocts
| Species                  | Tool            | Exon Sn | Exon Pr |  Exon F1 | Gene Sn | Gene Pr | Gene F1 |
| :----------------------- | :-------------- | :-----: | :-----: | :------: | :------: | :------: | :------: |
| *Brachypodium stacei*    | Tiberius        |   85.7  |   91.2  |   88.3   |   64.3   |   69.2   |   66.7   |
|                          | ANNEVO          |   83.9  |   85.4  |   84.6   |   53.8   |   61.6   |   57.4   |
|                          | Helixer         |   88.7  |   78.9  |   83.5   |   61.7   |   56.9   |   59.2   |
| *Freycinetia multiflora* | Tiberius        |   82.1  |   85.3  |   83.7   |   57.3   |   54.0   |   55.6   |
|                          | ANNEVO          |   83.1  |   77.5  |   80.2   |   48.9   |   48.8   |   48.9   |
|                          | Helixer         |   86.5  |   67.6  |   75.9   |   47.5   |   39.5   |   43.2   |
| *Sorghum bicolor*        | Tiberius        |   86.5  |   92.7  |   89.5   |   67.6   |   74.9   |   71.1   |
|                          | ANNEVO          |   81.4  |   84.6  |   83.0   |   50.9   |   60.7   |   55.3   |
|                          | Helixer         |   85.9  |   80.6  |   83.2   |   58.3   |   59.3   |   58.8   |
| *Urochloa brizantha*     | Tiberius        |   84.5  |   85.0  |   84.7   |   65.1   |   60.8   |   62.9   |
|                          | ANNEVO          |   82.3  |   79.3  |   80.8   |   54.3   |   56.0   |   55.1   |
|                          | Helixer         |   87.6  |   62.2  |   72.7   |   61.9   |   43.1   |   51.0   |


## Mucoromycota *(superseded)*

| Species             | Tool     | Exon Sn | Exon Pr |  Exon F1 | Gene Sn | Gene Pr | Gene F1 |
| :------------------ | :------- | :-----: | :-----: | :------: | :------: | :------: | :------: |
| *Fennellomyces sp.* | Tiberius |   70.2  |   71.2  |   70.7   |   36.5   |   38.1   | 37.3 |
|                     | ANNEVO   |   72.5  |   67.0  |   69.6   |   35.2   |   35.8   |   35.5   |
|                     | Helixer  |   70.5  |   55.2  |   61.8   |   32.9   |   26.4   |   29.4   |
|                     | AUGUSTUS |   67.9  |   64.7  |   66.3   |   30.3   |   30.9   |   30.6   |
| *Mucor fuscus*      | Tiberius |   72.8  |   79.3  | 75.9 |   45.8   |   47.4   | 46.6 |
|                     | ANNEVO   |   75.3  |   77.0  |   76.1   |   43.8   |   46.5   |   45.1   |
|                     | Helixer  |   74.0  |   67.0  |   70.3   |   41.7   |   35.3   |   38.2   |
|                     | AUGUSTUS |   81.2  |   84.4  |   82.8   |   53.3   |   56.7   |   55.0   |


## Saccharomycota *(superseded)*

| Species                   | Tool     | Exon Sn | Exon Pr |  Exon F1 | Gene Sn | Gene Pr | Gene F1 |
| :------------------------ | :------- | :-----: | :-----: | :------: | :------: | :------: | :------: |
| *Candida tanzawaensis*    | Tiberius |   50.1  |   66.2  |   57.0   |   55.5   |   66.7   | 60.6 |
|                           | ANNEVO   |   46.5  |   60.0  |   52.3   |   50.7   |   61.0   |   55.4   |
|                           | Helixer  |   51.0  |   60.7  |   55.4   |   56.3   |   65.6   |   60.6   |
| *Sympodiomyces attinorum* | Tiberius |   66.5  |   75.5  | 70.7 |   70.3   |   73.5   | 71.9 |
|                           | ANNEVO   |   69.3  |   72.7  |   70.9   |   67.7   |   70.0   |   68.8   |
|                           | Helixer  |   37.9  |   86.5  |   52.8   |   83.3   |   86.1   | 84.7 |
|                           | AUGUSTUS |   46.0  |   63.9  |   53.5   |   55.2   |   62.7   |   58.7   |


## Sordariomycota *(superseded)*

| Species                     | Tool     | Exon Sn | Exon Pr |  Exon F1 | Gene Sn | Gene Pr | Gene F1 |
| :-------------------------- | :------- | :-----: | :-----: | :------: | :------: | :------: | :------: |
| *Purpureocillium lilacinum* | Tiberius |   71.1  |   70.3  | 70.7 |   57.6   |   57.5   | 57.6 |
|                             | ANNEVO   |   72.7  |   68.9  |   70.7   |   55.2   |   55.9   |   55.5   |
|                             | Helixer  |   72.1  |   55.8  |   62.8   |   54.0   |   43.3   |   48.1   |
|                             | AUGUSTUS |   68.2  |   62.5  |   65.2   |   49.9   |   50.7   |   50.3   |
| *Scedosporium apiospermum*  | Tiberius |   71.8  |   66.2  |   68.9   |   54.9   |   38.8   |   45.4   |
|                             | ANNEVO   |   74.4  |   60.7  |   66.8   |   51.9   |   35.3   |   42.0   |
|                             | Helixer  |   75.9  |   58.7  |   66.2   |   55.0   |   34.5   |   42.1   |
|                             | AUGUSTUS |   76.1  |   74.4  | 75.3 |   59.1   |   47.2   | 52.5 |
