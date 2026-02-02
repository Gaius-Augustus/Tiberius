## Tiberius Model Configuration YAML Files

This directory contains **one YAML file per trained Tiberius weight**.
Each file is read at runtime by Tiberius and therefore must follow the schema below. 

---

## 1  List of model weigths

| Model name | Target species | Softmasking required | ClaMSA input required| 
| ----------------|------------------|-------------------|------------------|
| basidiomycota | Basidiomycota | :x: |  :x:| 
| chlorophyta | Chlorophyta | :x: |  :x:| 
| diatoms | Diatoms | :white_check_mark: |  :x:| 
| eudicotyledons | Eudicotyledons | :white_check_mark: |  :x:| 
| fungi_incertae_sedis | Fungi incertae sedis |  :x:|  :x:| 
| insecta | Insecta | :white_check_mark: |  :x:| 
| monocotyledonae | Monocotyledonae | :white_check_mark: |  :x:| 
| mucoromycota | Mucoromycota | :white_check_mark: |  :x:| 
| mammalia_softmasking_v2 | Mammalia | :white_check_mark: |  :x:| 
| mammalia_softmasking_v2 | Mammalia | :white_check_mark: |  :x:| 
| mammalia_nosofttmasking_v2 | Mammalia | :x: |  :x:| 
| mammalia_clamsa_v2 | Mammalia | :white_check_mark: |  :white_check_mark:| 
| saccharomycota | Saccharomycota | :white_check_mark: |  :x:| 
| sordariomycota | Sordariomycota | :white_check_mark: |  :x:| 
| vertebrates | Vertebrates | :x: |  :x:| 


## 2  File naming convention

```
<target_species|clade>_<identifying_attribute>_<version>.yaml
```

Example:  `mammalia_nosoftmasking_v2.yaml`
* `target_species` or `clade` → scientific name of the focal clade or species the weights generalises to.
* `identifying_attribute` → a short description of the model (e.g. **"nosoftmasking"**, **"clamsa"**).
* `version` → version of the model (e.g. **"v2"**, **"v2.3-alpha"**).

---

## 3  Required keys

| Key                                | Type               | Constraints & semantics                                                                                                                                                                                       |
| ---------------------------------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `target_species`                   | *string*           | Scientific name of the focal clade or species the weights generalise to. |
| `ncbi_tax_id`                      | *integer*          | NCBI Taxonomy identifier corresponding to `target_species`.                       |
| `weights_url`                      | *string*           | Fully‑qualified URL (HTTP\[S], S3, or `file://`) that points to the model weights archive. Make sure the link is public and can be accessed with `wget` or `curl`. |
| `softmasking`                      | *boolean*          | `true` → model uses softmasing track as input,                                                                                                                    `false` → model does not use softmasking. |
| `clamsa`                           | *boolean*          | `true` if the model was pre‑trained with the CLAMSA track, `false` otherwise.                                                                                                                       |
| `tiberius_version`                 | *semver string*    | Tiberius version at training time that can load the weights (e.g. `1.1.5`). |
| `date`                             | *string*           | Date the weights were finalised **in ISO‑8601 format `YYYY-MM-DD`**.                                                                                                                               |
| `comment`                          | *multiline string* | Free text for provenance & citation. Begin the first line with author.                                                  |
| `training_species`                 | *YAML sequence*    | Ordered list of Latin binomials (one per line) included in the training set.                                                                                         |


---

## 4  Example (minimal)

```yaml
target_species: "Mammalia"
ncbi_tax_id: 40674
weights_url: "https://bioinf.uni-greifswald.de/bioinf/tiberius/models/tiberius_weights_v2.tar.gz"
softmasking: true
clamsa: false
tiberius_version: 1.1.5
date: "2025-05-12"
comment: |
  # Lars Gabriel, 2025‑05‑12
  Training weights from the original Tiberius paper:
  https://doi.org/10.1093/bioinformatics/btae685
  Saved in a different format to the original weights.
training_species:
  - Aotus_nancymaae
  - Camelus_bactrianus
  - …
```

---
