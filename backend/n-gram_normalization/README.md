# Filipino N-Gram Spelling Normalizer

This project normalizes informal or noisy Filipino spellings with automatically generated character N-gram replacement rules. It adapts Lorenzo Jaime Flores and Dragomir Radev's research repository for *Look Ma, Only 400 Samples! Revisiting the Effectiveness of Automatic N-Gram Rule Generation for Spelling Normalization in Filipino* (SustaiNLP at EMNLP 2022).

The workflow is limited to text normalization. It does not contain FILLY's separate grammar-correction component.

## Features

- Learns character replacement rules from paired `Input,Target` examples.
- Uses the original rule-frequency calculation, candidate generation, vocabulary filtering, and Damerau-Levenshtein ranking.
- Saves learned rules to a readable JSON model.
- Uses learned exact-pair rules for seen forms and N-gram rules for unseen variants.
- Normalizes words or whitespace-delimited sentences while preserving surrounding punctuation and capitalization.

## Requirements and installation

Python 3.12 is the verified version. From this repository's directory in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, call the environment's interpreter directly, as below.

## Dataset

Place training data at `data/train_words.csv`. It must be UTF-8 CSV with exactly these case-sensitive headers:

```csv
Input,Target
informal_form,normalized_form
```

Both fields must be nonblank. Targets may contain spaces. Quote fields containing commas according to normal CSV rules. Duplicate pairs provide additional frequency evidence. If one input has multiple targets, the most frequent target is preferred; ties follow first occurrence in the CSV.

The locally supplied dataset has 571 valid rows. It already has the expected headers and is used directly—no converted or overwritten copy is needed.

The clean public-ready copy intentionally excludes the supplied `train_words.csv`: no explicit redistribution terms or source documentation were found for the added data. Copy an authorized dataset into that path before training. A header-only `data/train_words.example.csv` is included as a format template.

## Training

```powershell
.\.venv\Scripts\python.exe train_normalizer.py
```

This reads `data/train_words.csv`, builds exact-pair and character N-gram rules, adds training targets and the bundled Tagalog vocabulary to candidate filtering, and writes `models/ngram_rules.json`.

Custom paths and N-gram size are supported:

```powershell
.\.venv\Scripts\python.exe train_normalizer.py --data data\train_words.csv --output models\ngram_rules.json --max-sub 2
```

## Usage

```powershell
.\.venv\Scripts\python.exe normalize.py prang
.\.venv\Scripts\python.exe normalize.py "tlga masaya ako"
```

Verified examples from the locally trained model:

```text
prang           -> parang
d2              -> dito
lodi            -> idol
maganda         -> maganda
tlga masaya     -> talaga masaya
kumusta po kayo -> kumusta po kayo
prng            -> parang
cge na          -> sige na
```

`prng` does not occur in the training CSV; it demonstrates N-gram candidate generation rather than exact lookup.

## Tests and legacy benchmark

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe main_algo.py --mode test --use_dld True --max_sub 2 --cutoff 100
```

On the current local training data and upstream headerless test set, the legacy command completed with accuracy at 1 of `0.56`. This is a compatibility check, not a claim about general accuracy or FILLY as a whole.

## Project structure

```text
data/train_words.csv                       local paired training data
models/ngram_rules.json                    generated model (not committed)
TagalogStemmerPython/output/with_info.txt  bundled vocabulary resource
ngram_normalizer.py                        reusable training/inference code
train_normalizer.py                        training CLI
normalize.py                               inference CLI
main_algo.py                               upstream evaluation CLI
utils.py                                   upstream N-gram/ranking functions
tests/test_normalizer.py                   workflow smoke test
```

`main_model.py` contains upstream GPU/deep-learning experiments. It is not required by this N-gram workflow.

## Known limitations

- Normalization is token-by-token and does not infer meaning from context.
- Ambiguous inputs are resolved by training frequency, then CSV order on ties.
- Vocabulary filtering can reject a valid form absent from the bundled vocabulary and training targets.
- Candidate generation can slow down with longer tokens or larger N-gram/cutoff values.
- Only surrounding punctuation is preserved; punctuation inside a token is not linguistically parsed.
- Dataset quality and coverage directly control learned behavior.

## Attribution and citation

Original repository: <https://github.com/ljyflores/efficient-spelling-normalization-filipino>

```bibtex
@inproceedings{flores-radev-2022-look,
  title = "Look Ma, Only 400 Samples! Revisiting the Effectiveness of Automatic N-Gram Rule Generation for Spelling Normalization in {F}ilipino",
  author = "Flores, Lorenzo Jaime and Radev, Dragomir",
  booktitle = "Proceedings of The Third Workshop on Simple and Efficient Natural Language Processing (SustaiNLP)",
  month = dec,
  year = "2022",
  publisher = "Association for Computational Linguistics",
  url = "https://aclanthology.org/2022.sustainlp-1.5",
  pages = "29--35"
}
```

The bundled `TagalogStemmerPython` component is by Carl Jerwin F. Gensaya and retains its MIT license in `TagalogStemmerPython/LICENSE`.

## License status

The cloned upstream repository has no top-level license file, and no explicit license for its main source or datasets was found in its README. Public visibility is not itself a software license. This adaptation does not invent or replace an upstream license. Before publishing the adapted source, confirm reuse/redistribution permission with the upstream author or obtain an explicit license. Keep all attribution and the third-party stemmer license intact.
