# Textual Noise and Denoising Workflow

The dataset used is the [DARE Dataset](https://huggingface.co/datasets/cambridgeltl/DARE) (Sterz et al.).

---

## Step 1: Download the Dataset

To download the dataset (without the heavy image column for faster processing), run:

```bash
python download_dataset.py
```

This generates:

```
data/[without images]1_correct_validation.csv
```

---

## Step 2: Add Noise to Questions

To introduce noise into the `question` column, run:

```bash
python noise_addition.py
```

This produces:

```
data/NoisyQuestionPairs.csv
```

Two new columns are added:

| Column | Description |
|--------|-------------|
| `modified_question` | The corrupted version of the original question |
| `modified_question_function_name` | The name of the perturbation function applied |

---

## Step 3: Add Denoised Questions

To create denoised versions of the noisy questions in `modified_question`, run:

```bash
python denoise_script.py
```

This produces the **final CSV**:

```
data/Noisy-Denoised_QuestionPairs.csv
```

One new column is added:

| Column | Description |
|--------|-------------|
| `denoised_question` | Gemini 2.0 Flash zero-shot denoised version of `modified_question` |

### Example Output Row

```
id,instance_id,question,answer,A,B,C,D,category,path,modified_question,modified_question_function_name,denoised_question
vcr_2321,2321,what are they doing,C,they are discussing divorce,...,vcr,000000130826.jpg,what are they doiÉ´g,substitute_with_homoglyphs,What are they doing?
```

---

## Perturbation Types

The `noise_addition.py` script applies 18 perturbation methods:

### Character-level
- OCR errors, keyboard errors, character insertions, deletions, swaps

### Word-level
- Synonym replacement, misspelling, abbreviation, word swapping

### Stylistic
- Leet speak, homoglyphs, case randomization, character repetition

### Semantic
- Back-translation, paraphrasing
