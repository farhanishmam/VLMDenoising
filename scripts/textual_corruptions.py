import os
import pandas as pd

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc
from nlpaug.util import Action
import numpy as np
import torch

import random
import nltk

import nltk
from nltk.corpus import words
import random

# Ensure NLTK words corpus is downloaded
nltk.download('words')
nltk.download('averaged_perceptron_tagger_eng')
word_list = set(words.words())  # Set of English words

# Set a global seed value for reproducibility
SEED_VALUE = 42

def set_seed(seed=SEED_VALUE):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Ensure seed is set at the beginning
set_seed()


# Load DataFrame - Update path as needed
# original_df = pd.read_csv('path/to/your_validation.csv')

# Define the keyboard layout with adjacent letters
adjacent_letters = {
    'q': 'qwa', 'w': 'qeasd', 'e': 'wrsdf', 'r': 'edfgt', 't': 'rfgy', 'y': 'tghu', 'u': 'yhij', 'i': 'ujko', 'o': 'iklp', 'p': 'ol',
    'a': 'qwsz', 's': 'qawdezxc', 'd': 'wersxfc', 'f': 'ertdgcv', 'g': 'rtyfvbh', 'h': 'gybnj', 'j': 'uhn', 'k': 'ij', 'l': 'ok',
    'z': 'za', 'x': 'zsd', 'c': 'xdfv', 'v': 'cfgb', 'b': 'vbn', 'n': 'bh', 'm': 'nm'
}

######### Substitute Character by Ocr #######
def substituteCharacterByOCR(text):
    set_seed()
    aug = nac.OcrAug()
    augmented_texts = aug.augment(text, n=1)
    print("Original:")
    print(text)
    print("Augmented Texts:")
    print(augmented_texts[0])
    return augmented_texts[0]


################ Substitute Character by Keyboard ##############

def substituteCharacterByKeyboard(text):
    set_seed()
    
    aug = nac.KeyboardAug()
    augmented_text = aug.augment(text)
    print("Original:")
    print(text)
    print("Augmented Text:")
    print(augmented_text[0])
    return augmented_text[0]

    
##########  Insert Character Randomly ###########

def insertCharacterRandomly(text):
    set_seed()
    
    aug = nac.RandomCharAug(action="insert")
    augmented_text = aug.augment(text)
    print("Original:")
    print(text)
    print("Augmented Text:")
    print(augmented_text[0])
    return augmented_text[0]

############ Substitute Char Randomly #########

def substituteCharacterRandomly(text):
    set_seed()
    
    aug = nac.RandomCharAug(action="substitute")
    augmented_text = aug.augment(text)
    print("Original:")
    print(text)
    print("Augmented Text:")
    print(augmented_text[0])
    return augmented_text[0]
    

####### Swap Character Randomly ##########
def swapCharacterRandomly(text):
    set_seed()
    
    aug = nac.RandomCharAug(action="swap")
    augmented_text = aug.augment(text)
    print("Original:")
    print(text)
    print("Augmented Text:")
    print(augmented_text[0])
    return augmented_text[0]
    

#################  Delete Character Randomly ##############
def delCharacterRandomly(text):
    set_seed()
    
    aug = nac.RandomCharAug(action="delete")
    augmented_text = aug.augment(text)
    print("Original:")
    print(text)
    print("Augmented Text:")
    print(augmented_text[0])
    return augmented_text[0]

#### Uppercase Char Randomly #######
def uppercaseRandomly(text):
    set_seed()
    
    severity_ratio = 2 / 5
    modified_text = ''.join(
        char.upper() if random.random() < severity_ratio else char.lower()
        for char in text
    )
    print(modified_text)
    return modified_text    

#### Repeat Character #####
def repeatCharacterRandomly(text):
    set_seed()
    
    # Set a repetition ratio (e.g., 0.2 for 20% of characters in each word)
    repetition_ratio = 0.2
    
    # Split text into words
    words = text.split()
    
    # Process each word individually
    augmented_words = []
    for word in words:
        new_word = ""
        for char in word:
            # Randomly decide whether to repeat the character based on the repetition ratio
            if random.random() < repetition_ratio:
                new_word += char * 2  # Repeat the character
            else:
                new_word += char
        augmented_words.append(new_word)
    
    # Join words back into a sentence
    augmented_text = ' '.join(augmented_words)
    
    print("Original:")
    print(text)
    print("Augmented Text:")
    print(augmented_text)
    return augmented_text

###################### Leet Letters ###################

def leetLettersWithPerturbation(text):
    set_seed()
    
    # Extended leet mapping based on the provided table
    leet_mapping = {
        'a': ['4', '@', '/-\\', '^'], 
        'b': ['8', '|3', '13', 'ß'], 
        'c': ['[', '<', '('], 
        'd': [')', '|)', '[)', 'I>'], 
        'e': ['3', '&', '€'], 
        'f': ['|=', 'ph', 'ƒ'], 
        'g': ['6', '&', '9'], 
        'h': ['#', '|-|', ']-['], 
        'i': ['1', '!', '|', 'eye'], 
        'j': [',_|', '_|', 'ʝ'], 
        'k': ['|<', '|{'], 
        'l': ['1', '|_', '|'], 
        'm': ['/\\/\\', '(V)', '[V]', '|\\/|'], 
        'n': ['/\\/', '|\\|', '[\\]'], 
        'o': ['0', '()', '°'], 
        'p': ['|*', '|>', '9'], 
        'q': ['(_,)', '0,'], 
        'r': ['|2', '12', '.-'], 
        's': ['5', '$', '§'], 
        't': ['7', '+', '†'], 
        'u': ['|_|', '\\_/', 'v'], 
        'v': ['\\/', '|/'], 
        'w': ['\\/\\/', 'vv', '\\/\\/'], 
        'x': ['><', '}{', ')('], 
        'y': ['`/', '¥'], 
        'z': ['2', '7_', '~/_']
    }
    
    # Split the text into words
    words = text.split()
    augmented_words = []
    
    for word in words:
        new_word = ''
        for char in word:
            # Only perturb about `perturbation_ratio` of the characters
            if char.lower() in leet_mapping and random.random() < 0.1:
                leet_options = leet_mapping[char.lower()]
                new_word += random.choice(leet_options)  # Choose a random leet replacement
            else:
                new_word += char  # Keep the original character

        augmented_words.append(new_word)
    
    # Join the augmented words back into a single string
    leet_text = ' '.join(augmented_words)
    
    print("Original:")
    print(text)
    print("Leet Text:")
    print(leet_text)
    return leet_text

############# Whitespace Perturbation ####################
def whiteSpacePerturbation(text):
    set_seed()
    
    aug = naw.SplitAug()
    augmented_text = aug.augment(text)
    print("Original:")
    print(text)
    print("Augmented Text:")
    print(augmented_text[0])
    return augmented_text[0]

############### Homoglyphs Substitution ##################

# Define a dictionary of homoglyphs, case-insensitive
homoglyphs = {
    
    'a': ['α', 'ɑ'],
    'b': ['ß', 'β', 'ᛒ'],
    'c': ['Ꮯ', 'Ⅽ', 'ⅽ'],
    'd': ['ḍ', 'ժ', 'đ'],
    'e': ['ĕ', 'ē', 'ë', 'ê', 'é'],
    'f': ['Ϝ', 'Ｆ', 'ｆ'],
    'g': ['ɢ', 'ｇ', 'Ꮐ'],
    'h': ['ʜ', 'Ꮋ'],
    'i': ['ｉ', 'Ꭵ', 'ɩ'],
    'j': ['ј', 'Ꭻ', 'Ｊ'],
    'k': ['К', 'κ', 'ｋ'],
    'l': ['ʟ', 'ⅼ', 'ｌ'],
    'm': ['Ϻ', 'ⅿ', 'ｍ'],
    'n': ['ɴ', 'Ｎ'],
    'o': ['0', 'o', 'Ⲟ', 'о'],
    'p': ['ρ', 'Ꮲ', 'ｐ'],
    'q': ['Ⴍ', 'Ｑ', 'Q'],
    'r': ['Ի', 'ᚱ', 'Ꮢ'],
    's': ['Ⴝ', 'Ꮪ', 'S'],
    't': ['τ', 'ｔ', 'Τ'],
    'u': ['μ', 'Ա', '⋃'],
    'v': ['Ѵ', 'ⅴ', 'ѵ'],
    'w': ['ѡ', 'ｗ', 'w'],
    'x': ['χ', 'х', 'X'],
    'y': ['γ', 'ʏ', 'Ү'],
    'z': ['Z', 'ｚ']
}

def substitute_with_homoglyphs(text):
    set_seed()
    
    # Convert text to a list of characters for easier substitution
    text_chars = list(text)
    # Calculate the number of characters to substitute
    num_substitutions = int(len(text_chars) * 0.1)
    
    # Get the indices of characters to potentially replace
    eligible_indices = [i for i, char in enumerate(text_chars) if char.lower() in homoglyphs]
    # Randomly select characters to substitute based on the calculated ratio
    selected_indices = random.sample(eligible_indices, min(num_substitutions, len(eligible_indices)))
    
    # Perform substitutions
    for i in selected_indices:
        char = text_chars[i].lower()
        if char in homoglyphs:
            # Choose a homoglyph randomly and keep the original character's case
            replacement = random.choice(homoglyphs[char])
            text_chars[i] = replacement if text_chars[i].islower() else replacement.upper()

    # Join the list back into a string
    return ''.join(text_chars)

##################### Synonym Replacement #################

def synonym_replacement(text):
    set_seed()
        
    aug = naw.SynonymAug(aug_src='ppdb', model_path='ppdb-2.0-tldr')
    augmented_text = aug.augment(text)
    print("Original:")
    print(text)
    print("Augmented Text:")
    print(augmented_text[0])
    return augmented_text[0]


##################### Random Swap ########################

def swap_word(text):
    set_seed()
    
    aug = naw.RandomWordAug(action="swap")
    augmented_text = aug.augment(text)
    print("Original:")
    print(text)
    print("Augmented Text:")
    print(augmented_text[0])
    return augmented_text[0]

    
################### Misspell Word ######################

def misspell_word(text):
    set_seed()
    
    aug = naw.SpellingAug()
    augmented_texts = aug.augment(text)
    print("Original:")
    print(text)
    print("Augmented Texts:")
    print(augmented_texts[0])
    return augmented_texts[0]

################### Abbreviate Word ########################

abbreviations = {
    "and": "&", "you": "u", "are": "r", "okay": "ok", "before": "b4", "ARE":"r", 
    "see": "c", "please": "pls", "be right back": "brb", "laugh out loud": "lol",
    "for": "4", "with": "w/", "people": "ppl", "thanks": "thx", 
    "at": "@", "tonight": "2nite", "by the way": "btw", "talk to you later": "ttyl",
    "your": "ur", "message": "msg", "tomorrow": "tmrw", "night": "nite",
    "great": "gr8", "because": "bc", "favorite": "fave", "whatever": "w/e",
    "weekend": "wknd", "text": "txt", "question": "q", "love": "<3", "why":"y","going":"goin'", "front":"frnt", "and":"&",
    "them":"'em","birthday": "bday", "as soon as possible": "asap", "good night": "gn",
    "good morning": "gm", "before": "b4", "to be honest": "tbh", "okay": "k",
    "in my opinion": "imo", "for your information": "fyi", "don't know": "dunno",
    "I don't know": "idk", "what's up": "sup", "I am": "im", "let me know": "lmk",
    "rolling on the floor laughing": "rofl", "shaking my head": "smh",
    "oh my god": "omg", "be back soon": "bbs", "let's see": "lmk",
    "face to face": "f2f", "for real": "fr", "good luck": "gl", "good game": "gg",
    "no problem": "np", "real life": "irl", "got to go": "gtg", "laughing my ass off": "lmao",
    "to be honest": "tbh", "by the way": "btw", "see you": "cya", 
    "picture": "pic", "boyfriend": "bf", "girlfriend": "gf", "as soon as possible": "asap",
    "with respect to": "wrt", "for the win": "ftw", "what about you": "wbu",
    "no worries": "nw", "do it yourself": "diy", "yes": "y", "tomorrow": "tmr",
    "take care": "tc", "in case you missed it": "icymi", "I love you": "ily",
    "as far as I know": "afaik", "let me think": "lmt", "in real life": "irl",
    "oh I see": "oic", "please call": "plz call", "best friend forever": "bff",
    "don't worry": "dw", "on my way": "omw", "sounds good": "sg", "no big deal": "nbd",
    "for example": "e.g.", "never mind": "nvm", "see you later": "cya",
    "yes or no": "y/n", "love you lots": "lyl", "work from home": "wfh",
    "all right": "aight", "could not care less": "cncl", "might as well": "maw",
    "to be honest": "tbh", "you only live once": "yolo", "what do you mean": "wdym",
    "to be fair": "tbf", "I don't care": "idc", "no offense": "no", "care of": "c/o",
    "for what it's worth": "fwiw", "not safe for work": "nsfw",
    "in my humble opinion": "imho", "shout out": "shoutout", "on the other hand": "otoh",
    "big time": "bt", "good question": "gq", "by": "bye", "to":"2", "next" : "nxt"
}


def abbreviation_replacement(text):
    set_seed()
    
    severity_ratio = 4 / 5
    words = nltk.word_tokenize(text)
    modified_words = []

    for word in words:
        if word.lower() in abbreviations and random.random() < severity_ratio:
            modified_words.append(abbreviations[word.lower()])
        else:
            modified_words.append(word)

    return ' '.join(modified_words)

####### Word Repeatation #######
import random

def repeat_words(text, level=1):
    set_seed()
    
    words = text.split()  # Split the text into individual words
    num_words = len(words)
    
    # Calculate number of words to repeat based on the level
    num_words_to_repeat = int((1 / 5) * num_words)
    indices_to_repeat = random.sample(range(num_words), num_words_to_repeat)  # Randomly pick words to repeat

    repeated_words = []
    
    for i, word in enumerate(words):
        # Repeat the word if its index is in the selected indices
        if i in indices_to_repeat:
            repeated_words.append(' '.join([word] * level))  # Repeat word based on level
        else:
            repeated_words.append(word)

    return ' '.join(repeated_words)  # Join all words back into a single string



########### Back Translation Augmentation ################

def backTranslation(text):
    set_seed()
    
    text = 'The quick brown fox jumped over the lazy dog'
    back_translation_aug = naw.BackTranslationAug(
        from_model_name='facebook/wmt19-en-de', 
        to_model_name='facebook/wmt19-de-en'
    )
    back_translated = back_translation_aug.augment(text)
    return back_translated[0]

######### Sentence Paraphrasing ############
def sentence_paraphrase(text):
    set_seed()
    
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    device = "cuda"

    tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")

    model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base").to(device)

    def paraphrase(
        question,
        num_beams=5,
        num_beam_groups=5,
        num_return_sequences=5,
        repetition_penalty=10.0,
        diversity_penalty=3.0,
        no_repeat_ngram_size=2,
        temperature=0.7,
        max_length=128
    ):
        input_ids = tokenizer(
            f'paraphrase: {question}',
            return_tensors="pt", padding="longest",
            max_length=max_length,
            truncation=True,
        ).input_ids.to(device)
        
        outputs = model.generate(
            input_ids, temperature=temperature, repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences, no_repeat_ngram_size=no_repeat_ngram_size,
            num_beams=num_beams, num_beam_groups=num_beam_groups,
            max_length=max_length, diversity_penalty=diversity_penalty
        )

        res = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return res
    p = random.randint(0,4)
    return paraphrase(text)[p]
    

# Function to apply all perturbation methods
def apply_perturbations(row):
    """Apply all textual corruption functions to a question"""
    question = row['question']
    modified_data = []
    
    corruption_funcs = [
        substituteCharacterByOCR, substituteCharacterByKeyboard, insertCharacterRandomly,
        substituteCharacterRandomly, swapCharacterRandomly, delCharacterRandomly,
        uppercaseRandomly, repeatCharacterRandomly, leetLettersWithPerturbation,
        whiteSpacePerturbation, substitute_with_homoglyphs, synonym_replacement,
        swap_word, misspell_word, abbreviation_replacement, repeat_words,
        backTranslation, sentence_paraphrase
    ]
    
    for func in corruption_funcs:
        try:
            modified_question = func(question)
            modified_data.append({
                'id': row['id'],
                'instance_id': row.get('instance_id', ''),
                'question': question,
                'modified_question': modified_question,
                'modified_question_function_name': func.__name__,
                'answer': row['answer'],
                'A': row['A'],
                'B': row['B'],
                'C': row['C'],
                'D': row['D'],
                'category': row['category'],
                'path': row['path']
            })
        except Exception as e:
            print(f"Error applying {func.__name__}: {e}")
    
    return modified_data


def process_dataframe(input_csv, output_csv='FinalNoisyText.csv'):
    """Process a DataFrame with textual corruptions"""
    df = pd.read_csv(input_csv)
    all_modified = []
    
    for _, row in df.iterrows():
        all_modified.extend(apply_perturbations(row))
    
    final_df = pd.DataFrame(all_modified)
    final_df.to_csv(output_csv, index=False)
    print(f"Saved {len(final_df)} corrupted questions to {output_csv}")
    return final_df


# Example usage:
if __name__ == "__main__":
    process_dataframe('[without images]1_correct_validation.csv', 'Noisy-Denoised_QuestionPairs[new].csv')


# substituteCharacterByKeyboard("The quick brown fox jumps over the lazy dog .")
# substituteCharacterByOCR("The quick brown fox jumps over the lazy dog .")
# insertCharacterRandomly("whi is the man loading the dishwasher in the morning?")
# substituteCharacterRandomly("The quick brown fox jumps over the lazy dog .")
# swapCharacterRandomly("The quick brown fox jumps over the lazy dog .")
# delCharacterRandomly("The quick brown fox jumps over the lazy dog .")
# uppercaseRandomly("The quick brown fox jumps over the lazy dog.")
# repeatCharacterRandomly("The quick brown fox jumps over the lazy dog.")
# leetLettersWithPerturbation("The band, including one man in black jacket and two women standing next to blue corrugated walls.")
# whiteSpacePerturbation("The quick brown fox jumps over the lazy dog.")
# print(substitute_with_homoglyphs("whi is the man loading the dishwasher in the morning?"))


# synonym_replacement("The quick brown fox jumps over the lazy dog")
# swap_word("The quick brown fox jumps over the lazy dog.")
# misspell_word("The quick brown fox jumps over the lazy dog.")
# print(abbreviation_replacement("See you later"))


# print(backTranslation("The quick brown fox jumps over the lazy dog."))
# print(sentence_paraphrase("The quick brown fox jumps over the lazy dog."))

















# ########## Close homophones Swap #######################
# def get_homophones(word):
#     # Generate the phonetic code for the word
#     phonetic_code = doublemetaphone(word)[0]
    
#     # Find words with the same or similar phonetic codes
#     homophones = [w for w in word_list if doublemetaphone(w)[0] == phonetic_code and w != word]
    
#     return homophones

# def close_homophones_swap(text):
#     words = text.split()
#     swapped_text = []

#     for word in words:
#         # Fetch homophones for the current word
#         homophones = get_homophones(word.lower())
        
#         # If we find homophones, randomly choose one; else, keep the original word
#         if homophones:
#             homophone = random.choice(homophones)
#             # Preserve capitalization
#             if word[0].isupper():
#                 homophone = homophone.capitalize()
#             swapped_text.append(homophone)
#         else:
#             swapped_text.append(word)  # No homophone found, keep the original word

#     # Join the list back into a string
#     result_text = ' '.join(swapped_text)
#     print("Original:", text)
#     print("Swapped Text:", result_text)
#     return result_text


############# Random Word Deletion #################

# def delete_word_randomly(text):
#     aug = naw.RandomWordAug()
#     augmented_text = aug.augment(text)
#     print("Original:")
#     print(text)
#     print("Augmented Text:")
#     print(augmented_text)
#     return augmented_text
