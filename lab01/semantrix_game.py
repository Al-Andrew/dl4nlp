import dearpygui.dearpygui as dpg
import nltk
from nltk.corpus import wordnet
import random

# Ensure NLTK data is downloaded
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/brown')
except LookupError:
    nltk.download('brown')


# --- Constants ---
# Main Window Dimensions
MAIN_WINDOW_WIDTH = 1440
MAIN_WINDOW_HEIGHT = 1280

# Game Configuration
NUM_RANDOM_WORDS = 100
MAX_WORDS_IN_LIST = 8
INITIAL_SCORE = 0
BASE_SCORE_POINTS = 100
SIMILARITY_THRESHOLD = 0.25
LOW_SIMILARITY_ADD_THRESHOLD = 0.05
ADD_WORDS_ON_LOW_SIMILARITY = 1
MIN_BROWN_FREQUENCY = 10
WORD_LIST_WINDOW_HEIGHT = 200
INITIAL_WORDS_IN_LIST = 3

# --- Global Game State ---
game_state = {
    "current_score": INITIAL_SCORE,
    "word_list": [], # Stores tuples: (word_string, is_target_word_bool)
    "game_over": False,
    "all_words": [],
    "word_similarities": {}, # Cache for pre-computed similarities
    "played_targets": set(), # Track words that have been successfully guessed (don't re-add as targets)
}

def generate_and_cache_words():
    """
    Generates a list of random words from WordNet and pre-computes
    and caches their similarities.
    """
    print("Generating words and pre-computing similarities...")
    
    # Use words from the Brown corpus as a base for common words and filter by frequency
    from nltk.corpus import brown
    from collections import Counter

    brown_word_counts = Counter(word.lower() for word in brown.words() if word.isalpha() and len(word) > 2)
    frequent_brown_words = {word for word, count in brown_word_counts.items() if count >= MIN_BROWN_FREQUENCY}

    all_wordnet_lemmas = set()
    for synset in wordnet.all_synsets('n'): # Focusing on nouns
        for lemma in synset.lemmas():
            word = lemma.name().lower()
            if '_' not in word and '-' not in word and word.isalpha():
                all_wordnet_lemmas.add(word)
    
    # Filter WordNet lemmas to include only those present in the frequent Brown words
    common_wordnet_words = list(all_wordnet_lemmas.intersection(frequent_brown_words))
    random.shuffle(common_wordnet_words)
    
    game_state["all_words"] = random.sample(common_wordnet_words, min(NUM_RANDOM_WORDS, len(common_wordnet_words)))
    
    for i, word1_str in enumerate(game_state["all_words"]):
        print(f"Caching similarities for word {i+1}/{len(game_state['all_words'])}: {word1_str}")
        word1_synset = wordnet.synsets(word1_str, pos='n')
        if not word1_synset:
            continue
        word1_synset = word1_synset[0] # Take the first synset
        
        game_state["word_similarities"][word1_str] = {}
        
        # Find close words (e.g., synonyms, hypernyms, hyponyms)
        close_words_synsets = set()
        close_words_synsets.add(word1_synset)
        
        # Add hypernyms and hyponyms
        for hypernym in word1_synset.hypernyms():
            close_words_synsets.add(hypernym)
        for hyponym in word1_synset.hyponyms():
            close_words_synsets.add(hyponym)
            
        # Add synonyms (lemmas of the same synset)
        for syn_lemma in word1_synset.lemmas(): # Iterate through lemmas of the synset
            synonym_word = syn_lemma.name().lower()
            if '_' not in synonym_word and '-' not in synonym_word and synonym_word.isalpha():
                synonym_synsets = wordnet.synsets(synonym_word, pos='n')
                if synonym_synsets:
                    close_words_synsets.add(synonym_synsets[0]) # Add the synset of the synonym
        
        for word2_synset in close_words_synsets:
            for word2_lemma in word2_synset.lemmas():
                word2_str = word2_lemma.name().lower()
                if '_' not in word2_str and '-' not in word2_str and word2_str.isalpha():
                    if word1_str != word2_str:
                        similarity = word1_synset.path_similarity(word2_synset)
                        if similarity is not None:
                            game_state["word_similarities"][word1_str][word2_str] = similarity
    print("Similarity caching complete.")

def get_word_similarity(word1_str, word2_str):
    if word1_str in game_state["word_similarities"] and word2_str in game_state["word_similarities"][word1_str]:
        return game_state["word_similarities"][word1_str][word2_str]
    
    # Compute on the fly
    word1_synsets = wordnet.synsets(word1_str, pos='n')
    word2_synsets = wordnet.synsets(word2_str, pos='n')

    if not word1_synsets or not word2_synsets:
        return 0.0

    # Take the most common synset (first one)
    synset1 = word1_synsets[0]
    synset2 = word2_synsets[0]

    similarity = synset1.path_similarity(synset2)
    return similarity if similarity is not None else 0.0

def get_words_in_list():
    return {w for w, _ in game_state["word_list"]}

def pick_new_words(n):
    if n <= 0:
        return
    current = get_words_in_list()
    candidates = [w for w in game_state["all_words"] if w not in current and w not in game_state.get("played_targets", set())]
    random.shuffle(candidates)
    to_add = candidates[:max(0, n)]
    for w in to_add:
        game_state["word_list"].append((w, False))

def remove_current_target():
    removed = None
    new_list = []
    for w, is_target in game_state["word_list"]:
        if is_target and removed is None:
            removed = w
            game_state.setdefault("played_targets", set()).add(w)
            # do not re-add this word; it's been played
        else:
            new_list.append((w, False))
    game_state["word_list"] = new_list
    return removed

def choose_new_target():
    if not game_state["word_list"]:
        pick_new_words(min(INITIAL_WORDS_IN_LIST, MAX_WORDS_IN_LIST))
    if game_state["word_list"]:
        idx = random.randrange(len(game_state["word_list"]))
        w, _ = game_state["word_list"][idx]
        game_state["word_list"][idx] = (w, True)

def start_game():
    game_state["current_score"] = INITIAL_SCORE
    game_state["word_list"] = []
    game_state["game_over"] = False
    game_state["played_targets"] = set()
    
    # Pre-populate word list and select initial target word
    initial_words = random.sample(game_state["all_words"], min(INITIAL_WORDS_IN_LIST, len(game_state["all_words"])))
    target_word_index = random.randrange(len(initial_words)) if initial_words else 0

    for i, word in enumerate(initial_words):
        is_target = (i == target_word_index)
        game_state["word_list"].append((word, is_target))
    
    dpg.set_value("score_text", f"Score: {game_state['current_score']}")
    update_word_list_display() # Helper to update listbox
    dpg.show_item("main_game_window")
    dpg.hide_item("game_over_window")

def end_game():
    game_state["game_over"] = True
    dpg.set_value("final_score_text", f"Your score: {game_state['current_score']}")
    dpg.hide_item("main_game_window")
    dpg.show_item("game_over_window")

def handle_input(sender, app_data):
    if game_state["game_over"]:
        return

    entered_word = dpg.get_value(sender).strip().lower()
    dpg.set_value(sender, "") # Clear input box

    if not entered_word:
        return

    current_target_word_str = ""
    for word_tuple in game_state["word_list"]:
        if word_tuple[1]: # If is_target_word is True
            current_target_word_str = word_tuple[0]
            break
    
    if not current_target_word_str: # Should not happen if game state is correct
        print("Error: No target word found in list.")
        return

    similarity = get_word_similarity(current_target_word_str, entered_word)
    dpg.set_value("similarity_feedback_text", f"Similarity: {similarity:.2f}")
    print(f"Similarity between '{current_target_word_str}' and '{entered_word}': {similarity}")

    if similarity >= SIMILARITY_THRESHOLD:
        game_state["current_score"] += similarity * BASE_SCORE_POINTS
        dpg.set_value("score_text", f"Score: {game_state['current_score']}")
        
        # Remove the current target from the list and mark it as played
        removed_target = remove_current_target()
        
        # Keep the list size stable by adding a replacement word if available
        if len(game_state["word_list"]) < MAX_WORDS_IN_LIST:
            pick_new_words(1)

        # If we reached the maximum number of words, end the game
        if len(game_state["word_list"]) >= MAX_WORDS_IN_LIST:
            end_game()
            return
        
        # Select a new target from the existing words
        choose_new_target()
        
        update_word_list_display()
    else:
        if similarity < LOW_SIMILARITY_ADD_THRESHOLD and len(game_state["word_list"]) < MAX_WORDS_IN_LIST:
            pick_new_words(min(ADD_WORDS_ON_LOW_SIMILARITY, MAX_WORDS_IN_LIST - len(game_state["word_list"])))
            if len(game_state["word_list"]) >= MAX_WORDS_IN_LIST:
                end_game()
                return
            update_word_list_display()
    
    dpg.focus_item("word_input") # Keep input box selected

def restart_game():
    start_game()

def main():
    dpg.create_context()

    with dpg.font_registry():
        default_font = dpg.add_font("./assets/Roboto-Regular.ttf", 48)
        dpg.bind_font(default_font)


    # --- UI Layout ---
    with dpg.window(tag="main_game_window", label="Semantrix Game", autosize=True, no_title_bar=True, no_resize=True, no_move=True):
        dpg.add_text("Score: 0", tag="score_text")
        dpg.add_text("Similarity: N/A", tag="similarity_feedback_text")
        
        dpg.add_spacer(height=10)
        dpg.add_separator()
        dpg.add_spacer(height=10)

        with dpg.child_window(width=-1, height=WORD_LIST_WINDOW_HEIGHT, tag="word_list_window", border=True, autosize_y=False):
            # We will add text items dynamically here, not use a listbox directly
            pass
        
        dpg.add_spacer(height=10)
        dpg.add_separator()
        dpg.add_spacer(height=10)

        dpg.add_input_text(label="Enter word", tag="word_input", on_enter=True, callback=handle_input, width=-1)

    with dpg.window(tag="game_over_window", pos=(MAIN_WINDOW_WIDTH // 2, MAIN_WINDOW_HEIGHT // 2), show=False, autosize=True, no_title_bar=True, no_resize=True, no_move=True):
        dpg.add_text("Game Over!", tag="game_over_title")
        dpg.add_text(f"Your score: {game_state['current_score']}", tag="final_score_text")
        dpg.add_button(label="Restart Game", callback=restart_game, width=-1)
    
    dpg.create_viewport(title='Semantrix Game', width=MAIN_WINDOW_WIDTH, height=MAIN_WINDOW_HEIGHT)
    dpg.setup_dearpygui()

    # Set a single resize callback for both windows
    def resize_callback():
        viewport_width = dpg.get_viewport_width()
        viewport_height = dpg.get_viewport_height()
        dpg.set_item_pos("main_game_window", [(viewport_width - dpg.get_item_width("main_game_window")) // 2, (viewport_height - dpg.get_item_height("main_game_window")) // 2])
        dpg.set_item_pos("game_over_window", [(viewport_width - dpg.get_item_width("game_over_window")) // 2, (viewport_height - dpg.get_item_height("game_over_window")) // 2])

    dpg.set_viewport_resize_callback(resize_callback)

    generate_and_cache_words()
    start_game()

    dpg.set_primary_window("main_game_window", True)
    dpg.show_viewport()
    dpg.start_dearpygui()
    resize_callback()
    dpg.destroy_context() 

def update_word_list_display():
    dpg.delete_item("word_list_window", children_only=True)
    
    for i, (word, is_target) in enumerate(game_state["word_list"]):
        display_text = f"-> {word}" if is_target else word
        dpg.add_text(display_text, parent="word_list_window", tag=f"word_item_{i}")

if __name__ == "__main__":
    main()