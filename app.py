import os
import time
import torch
import difflib
import re
import string
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import T5TokenizerFast, pipeline
from cursor_t5_modeling import CursorT5ForConditionalGeneration
from pathlib import Path

app = Flask(__name__)
CORS(app)

# ========== 1. Load the Tap&Say correction model and tokenizer ==========
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "saved_model" / "correction-normal-multiple cursor-uniform-point-mask-5"

model = CursorT5ForConditionalGeneration.from_pretrained(str(MODEL_DIR), 'point')
tokenizer = T5TokenizerFast.from_pretrained("google/flan-t5-base")

# ========== 2. Initialize the punctuation restoration pipeline ==========
punctuation_pipeline = pipeline(
    "token-classification",
    model="oliverguhr/fullstop-punctuation-multilang-large",
    aggregation_strategy="simple"
)

# Generation parameter settings
NUM_BEAMS = 5
MAX_NEW_TOKENS = 50
NUM_RETURN_SEQUENCES = 5  # Generate multiple candidates for filtering

def restore_punctuation_with_huggingface(text):
    print("========== [Server] /restore_punctuation => Punctuation restoration starts ==========")
    print("Text input for punctuation restoration:", text)
    predictions = punctuation_pipeline(text)
    print("Predicted results returned by the punctuation model:", predictions)

    def insert_punctuation(predictions, original_text):
        words = original_text.split()
        restored_text = ""
        word_index = 0
        for prediction in predictions:
            entity = prediction['entity_group']
            word = prediction['word']
            token_words = word.split()
            for token in token_words:
                if word_index < len(words):
                    restored_text += words[word_index]
                    if entity != '0':
                        restored_text += entity
                    restored_text += " "
                    word_index += 1
        while word_index < len(words):
            restored_text += words[word_index] + " "
            word_index += 1
        return restored_text.strip()

    restored = insert_punctuation(predictions, text)
    print("Text after punctuation restoration:", restored)
    print("========== [Server] /restore_punctuation => Punctuation restoration ends ==========")
    return restored

def compute_diff_ranges(original: str, corrected: str):
    sm = difflib.SequenceMatcher(None, original, corrected)
    diff_ranges = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != 'equal':
            diff_ranges.append((j1, j2))
    return diff_ranges

def is_single_sentence(candidate: str) -> bool:
    candidate = candidate.strip()
    if not candidate:
        return False
    if candidate[-1] not in ".!?":
        return False
    count = candidate.count('.') + candidate.count('!') + candidate.count('?')
    return count == 1

def remove_punctuation_for_words(text: str) -> str:
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def remove_punctuation_and_lower(text: str) -> str:
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator).lower()

def filter_and_rank_corrections(original_text: str,
                                all_candidates: list,
                                merged_string: str,
                                min_accept=3):
    print("\n[Filter & Rank] Start filtering and sorting candidate results...")

    parts = merged_string.split(" <||> ")
    if len(parts) >= 2:
        original_part = parts[0]
        correction_part = parts[1]
    else:
        original_part = merged_string
        correction_part = ""

    allowed_words = set(
        remove_punctuation_for_words(
            (original_part + " " + correction_part)
        ).lower().split()
    )

    accepted = []
    seen = set()

    for candidate_str, candidate_prob in all_candidates:
        reasons = []
        diff_ranges = compute_diff_ranges(original_text, candidate_str)

        if not diff_ranges:
            reasons.append("Identical to original text")

        if correction_part:
            correction_clean = remove_punctuation_and_lower(correction_part).strip()
            candidate_clean = remove_punctuation_and_lower(candidate_str)
            if correction_clean and (correction_clean not in candidate_clean):
                reasons.append("Missing correction phrase")

        if candidate_str in seen:
            reasons.append("Duplicate candidate")

        if not is_single_sentence(candidate_str):
            reasons.append("Not a single correct sentence")

        if re.search(r' {2,}', candidate_str):
            reasons.append("Contains extra spaces")

        normalized_candidate = set(remove_punctuation_for_words(candidate_str).lower().split())
        if normalized_candidate and not normalized_candidate.issubset(allowed_words):
            reasons.append("Contains new words not in merged_string")

        print("-------------------------------------------------")
        print(f"Candidate: {candidate_str}")
        print(f"Probability: {candidate_prob}")
        print("Diff ranges:", diff_ranges)

        if reasons:
            print("[Filtered out] Reason(s):", ", ".join(reasons))
        else:
            print("[Accepted]")
            accepted.append((candidate_str, candidate_prob, diff_ranges))
            seen.add(candidate_str)

    accepted.sort(key=lambda x: x[1], reverse=True)
    top_k = accepted[:min_accept]

    print("\n[Filter & Rank] Final retained candidates:")
    for cand_str, cand_prob, ranges in top_k:
        print(f"  - {cand_str} (prob={cand_prob}, diff_ranges={ranges})")
    print("=================================================\n")

    return top_k

@app.route("/")
def home():
    return "<h1>Flask server has started!</h1><p>You can visit /correct and /restore_punctuation.</p>"

@app.route("/restore_punctuation", methods=["POST"])
def restore_punctuation():
    data = request.json
    print("========== [Server] /restore_punctuation is called ==========")
    print("Data received by /restore_punctuation:", data)
    voice_text = data.get("voice_text", "")
    if not voice_text:
        print("voice_text is empty, return an empty string.")
        return jsonify({"restored_text": ""})
    restored_text = restore_punctuation_with_huggingface(voice_text)
    print("Text after punctuation restoration:", restored_text)
    print("========== [Server] /restore_punctuation => Return to front end ==========")
    return jsonify({"restored_text": restored_text})

@app.route("/correct", methods=["POST"])
def correct_text():
    begin_time = time.time()
    data = request.json
    print("========== [Server] /correct is called ==========")
    print("Data received by /correct:", data)
    merged_string = data.get("merged_string", "")
    touch_location = data.get("touch_location", 0)

    if not merged_string:
        print("merged_string is empty, cannot generate correction suggestions.")
        return jsonify({"filtered_corrections": []})

    print("merged_string:", merged_string)
    print("touch_location:", touch_location)

    parts = merged_string.split(" <||> ")
    if len(parts) > 1:
        parts[1] = parts[1].lower()
        merged_string = " <||> ".join(parts)
    print("========== After forcing <||> part to lowercase: ", merged_string)

    parts = merged_string.split(" <||> ")
    original_text = parts[0] if parts else merged_string

    source = tokenizer(
        merged_string,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        max_length=512
    )
    source["cursor_pos"] = torch.LongTensor([touch_location])

    generated = model.generate(
        source["input_ids"],
        decoder_start_token_id=model.config.decoder_start_token_id,
        num_beams=NUM_BEAMS,
        max_new_tokens=MAX_NEW_TOKENS,
        num_return_sequences=NUM_RETURN_SEQUENCES,
        return_dict_in_generate=True,
        output_scores=True
    )

    generation_time = time.time() - begin_time

    sequences = generated.sequences[:, 1:]
    suggestions = tokenizer.batch_decode(sequences, skip_special_tokens=True)
    print("Generated correction suggestions:", suggestions)

    probs = torch.stack(generated.scores, dim=1).softmax(-1)
    gen_probs = torch.gather(probs, 2, sequences[:, :, None]).squeeze(-1)
    unique_prob_per_sequence = gen_probs.prod(-1)

    raw_candidates = []
    for cand_str, cand_prob in zip(suggestions, unique_prob_per_sequence.tolist()):
        raw_candidates.append((cand_str, cand_prob))

    final_candidates = filter_and_rank_corrections(
        original_text=original_text,
        all_candidates=raw_candidates,
        merged_string=merged_string,
        min_accept=3
    )

    filtered_list = []
    for cand_str, cand_prob, diff_ranges in final_candidates:
        filtered_list.append({
            "corrected_text": cand_str,
            "prob": cand_prob,
            "diff_ranges": diff_ranges
        })

    response = {
        "original_text": original_text,
        "filtered_corrections": filtered_list,
        "model_runtime": generation_time,
        "total_server_time": time.time() - begin_time
    }
    print("Response returned:", response)
    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
