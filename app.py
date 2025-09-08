import os
import uuid
import librosa
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jensenshannon
from fastdtw import fastdtw
from flask import Flask, render_template, request

app = Flask(__name__)

# --- Swara and Raga Definitions ---

SWARA_MAP = {
    0: 'Sa', 1: 'Ri1', 2: 'Ri2', 3: 'Ga1', 4: 'Ga2', 5: 'Ma1', 6: 'Ma2',
    7: 'Pa', 8: 'Dha1', 9: 'Dha2', 10: 'Ni1', 11: 'Ni2'
}

# Define the valid swara indices (0-11) for each raga
RAGA_SCALES = {
    'Shankarabharanam': [0, 2, 4, 5, 7, 9, 11], # S, R2, G2, M1, P, D2, N2
    'Kalyani':          [0, 2, 4, 6, 7, 9, 11], # S, R2, G2, M2, P, D2, N2
    'Mayamalavagowla':  [0, 1, 4, 5, 7, 8, 11], # S, R1, G2, M1, P, D1, N2
    'Kharaharapriya':   [0, 2, 3, 5, 7, 9, 10], # S, R2, G1, M1, P, D2, N1
    'Gambheeranattai': [0, 4, 5, 7, 11], # S, G2, M1, P, N2
    'Navaroj':          [0, 2, 4, 5, 7, 9, 11], # S, R2, G2, M1, P, D2, N2
    'Kurinji':         [0, 2, 4, 5, 7, 9, 11],  # S, R2, G2, M1, P, D2, N2
    'Neelambari':      [0, 2, 4, 5, 7, 9, 11]  # S, R2, G2, M1, P, D2, N2
}

BASE_FREQ = 261.63 # Default tonic (C4)

# --- Core Audio Processing Functions ---

def estimate_tonic(y, sr):
    f0, _, _ = librosa.pyin(y, fmin=60, fmax=500, sr=sr)
    f0_clean = f0[np.isfinite(f0)]
    if len(f0_clean) == 0:
        return BASE_FREQ
    hist, edges = np.histogram(f0_clean, bins=120, range=(60, 500))
    peak_index = np.argmax(hist)
    return (edges[peak_index] + edges[peak_index + 1]) / 2

def shift_to_reference_tonic(y, sr, current_sa, reference_sa):
    semitone_shift = 12 * np.log2(reference_sa / current_sa)
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=semitone_shift)

def hz_to_swara_raga_aware(freq, tonic, raga_scale_indices):
    """
    Maps a frequency to the closest note within a specific raga's scale.
    """
    if freq <= 0:
        return None
    
    semitones_from_tonic = 12 * np.log2(freq / tonic)
    note_in_octave = semitones_from_tonic % 12
    
    # Find the index of the closest valid swara from the raga's scale
    closest_swara_index = min(raga_scale_indices, key=lambda x: abs(x - note_in_octave))

    octave = round(semitones_from_tonic - note_in_octave) // 12
    base_swara = SWARA_MAP.get(closest_swara_index)
    
    if base_swara is None:
        return None
        
    return f"{base_swara}_{octave}"

def extract_note_sequence_with_durations(y, sr, tonic, raga_scale_indices, hop_length=512, min_duration_frames=4):
    f0, _, _ = librosa.pyin(y, fmin=60, fmax=1200, sr=sr, hop_length=hop_length)
    notes = []
    durations = []
    current_note = None
    current_duration = 0

    for freq in f0:
        note = hz_to_swara_raga_aware(freq, tonic, raga_scale_indices) if freq > 0 else 'Rest'
        if note == current_note:
            current_duration += 1
        else:
            if current_note and current_note != 'Rest' and current_duration >= min_duration_frames:
                notes.append(current_note)
                durations.append(current_duration)
            current_note = note
            current_duration = 1

    if current_note and current_note != 'Rest' and current_duration >= min_duration_frames:
        notes.append(current_note)
        durations.append(current_duration)
        
    return notes, durations

def compute_duration_weighted_distribution(notes, durations):
    if not durations: return {}
    total_duration = sum(durations)
    counter = {}
    for n, d in zip(notes, durations):
        counter[n] = counter.get(n, 0) + d
    return {k: v / total_duration for k, v in counter.items() if v / total_duration > 0.02}

def extract_pitch_contour(y, sr, tonic):
    f0, _, _ = librosa.pyin(y, fmin=60, fmax=1200, sr=sr)
    # Use np.nan for unvoiced frames for cleaner plotting
    contour = [12 * np.log2(f / tonic) if f and f > 0 else np.nan for f in f0]
    return np.array(contour)

def flatten_octaves(distribution):
    flat = Counter()
    for note, val in distribution.items():
        if note != 'Rest':
            swara = note.split('_')[0]
            flat[swara] += val
    return dict(flat)

def compute_ngram_similarity(notes1, notes2, n=2):
    if len(notes1) < n or len(notes2) < n: return 0.0
    ngrams1 = list(zip(*[notes1[i:] for i in range(n)]))
    ngrams2 = list(zip(*[notes2[i:] for i in range(n)]))
    counts1 = Counter(ngrams1)
    counts2 = Counter(ngrams2)
    all_keys = set(counts1) | set(counts2)
    if not all_keys: return 1.0
    vec1 = np.array([counts1.get(k, 0) for k in all_keys])
    vec2 = np.array([counts2.get(k, 0) for k in all_keys])
    vec1 = vec1 / vec1.sum()
    vec2 = vec2 / vec2.sum()
    return 1 - jensenshannon(vec1, vec2)

def compute_dtw_similarity(c1, c2):
    # Remove NaNs and align length for better DTW comparison
    c1_clean = c1[~np.isnan(c1)]
    c2_clean = c2[~np.isnan(c2)]
    if len(c1_clean) == 0 or len(c2_clean) == 0:
        return 0.0
    dist, _ = fastdtw(c1_clean, c2_clean, dist=lambda x, y: abs(x - y))
    return np.exp(-dist / max(len(c1_clean), len(c2_clean)))

def visualize_comparison(dist1, dist2, notes1, notes2, contour1, contour2, file_id, raga_name1, raga_name2):
    # Note Distribution Plot
    flat1 = flatten_octaves(dist1)
    flat2 = flatten_octaves(dist2)
    all_swaras = list(SWARA_MAP.values())
    v1 = [flat1.get(n, 0) for n in all_swaras]
    v2 = [flat2.get(n, 0) for n in all_swaras]

    plt.figure(figsize=(12, 5))
    x = np.arange(len(all_swaras))
    plt.bar(x - 0.2, v1, width=0.4, label=raga_name1)
    plt.bar(x + 0.2, v2, width=0.4, label=raga_name2)
    plt.xticks(x, all_swaras, rotation=45)
    plt.ylabel("Proportional Duration")
    plt.title("Note Distribution (Octave-Agnostic)")
    plt.legend()
    plt.tight_layout()

    # Create filename and save the plot
    note_img_filename = f'{file_id}_note.png'
    plt.savefig(os.path.join('static', note_img_filename))
    plt.close()

    # N-gram Plotting function
    def plot_ngrams(notesA, notesB, n, title, filename_suffix):
        if len(notesA) < n or len(notesB) < n:
            return None
            
        ngramsA = Counter(zip(*[notesA[i:] for i in range(n)]))
        ngramsB = Counter(zip(*[notesB[i:] for i in range(n)]))
        combined_counts = ngramsA + ngramsB
        top_ngrams = [item[0] for item in combined_counts.most_common(15)]
        
        if not top_ngrams:
            return None

        valsA = [ngramsA.get(k, 0) for k in top_ngrams]
        valsB = [ngramsB.get(k, 0) for k in top_ngrams]
        labels = ['-'.join(k) for k in top_ngrams]

        plt.figure(figsize=(10, 6))
        y_pos = np.arange(len(labels))
        plt.barh(y_pos - 0.2, valsA, height=0.4, label=raga_name1)
        plt.barh(y_pos + 0.2, valsB, height=0.4, label=raga_name2)
        plt.yticks(y_pos, labels)
        plt.xlabel("Frequency Count")
        plt.title(title)
        plt.legend()
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        # Create filename, save plot, and return ONLY the filename
        ngram_filename = f'{file_id}_{filename_suffix}.png'
        plt.savefig(os.path.join('static', ngram_filename))
        plt.close()
        return ngram_filename

    # Call the inner function and store just the filenames
    bigram_img_filename = plot_ngrams(notes1, notes2, 2, "Top 15 Bigrams (Melodic Phrases)", "bigram")
    trigram_img_filename = plot_ngrams(notes1, notes2, 3, "Top 15 Trigrams (Melodic Phrases)", "trigram")

    # Pitch Contour Plot
    plt.figure(figsize=(12, 4))
    sr_for_time_calc = 22050 # Assuming this default, or pass sr in
    hop_length_for_time_calc = 512
    time1 = np.arange(len(contour1)) * (hop_length_for_time_calc / sr_for_time_calc)
    time2 = np.arange(len(contour2)) * (hop_length_for_time_calc / sr_for_time_calc)
    plt.plot(time1, contour1, label=raga_name1, alpha=0.7)
    plt.plot(time2, contour2, label=raga_name2, alpha=0.7)
    plt.title("Pitch Contour (semitones from Sa)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Semitones")
    plt.legend()
    plt.tight_layout()
    
    # Create filename and save the plot
    contour_img_filename = f'{file_id}_contour.png'
    plt.savefig(os.path.join('static', contour_img_filename))
    plt.close()

    # **THE FIX**: Return only the filenames, not the full paths.
    return note_img_filename, bigram_img_filename, trigram_img_filename, contour_img_filename

# --- Main Comparison Logic and Flask Routes ---

def compare_ragas(file1, file2, raga_name1, raga_name2):
    y1, sr1 = librosa.load(file1, sr=None, mono=True)
    y2, sr2 = librosa.load(file2, sr=None, mono=True)

    sa1 = estimate_tonic(y1, sr1)
    sa2 = estimate_tonic(y2, sr2)
    y2_shifted = shift_to_reference_tonic(y2, sr2, sa2, sa1)
    
    # Get the correct scales for note extraction
    raga_scale1 = RAGA_SCALES[raga_name1]
    raga_scale2 = RAGA_SCALES[raga_name2]

    notes1, dur1 = extract_note_sequence_with_durations(y1, sr1, sa1, raga_scale1)
    notes2, dur2 = extract_note_sequence_with_durations(y2_shifted, sr2, sa1, raga_scale2)
    
    dist1 = compute_duration_weighted_distribution(notes1, dur1)
    dist2 = compute_duration_weighted_distribution(notes2, dur2)

    all_notes = sorted(set(dist1) | set(dist2))
    vec1 = np.array([dist1.get(n, 0) for n in all_notes]).reshape(1, -1)
    vec2 = np.array([dist2.get(n, 0) for n in all_notes]).reshape(1, -1)

    note_sim = 0.0 if vec1.shape[1] == 0 else cosine_similarity(vec1, vec2)[0][0]
    bigram_sim = compute_ngram_similarity(notes1, notes2, 2)
    trigram_sim = compute_ngram_similarity(notes1, notes2, 3)
    
    contour1 = extract_pitch_contour(y1, sr1, sa1)
    contour2 = extract_pitch_contour(y2_shifted, sr2, sa1)
    contour_sim = compute_dtw_similarity(contour1, contour2)

    file_id = str(uuid.uuid4())
    imgs = visualize_comparison(dist1, dist2, notes1, notes2, contour1, contour2, file_id, raga_name1, raga_name2)

    return {
        "note_similarity": round(note_sim, 3),
        "bigram_similarity": round(bigram_sim, 3),
        "trigram_similarity": round(trigram_sim, 3),
        "contour_similarity": round(contour_sim, 3),
        "note_img": imgs[0], "bigram_img": imgs[1], "trigram_img": imgs[2], "contour_img": imgs[3]
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file1' not in request.files or 'file2' not in request.files:
            return "Please upload both files", 400
        
        file1 = request.files['file1']
        file2 = request.files['file2']
        raga1 = request.form['raga1']
        raga2 = request.form['raga2']

        # Create directories if they don't exist
        os.makedirs("uploads", exist_ok=True)
        os.makedirs("static", exist_ok=True)
        
        path1 = os.path.join("uploads", file1.filename)
        path2 = os.path.join("uploads", file2.filename)
        file1.save(path1)
        file2.save(path2)

        result = compare_ragas(path1, path2, raga1, raga2)

        return render_template("results.html", 
            raga1_name=raga1,
            raga2_name=raga2,
            result=result
        )
    # For GET request, pass the list of raga names to the template
    return render_template("index.html", raga_names=list(RAGA_SCALES.keys()))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)