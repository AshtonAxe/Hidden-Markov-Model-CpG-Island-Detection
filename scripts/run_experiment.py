!wget https://hgdownload.cse.ucsc.edu/goldenpath/hg38/chromosomes/chr21.fa.gz
!wget https://hgdownload.cse.ucsc.edu/goldenpath/hg38/chromosomes/chr22.fa.gz
!wget https://hgdownload.cse.ucsc.edu/goldenpath/hg38/database/cpgIslandExt.txt.gz
# Schema at https://hgdownload.cse.ucsc.edu/goldenpath/hg38/database/cpgIslandExt.sql

# Decompress input files
!gzip -d /content/cpgIslandExt.txt.gz
!gzip -d /content/chr21.fa.gz
!gzip -d /content/chr22.fa.gz

# Extract CpG regions as a list of coordinate pairs (start and end positions of known CpG islands)
cpg_file = open("/content/cpgIslandExt.txt")
cpg_line = cpg_file.readline()

cpg_regions = []
while(cpg_line):
    cpg_split = cpg_line.split()
    if(cpg_split[1] == "chr21"):
        region_start = int(cpg_split[2])
        region_end = int(cpg_split[3])
        cpg_regions.append([region_start, region_end])
    cpg_line = cpg_file.readline()
cpg_file.close()

# Define the alphabet of valid DNA bases
alphabet = ["A", "C", "G", "T"]
alphabet_dict = {'A' : 0, 'C' : 1, 'G' : 2, 'T' :3 }

# Initialize matrices and distributions for transition probabilities and initial states

# 1. A count_matrix to track transitions between states (what size should this be?)
# 2. An initial_state_distribution to track how often each state occurs

transition_matrix = torch.zeros(8, 8) # rows are from state and cols are to state
initial_state_distribution = torch.zeros(8)

# Open the file for reading
with open("/content/chr21.fa", 'r') as training_file:

    # Skip the header
    training_file.readline()

    def advance_cpg_region(cpg_regions, cpg_index, current_pos):
      while cpg_index < len(cpg_regions) and current_pos >= cpg_regions[cpg_index][1]:
        cpg_index += 1

      in_cpg = False
      if cpg_index < len(cpg_regions):
        # Is current position in a cpg island
        in_cpg = cpg_regions[cpg_index][0] <= current_pos < cpg_regions[cpg_index][1]
      return in_cpg, cpg_index

    current_pos = 0
    prev_nucleotide = None
    cpg_index = 0
    first_state = True

    for line in training_file:
      for nucleo in line.strip().upper():

        in_cpg, cpg_index = advance_cpg_region(cpg_regions, cpg_index, current_pos)
        if nucleo in alphabet:
          num_nuc = alphabet_dict[nucleo]
          current_nuc = (4 if in_cpg else 0) + num_nuc

          if first_state:
            initial_state_distribution[current_nuc] += 1
            first_state = False

          if prev_nucleotide is not None:
            transition_matrix[prev_nucleotide][current_nuc] += 1

          prev_nucleotide = current_nuc
        current_pos += 1

initial_state_distribution = (initial_state_distribution + 1) / (initial_state_distribution.sum() + 8)
transition_matrix = (transition_matrix + 1) / (transition_matrix.sum(dim=1, keepdim=True) + 8)

# Generate emission probability matrix
alpha = 1e-7 # Smoothing parameter

# Initialize the emission matrix
emission_matrix = np.zeros((8, 4))

# Fill in the emission probability matrix
emission_matrix = np.full((8, 4), alpha)
for state in range(8):
  nuc_index = state % 4
  emission_matrix[state, nuc_index] = 1.0 - 3*alpha # each row sums to 1


def load_test_file(path):
    """
    Load and preprocess the file
    """
    with open(path, 'r') as f:
        # Skip header
        f.readline()
        content = f.read().replace('\n', '').upper()  # Convert the entire sequence into a single string
    return content

def run_viterbi(sequence, alphabet, initial_state_distribution, emission_matrix, transition_matrix, start_pos, end_pos):
    """
    Run the Viterbi algorithm to find the most likely sequence of hidden states.

    Args:
        sequence (str): Full DNA sequence (entire chromosome)
        alphabet (list): List of valid nucleotides ['A', 'C', 'G', 'T']
        initial_state_distribution (numpy.array): Initial probabilities for each state
        emission_matrix (numpy.array): Emission probabilities (8x4 matrix)
        transition_matrix (numpy.array): Transition probabilities (8x8 matrix)
        start_pos (int): Starting position to analyze (inclusive)
        end_pos (int): Ending position to analyze (inclusive)

    Returns:
        tuple: (vit_matrix, backpointers)
            - vit_matrix: List of lists containing log probabilities for each state at each position
            - backpointers: List of lists containing the best previous state for each current state
            Both are indexed from 0 and correspond to the filtered region (start_pos to end_pos)

    Note:
        Uses log probabilities to avoid numerical underflow.
        Returns relative positions within the analyzed region.
    """

    alphabet_dict = {'A' : 0, 'C' : 1, 'G' : 2, 'T' :3 }

    target_seq = sequence[start_pos:end_pos + 1].upper()
    seq_length = len(target_seq)

    initial_state_distribution = np.asarray(initial_state_distribution, dtype=float)
    transition_matrix = np.asarray(transition_matrix, dtype=float)
    emission_matrix = np.asarray(emission_matrix, dtype=float)

    log_initial_state_distribution = np.log(initial_state_distribution)
    log_transition_matrix = np.log(transition_matrix)
    log_emission_matrix = np.log(emission_matrix)

    probabilities = np.full((seq_length, 8), 0, dtype=float) # Best probability of reaching each state at each position
    back_pointers = np.full((seq_length, 8), 0, dtype=int) # Records which previous state led to that best probability

    b0 = alphabet_dict[target_seq[0]]
    emission0 = log_emission_matrix[:, b0] #probability of emitting b0 from each of the 8 states

    probabilities[0, :] = emission0 + log_initial_state_distribution
    back_pointers[0, :] = -1

    for i in range(1, seq_length):
      b_i = alphabet_dict[target_seq[i]]

      emission_current = log_emission_matrix[:, b_i]
      log_prob_prev = probabilities[i-1, :][:, None] + log_transition_matrix

      probabilities[i, :] = emission_current + np.max(log_prob_prev, axis=0)
      back_pointers[i, :] = np.argmax(log_prob_prev, axis=0)

    return (probabilities.tolist(), back_pointers.tolist())




def backtrack(vit_matrix, backpointers):
    """
    Backtrack through the Viterbi matrix to find the most likely state sequence.

    Args:
        vit_matrix (list): Matrix of log probabilities from Viterbi algorithm
        backpointers (list): Matrix of backpointers from Viterbi algorithm

    Returns:
        list: Binary sequence (list) where 1 indicates CpG island, 0 indicates background
              (states 0-3 → 0, states 4-7 → 1)
    """
    conversion_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1}
    path = []
    final_state = max(range(len(backpointers[-1])), key=lambda x: vit_matrix[-1][x])
    path.append(final_state)

    for i in range(len(backpointers) - 1, 0, -1):

      current_state = backpointers[i][path[-1]]
      path.append(current_state)

    path.reverse()
    binary_sequence = [conversion_dict[x] for x in path]
    return binary_sequence


def extract_regions(cpg_results, start_pos):
    """
    Extract CpG island regions from binary classification results.

    Args:
        cpg_results (list): Binary list where 1 indicates CpG island position
                           (indexed from 0, representing relative positions in the analyzed region)
        start_pos (int): Starting position of the analyzed region (for coordinate conversion)

    Returns:
        list: List of [start, end] coordinate pairs for each CpG island region
              Coordinates are converted from relative positions to absolute
              chromosome positions (adding start_pos offset)
    """
    cpg_islands = []

    current_island_truthy = False
    current_island = [0, 0]

    for idx, binary in enumerate(cpg_results):
      if binary == 1 and current_island_truthy == False:
        current_island_truthy = True
        current_island[0] = start_pos + idx

      elif binary == 0 and current_island_truthy == True:
        current_island_truthy = False
        current_island[1] = start_pos + idx
        cpg_islands.append(current_island)
        current_island = [0,0]

      elif binary == 1 and idx == len(cpg_results) - 1:
        if current_island_truthy:
          current_island[1] = start_pos + idx
          cpg_islands.append(current_island)
        else:
          cpg_islands.append([start_pos + idx, start_pos + idx])

    return cpg_islands

# Load chromosome 22
test_sequence = load_test_file("/content/chr22.fa")

# Define the region of interest (SOX10 gene region)
START_POS = 38000000
END_POS = 39000000

vit_matrix, backpointers = run_viterbi(test_sequence, alphabet, initial_state_distribution, emission_matrix, transition_matrix, START_POS, END_POS)
cpg_results = backtrack(vit_matrix, backpointers)
cpg_regions = extract_regions(cpg_results, START_POS)
print(f'Regions found = {cpg_regions}')
print(f'Number of regions found = {len(cpg_regions)}')

def get_overlap(region1, region2):
    """
    Calculate overlap between two regions.

    Args:
        region1 (list): [start, end] coordinates of first region
        region2 (list): [start, end] coordinates of second region

    Returns:
        int: Length of overlap between the regions

    Hint: Find the intersection of the two regions and return its length.
    Handle cases where regions don't overlap (return 0).
    """
    return max(0, min(region1[1], region2[1]) - max(region1[0], region2[0]))

def check_overlap(coord, region):
    """
    Check whether a given coordinate falls within a region.

    Args:
        coord (int): Position to check
        region (list): [start, end] coordinates of region

    Returns:
        bool: True if coordinate is within region, False otherwise

    Hint: Check if coord is between region[0] and region[1] (inclusive)
    """
    return region[0] <= coord <= region[1]

def load_ground_truth_regions(start_pos, end_pos):
    """
    Load annotated CpG regions that overlap with the target region.

    Args:
        start_pos (int): Start of target region
        end_pos (int): End of target region

    Returns:
        list: List of [start, end] coordinate pairs for CpG islands in target region
    """
    overlap_regions = []
    with open('/content/cpgIslandExt.txt', 'r') as file:
      line = file.readline()
      while line:
        columns = line.split()
        if columns[1] == 'chr22':
          region_start = int(columns[2])
          region_end = int(columns[3])
          if (region_end >= start_pos and region_end <= end_pos) or (region_start >= start_pos and region_start <= end_pos):
            overlap_regions.append([region_start, region_end])

        line = file.readline()

    return overlap_regions

def evaluate_islands(predicted_regions, true_regions):
    """
    Evaluate predicted CpG islands against ground truth using region-level analysis.

    Args:
        predicted_regions (list): List of [start, end] pairs from your model
        true_regions (list): List of [start, end] pairs from annotations

    Returns:
        numpy.array: 2x2 confusion matrix for island-level predictions

    Hint:
    1. Create 2x2 confusion matrix: [[TP, FN], [FP, TN]]
    2. For each predicted region, check if it overlaps significantly with any true region
    3. For regions between predicted islands, check if they contain true CpG islands
    4. Use get_overlap() and check if overlap >= 50% of the true region length
    """

    confusion_matrix = [[0, 0], [0, 0]]

    for p_region in predicted_regions:
      true_positive = False
      for t_region in true_regions:
        if (get_overlap(p_region, t_region) / (t_region[1] - t_region[0])) >= 0.50:
          confusion_matrix[0][0] += 1
          true_positive = True
          break

      if not true_positive:
        confusion_matrix[1][0] += 1

    for t_region in true_regions:
      false_negative = True
      for p_region in predicted_regions:
        if (get_overlap(p_region, t_region) / (t_region[1] - t_region[0])) >= 0.50:
          false_negative = False
          break

      if false_negative:
        confusion_matrix[0][1] += 1

    return np.array(confusion_matrix)
  
def evaluate_positions(cpg_results, true_regions, base_offset):
    """
    Evaluate predictions at the nucleotide position level.

    Args:
        cpg_results (list): Binary list where 1 = CpG island prediction, 0 = background
        true_regions (list): List of [start, end] pairs from annotations
        base_offset (int): Starting position offset (to convert relative to absolute coordinates)

    Returns:
        numpy.array: 2x2 confusion matrix for position-level predictions

    Hint:
    1. Create 2x2 confusion matrix: [[TP, FN], [FP, TN]]
    2. For each position in cpg_results:
       - Convert relative position to absolute: base_offset + position
       - Check if absolute position falls within any true CpG region using check_overlap()
       - Compare prediction (0 or 1) with ground truth (in CpG region or not)
       - Update appropriate cell in confusion matrix
    """
    confusion_matrix = [[0, 0], [0, 0]]

    for pos, value in enumerate(cpg_results):
      coord = base_offset + pos
      in_region = False
      for region in true_regions:
        if check_overlap(coord, region):
          in_region = True
          break
      if in_region and value == 1:
        confusion_matrix[0][0] += 1
      elif in_region and value == 0:
        confusion_matrix[0][1] += 1
      elif not in_region and value == 1:
        confusion_matrix[1][0] += 1
      elif not in_region and value == 0:
        confusion_matrix[1][1] += 1

    return np.array(confusion_matrix)

def calculate_rates_from_confusion_matrix(confusion_matrix):
    """
    Calculate false positive and false negative rates from confusion matrix.

    Args:
        confusion_matrix (numpy.array): 2x2 matrix [[TP, FN], [FP, TN]]

    Returns:
        tuple: (false_positive_rate, false_negative_rate)

    Hint:
    - Confusion matrix structure: [[TP, FN], [FP, TN]]
    - False Positive Rate = FP / (FP + TN) = confusion_matrix[1,0] / sum(confusion_matrix[1,:])
    - False Negative Rate = FN / (TP + FN) = confusion_matrix[0,1] / sum(confusion_matrix[0,:])
    """
    return (confusion_matrix[1,0] / sum(confusion_matrix[1,:])), (confusion_matrix[0,1] / sum(confusion_matrix[0,:]))

# Constants
START_POS = 38000000
END_POS = 39000000
BASE_OFFSET = 38000000

# Load ground truth regions for chr22 in the target region
true_regions = load_ground_truth_regions(START_POS, END_POS)

# Get your predicted regions and position-level results
vit_matrix, backpointers = run_viterbi(test_sequence, alphabet, initial_state_distribution, emission_matrix, transition_matrix, START_POS, END_POS)
cpg_results = backtrack(vit_matrix, backpointers)
predicted_regions = extract_regions(cpg_results, START_POS)

print(f"Found {len(true_regions)} annotated CpG islands in target region")
print(f"Predicted {len(predicted_regions)} CpG islands")

# Method 1: Island-level evaluation
print("\n=== Island-Level Evaluation ===")
island_confusion_matrix = evaluate_islands(predicted_regions, true_regions)
print("Island Confusion Matrix:")
print(island_confusion_matrix)

island_fpr, island_fnr = calculate_rates_from_confusion_matrix(island_confusion_matrix)
print(f"Island-level False Positive Rate: {island_fpr:.3f}")
print(f"Island-level False Negative Rate: {island_fnr:.3f}")

# Method 2: Position-level evaluation
print("\n=== Position-Level Evaluation ===")
position_confusion_matrix = evaluate_positions(cpg_results, true_regions, BASE_OFFSET)
print("Position Confusion Matrix:")
print(position_confusion_matrix)

position_fpr, position_fnr = calculate_rates_from_confusion_matrix(position_confusion_matrix)
print(f"Position-level False Positive Rate: {position_fpr:.3f}")
print(f"Position-level False Negative Rate: {position_fnr:.3f}")

def calculate_overlap_percentage(pred_region, true_region):
    """
    Calculate what percentage of the predicted region overlaps with the true region.

    Args:
        pred_region (list): [start, end] of predicted CpG island
        true_region (list): [start, end] of true CpG island

    Returns:
        float: Percentage of predicted region that overlaps (0.0 to 1.0)
    """

    return get_overlap(pred_region, true_region) / len(pred_region)

def evaluate_predictions(predicted_regions, true_regions):
    """
    Evaluate predicted CpG islands against ground truth annotations.

    Args:
        predicted_regions (list): List of [start, end] pairs from your model
        true_regions (list): List of [start, end] pairs from annotations

    Returns:
        tuple: (true_positives, false_positives, false_negatives)
    """
    true_p = 0
    false_p = 0
    false_n = 0

    matched = [False] * len(true_regions)

    for p_region in predicted_regions:
      true_positive = False
      for idx, t_region in enumerate(true_regions):
        if (get_overlap(p_region, t_region) / (p_region[1] - p_region[0])) >= 0.50:
          true_p += 1
          matched[idx] = True
          true_positive = True
          break

      if not true_positive:
        false_p += 1

    false_n = sum(1 for truthy in matched if not truthy)

    return (true_p, false_p, false_n)


def calculate_rates(true_positives, false_positives, false_negatives):
    """
    Calculate false positive and false negative rates.

    Args:
        true_positives (int): Number of correct predictions
        false_positives (int): Number of incorrect positive predictions
        false_negatives (int): Number of missed true regions

    Returns:
        tuple: (false_positive_rate, false_negative_rate)
    """
    return (false_positives / (false_positives + true_positives)), (false_negatives / (false_negatives + true_positives))


# Load ground truth annotations for chr22 in the target region
def load_ground_truth_regions(start_pos, end_pos):
    """
    Load annotated CpG regions that overlap with the target region.
    """
    overlap_regions = []
    with open('/content/cpgIslandExt.txt', 'r') as file:
      line = file.readline()
      while line:
        columns = line.split()
        if columns[1] == 'chr22':
          region_start = int(columns[2])
          region_end = int(columns[3])
          if (region_end >= start_pos and region_end <= end_pos) or (region_start >= start_pos and region_start <= end_pos):
            overlap_regions.append([region_start, region_end])

        line = file.readline()

    return overlap_regions

# Evaluate your model
START_POS = 38000000
END_POS = 39000000

true_regions = load_ground_truth_regions(START_POS, END_POS)
vit_matrix, backpointers = run_viterbi(test_sequence, alphabet, initial_state_distribution, emission_matrix, transition_matrix, START_POS, END_POS)
cpg_results = backtrack(vit_matrix, backpointers)
predicted_regions = extract_regions(cpg_results, START_POS)

# Calculate performance metrics
tp, fp, fn = evaluate_predictions(predicted_regions, true_regions)
fpr, fnr = calculate_rates(tp, fp, fn)

print(f"True Positives: {tp}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"False Positive Rate: {fpr:.3f}")
print(f"False Negative Rate: {fnr:.3f}")
