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
