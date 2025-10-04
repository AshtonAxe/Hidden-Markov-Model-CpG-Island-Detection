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
