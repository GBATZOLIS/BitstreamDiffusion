"""Sudoku puzzle generator with deterministic seeding.

The core generation logic (grid filling via backtracking, cell
removal with unique-solution checks) is adapted from Ali Alp's
sudoku generator: https://github.com/alicommit-malp/sudoku

We extended the original code with:
  - Deterministic per-puzzle seeding for reproducibility
  - Parallelized generation with correct worker seeds
  - Deduplication to guarantee disjoint train/valid splits
  - Conversion to the [BOS] puzzle [BOS] solution format
"""

import random
from multiprocessing import Pool

from tqdm import tqdm


DIFFICULTY_TO_CLUES = {
  'easy': 40,
  'medium': 35,
  'hard': 30,
}


def _is_valid(board, row, col, num):
  """Check if placing `num` at (row, col) is valid."""
  for i in range(9):
    if board[row][i] == num or board[i][col] == num:
      return False
  box_r = row - row % 3
  box_c = col - col % 3
  for i in range(3):
    for j in range(3):
      if board[box_r + i][box_c + j] == num:
        return False
  return True


def _fill_grid(grid, rng):
  """Fill an empty 9x9 grid with a valid solution via backtracking."""
  for i in range(9):
    for j in range(9):
      if grid[i][j] == 0:
        nums = list(range(1, 10))
        rng.shuffle(nums)
        for num in nums:
          if _is_valid(grid, i, j, num):
            grid[i][j] = num
            if _fill_grid(grid, rng):
              return True
            grid[i][j] = 0
        return False
  return True


def _count_solutions(grid, limit=2):
  """Count solutions up to `limit` (early stop for uniqueness check)."""
  count = [0]

  def solve(g):
    if count[0] >= limit:
      return
    for i in range(9):
      for j in range(9):
        if g[i][j] == 0:
          for num in range(1, 10):
            if _is_valid(g, i, j, num):
              g[i][j] = num
              solve(g)
              g[i][j] = 0
              if count[0] >= limit:
                return
          return
    count[0] += 1

  solve([row[:] for row in grid])
  return count[0]


def _remove_cells(grid, num_clues, rng):
  """Remove cells from a solved grid, ensuring a unique solution."""
  cells_to_remove = 81 - num_clues
  removed = 0
  all_cells = [(r, c) for r in range(9) for c in range(9)]
  rng.shuffle(all_cells)
  for row, col in all_cells:
    if removed >= cells_to_remove:
      break
    if grid[row][col] == 0:
      continue
    backup = grid[row][col]
    grid[row][col] = 0
    if _count_solutions(grid, limit=2) == 1:
      removed += 1
    else:
      grid[row][col] = backup
  return grid


def _generate_one(args):
  """Generate one (puzzle_grid, solution_grid) pair.

  Args:
    args: (seed, num_clues) tuple.

  Returns:
    (puzzle, solution) where each is a list of 9 lists of 9 ints.
  """
  seed, num_clues = args
  rng = random.Random(seed)
  grid = [[0] * 9 for _ in range(9)]
  _fill_grid(grid, rng)
  solution = [row[:] for row in grid]
  puzzle = _remove_cells(grid, num_clues, rng)
  return puzzle, solution


def _generate_raw_grids(num_needed, num_clues, seed,
                        num_workers):
  """Generate deduplicated (puzzle, solution) grid pairs."""
  all_puzzles = []
  all_solutions = []
  seen = set()
  task_seed = seed
  pbar = tqdm(total=num_needed, desc='Generating sudokus')

  while len(all_puzzles) < num_needed:
    remaining = num_needed - len(all_puzzles)
    batch_size = remaining + remaining // 10 + 16
    tasks = [(task_seed + i, num_clues)
             for i in range(batch_size)]
    task_seed += batch_size

    if num_workers > 1:
      pool = Pool(processes=num_workers)
      results = pool.imap(_generate_one, tasks)
    else:
      results = map(_generate_one, tasks)

    for puzzle, solution in results:
      key = tuple(c for row in solution for c in row)
      if key in seen:
        continue
      seen.add(key)
      all_puzzles.append(puzzle)
      all_solutions.append(solution)
      pbar.update(1)
      if len(all_puzzles) >= num_needed:
        break

    if num_workers > 1:
      pool.terminate()
      pool.join()

  pbar.close()
  return all_puzzles, all_solutions


def _tokenize_grids(puzzles, solutions, tokenizer):
  """Convert raw 9x9 grids into tokenized training examples.

  Each grid is flattened with row separators, then combined as:
    input_ids:      [BOS] puzzle(89) [BOS] solution(89)
    attention_mask: [0 ...... prompt ...... 0] [1 .. solution .. 1]
  """
  bos = tokenizer.bos_token_id
  sep = tokenizer.row_separator_id
  prompt_len = tokenizer.prompt_len
  seq_len = tokenizer.seq_len
  input_ids = []
  attention_masks = []
  for puzzle, solution in zip(puzzles, solutions):
    ids = [bos]
    for grid in (puzzle, solution):
      for i in range(9):
        ids.extend(grid[i])
        if i < 8:
          ids.append(sep)
      ids.append(bos)
    ids.pop()  # remove trailing BOS
    mask = [0] * prompt_len + [1] * seq_len
    input_ids.append(ids)
    attention_masks.append(mask)
  return {'input_ids': input_ids, 'attention_mask': attention_masks}


def generate_sudoku_dataset(num_train, num_valid, difficulty,
                            seed, tokenizer, num_workers=1):
  """Generate a deduplicated sudoku dataset.

  Args:
    num_train: Number of training examples.
    num_valid: Number of validation examples.
    difficulty: 'easy', 'medium', 'hard', or an int (num clues).
    seed: Base random seed for deterministic generation.
    tokenizer: SudokuTokenizer instance (single source of truth
      for token ids and sequence layout).
    num_workers: Number of parallel workers.

  Returns:
    dict with 'train' and 'validation' keys, each mapping to
    a dict with 'input_ids' and 'attention_mask' lists.
  """
  if difficulty not in DIFFICULTY_TO_CLUES:
    raise ValueError(
      f'Invalid difficulty: {difficulty!r}. '
      f'Must be one of {list(DIFFICULTY_TO_CLUES.keys())}.')
  num_clues = DIFFICULTY_TO_CLUES[difficulty]

  total_needed = num_train + num_valid
  all_puzzles, all_solutions = _generate_raw_grids(
    total_needed, num_clues, seed, num_workers)

  # Deterministic shuffle before splitting
  rng = random.Random(seed)
  indices = list(range(total_needed))
  rng.shuffle(indices)
  all_puzzles = [all_puzzles[i] for i in indices]
  all_solutions = [all_solutions[i] for i in indices]

  train = _tokenize_grids(
    all_puzzles[:num_train], all_solutions[:num_train],
    tokenizer)
  valid = _tokenize_grids(
    all_puzzles[num_train:], all_solutions[num_train:],
    tokenizer)
  return {'train': train, 'validation': valid}
