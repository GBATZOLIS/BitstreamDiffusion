import tempfile, torch
from ml_collections import config_dict
from data.sudoku import (
    SudokuDataset, build_or_load_sudoku, PROMPT_LEN_TOKENS, TOTAL_LEN_TOKENS,
    BOS_TOKEN_ID, ROW_SEPARATOR_ID, BITS_PER_TOKEN,
)
from data.task_codec import bits_to_token_ids


def _valid_solution(grid81):
    g = [grid81[i*9:(i+1)*9] for i in range(9)]
    rng = set(range(1, 10))
    for i in range(9):
        if set(g[i]) != rng: return False
        if set(row[i] for row in g) != rng: return False
    for br in range(0,9,3):
        for bc in range(0,9,3):
            box = [g[br+i][bc+j] for i in range(3) for j in range(3)]
            if set(box) != rng: return False
    return True


def _extract_grid(ids180, start):
    # tokens start..start+88 are a grid: 9 rows of 9, separators between.
    cells = []
    i = start
    for r in range(9):
        cells.extend(ids180[i:i+9]); i += 9
        if r < 8:
            assert ids180[i] == ROW_SEPARATOR_ID, f"expected sep at {i}, got {ids180[i]}"
            i += 1
    return cells


def test_sudoku_format_and_validity():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = config_dict.ConfigDict()
        cfg.data = config_dict.ConfigDict()
        cfg.data.difficulty = "easy"
        cfg.data.num_train = 16
        cfg.data.num_valid = 4
        cfg.data.data_seed = 42
        cfg.data.bits_per_token = 4
        cfg.data.root = tmp
        cfg.data.sudoku_num_workers = 1
        cfg.data.num_workers = 0
        cfg.train = config_dict.ConfigDict(); cfg.train.batch_size = 4

        ds = SudokuDataset(cfg, split="train")
        assert len(ds) == 16
        ex = ds[0]
        assert ex["x0"].shape == (TOTAL_LEN_TOKENS * BITS_PER_TOKEN,)
        assert ex["input_ids"].shape == (TOTAL_LEN_TOKENS,)
        assert ex["prefix_mask"].dtype == torch.bool
        assert int(ex["prefix_mask"].sum()) == PROMPT_LEN_TOKENS * BITS_PER_TOKEN

        ids = ex["input_ids"].tolist()
        assert ids[0] == BOS_TOKEN_ID
        assert ids[PROMPT_LEN_TOKENS - 1] == BOS_TOKEN_ID  # index 90

        # x0 decodes back to the same ids (codec consistency through dataset).
        dec = bits_to_token_ids(ex["x0"], BITS_PER_TOKEN)
        assert torch.equal(dec, ex["input_ids"])

        # Solution (positions 91..179) is a valid completed sudoku.
        sol = _extract_grid(ids, PROMPT_LEN_TOKENS)
        assert len(sol) == 81
        assert _valid_solution(sol), "solution grid is not a valid sudoku"

        # Puzzle clues are consistent with the solution (non-empty puzzle cells match).
        puz = _extract_grid(ids, 1)
        clues = sum(1 for c in puz if c != 0)
        assert clues == 40, f"easy should have 40 clues, got {clues}"
        for p, s in zip(puz, sol):
            if p != 0:
                assert p == s


if __name__ == "__main__":
    test_sudoku_format_and_validity()
    print("ALL SUDOKU FORMAT TESTS PASSED")
