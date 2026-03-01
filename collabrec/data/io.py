# This file handles reading from and writing to CSV files.
# It provides functions to load ratings from a CSV, save ratings to a CSV, and generate a sample CSV for testing.




import csv
import os
from typing import List, Set, Tuple
from collabrec.data.exceptions import DataError
from collabrec.data.rating import RatingMatrix


# These are the column names that are recognised in the CSV header, in any case.
# Several common variations are allowed.
_USER_ALIASES = {"user_id", "user", "userid", "uid", "user id"}
_ITEM_ALIASES = {"item_id", "item", "itemid", "iid", "movie_id", "movie", "item id"}
_RATING_ALIASES = {"rating", "score", "value", "rate", "stars"}

def _detect_header(row: List[str]) -> bool:
    """
    Determine whether a row is likely a header.
    If any cell in the row matches a known column alias, the row is considered a header.
    """
    for cell in row:
        if cell.strip().lower() in (_USER_ALIASES | _ITEM_ALIASES | _RATING_ALIASES):
            return True
    return False

def _find_column_indices(header: List[str]) -> Tuple[int, int, int]:
    """
    Given a header row (a list of column names), find the indices of the user, item, and rating columns.
    Raises DataError if any required column cannot be found.
    """
    h = [c.strip().lower() for c in header]

    def find(aliases: Set[str]) -> int:
        for i, col in enumerate(h):
            if col in aliases:
                return i
        return -1

    user_col = find(_USER_ALIASES)
    item_col = find(_ITEM_ALIASES)
    rating_col = find(_RATING_ALIASES)

    if user_col == -1:
        raise DataError("Cannot find a user column. Expected one of: " + ", ".join(sorted(_USER_ALIASES)))
    if item_col == -1:
        raise DataError("Cannot find an item column. Expected one of: " + ", ".join(sorted(_ITEM_ALIASES)))
    if rating_col == -1:
        raise DataError("Cannot find a rating column. Expected one of: " + ", ".join(sorted(_RATING_ALIASES)))
    return user_col, item_col, rating_col

def load_ratings_csv(filepath: str, delimiter: str = ",", encoding: str = "utf-8") -> RatingMatrix:
    """
    Read a CSV file and load the ratings into a RatingMatrix.

    The file may or may not have a header row. If a header is present, the columns are identified by name.
    If no header is present, the columns are assumed to be in order: user_id, item_id, rating.

    Empty rows and rows with missing values are skipped. Rows that cannot be parsed are also skipped.

    Parameters:
        filepath: path to the CSV file
        delimiter: character that separates fields (default comma)
        encoding: file encoding (default utf-8)

    Returns:
        A RatingMatrix containing all valid ratings.

    Raises:
        DataError if the file cannot be opened, is empty, or contains no valid ratings.
    """
    if not os.path.isfile(filepath):
        raise DataError(f"File not found: {filepath!r}")

    matrix = RatingMatrix()
    skipped = 0
    loaded = 0
    user_col, item_col, rating_col = 0, 1, 2   # default positions for a file without a header
    has_header = False

    # Read all rows into memory (files are assumed to be small enough)
    try:
        with open(filepath, newline="", encoding=encoding) as fh:
            reader = csv.reader(fh, delimiter=delimiter)
            rows = list(reader)
    except (OSError, UnicodeDecodeError) as exc:
        raise DataError(f"Cannot open {filepath!r}: {exc}") from exc

    if not rows:
        raise DataError(f"File is empty: {filepath!r}")

    first_row = rows[0]
    if _detect_header(first_row):
        has_header = True
        user_col, item_col, rating_col = _find_column_indices(first_row)
        data_rows = rows[1:]
    else:
        # No header – require at least three columns
        if len(first_row) < 3:
            raise DataError("CSV has fewer than 3 columns and no recognisable header. Expected: user_id, item_id, rating")
        data_rows = rows

    # Process each data row
    for lineno, row in enumerate(data_rows, start=2 if has_header else 1):
        # Skip empty lines
        if not row or all(cell.strip() == "" for cell in row):
            continue
        try:
            uid = row[user_col].strip()
            iid = row[item_col].strip()
            raw = row[rating_col].strip()
            if not uid or not iid or not raw:
                skipped += 1
                continue
            r = float(raw)
            matrix.add_rating(uid, iid, r)
            loaded += 1
        except (IndexError, ValueError):
            skipped += 1

    if loaded == 0:
        raise DataError(f"No valid ratings found in {filepath!r}. ({skipped} rows were skipped due to parsing errors)")

    return matrix

def save_ratings_csv(matrix: RatingMatrix, filepath: str, delimiter: str = ",", encoding: str = "utf-8") -> None:
    """
    Write a RatingMatrix to a CSV file.
    A header row with "user_id", "item_id", "rating" is always included.
    """
    with open(filepath, "w", newline="", encoding=encoding) as fh:
        writer = csv.writer(fh, delimiter=delimiter)
        writer.writerow(["user_id", "item_id", "rating"])
        for rating in matrix.iter_ratings():
            writer.writerow([rating.user_id, rating.item_id, rating.rating])

def generate_sample_csv(filepath: str) -> None:
    """
    Create a small sample CSV file with fictional ratings.
    This is useful for testing the program without real data.
    The data is deterministic (the same every time).
    """
    rows = [
        ("alice",   "inception",     5.0), ("alice",   "matrix",        4.0),
        ("alice",   "interstellar",  5.0), ("alice",   "avatar",        3.0),
        ("alice",   "godfather",     4.5), ("alice",   "goodfellas",    4.0),
        ("bob",     "inception",     4.0), ("bob",     "matrix",        5.0),
        ("bob",     "interstellar",  4.5), ("bob",     "avatar",        2.0),
        ("bob",     "dark_knight",   5.0), ("bob",     "pulp_fiction",  4.0),
        ("carol",   "inception",     3.5), ("carol",   "matrix",        3.0),
        ("carol",   "avatar",        4.0), ("carol",   "dark_knight",   4.5),
        ("carol",   "goodfellas",    3.0), ("carol",   "schindler",     5.0),
        ("dave",    "inception",     2.0), ("dave",    "interstellar",  3.0),
        ("dave",    "avatar",        5.0), ("dave",    "forrest_gump",  4.0),
        ("dave",    "schindler",     4.5), ("dave",    "pulp_fiction",  3.5),
        ("eve",     "matrix",        4.5), ("eve",     "dark_knight",   5.0),
        ("eve",     "pulp_fiction",  4.5), ("eve",     "goodfellas",    4.0),
        ("eve",     "schindler",     4.5), ("eve",     "forrest_gump",  3.0),
        ("frank",   "inception",     4.0), ("frank",   "matrix",        3.5),
        ("frank",   "interstellar",  4.5), ("frank",   "dark_knight",   4.0),
        ("frank",   "godfather",     5.0), ("frank",   "schindler",     4.5),
        ("grace",   "inception",     5.0), ("grace",   "avatar",        3.0),
        ("grace",   "godfather",     4.5), ("grace",   "goodfellas",    4.0),
        ("grace",   "forrest_gump",  4.5), ("grace",   "schindler",     3.5),
        ("henry",   "matrix",        3.0), ("henry",   "interstellar",  4.0),
        ("henry",   "dark_knight",   3.5), ("henry",   "pulp_fiction",  4.0),
        ("henry",   "godfather",     4.5), ("henry",   "forrest_gump",  5.0),
    ]
    with open(filepath, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["user_id", "item_id", "rating"])
        writer.writerows(rows)