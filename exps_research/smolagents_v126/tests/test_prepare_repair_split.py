from scripts.repair.prepare_repair_split import (
    extract_last_boxed,
    question_hash,
    stratified_select,
)


def _rows(per_stratum=5):
    rows = []
    for level in (2, 3):
        for subject in ("Algebra", "Geometry"):
            for index in range(per_stratum):
                question = f"L{level} {subject} {index}"
                rows.append(
                    {
                        "question": question,
                        "answer": str(index),
                        "level": level,
                        "type": subject,
                        "source_config": subject.lower(),
                        "source_index": index,
                        "source_question_sha256": question_hash(question),
                    }
                )
    return rows


def test_extract_last_boxed_handles_nested_braces_and_last_answer():
    solution = r"First \boxed{1}, finally \boxed{\frac{2}{3}}."
    assert extract_last_boxed(solution) == r"\frac{2}{3}"
    assert extract_last_boxed(r"Thus the answer is $\boxed 2$.") == "2"


def test_stratified_selection_is_deterministic_balanced_and_disjoint():
    rows = _rows()
    first = stratified_select(rows, {2: 4, 3: 4}, seed=42)
    second = stratified_select(rows, {2: 4, 3: 4}, seed=42)
    assert [row["question"] for row in first] == [row["question"] for row in second]

    for level in (2, 3):
        subjects = [row["type"] for row in first if row["level"] == level]
        assert subjects.count("Algebra") == 2
        assert subjects.count("Geometry") == 2

    reserved = {row["source_question_sha256"] for row in first}
    third = stratified_select(rows, {2: 2, 3: 2}, seed=43, reserved_hashes=reserved)
    assert not reserved.intersection(row["source_question_sha256"] for row in third)
