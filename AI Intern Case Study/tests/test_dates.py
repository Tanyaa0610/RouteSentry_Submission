from datetime import date

from routesentry.dates import monday_of, ranges_overlap, sunday_of


def test_monday_of_on_a_monday_is_itself():
    d = date(2024, 1, 1)  # a Monday
    assert monday_of(d) == d


def test_monday_of_on_a_sunday_goes_back_six_days():
    d = date(2024, 1, 7)  # Sunday ending the week starting Jan 1
    assert monday_of(d) == date(2024, 1, 1)


def test_monday_of_on_a_wednesday():
    d = date(2024, 1, 3)
    assert monday_of(d) == date(2024, 1, 1)


def test_sunday_of_matches_monday_plus_six():
    d = date(2024, 1, 4)
    assert sunday_of(d) == date(2024, 1, 7)


def test_week_boundary_jan1_vs_dec31():
    # Dec 31 2023 was a Sunday; Jan 1 2024 a Monday -- they must NOT share a week.
    assert monday_of(date(2023, 12, 31)) != monday_of(date(2024, 1, 1))
    assert monday_of(date(2023, 12, 31)) == date(2023, 12, 25)


def test_ranges_overlap_true_when_touching():
    a = (date(2024, 1, 1), date(2024, 1, 7))
    b = (date(2024, 1, 7), date(2024, 1, 14))
    assert ranges_overlap(*a, *b) is True


def test_ranges_overlap_false_when_disjoint():
    a = (date(2024, 1, 1), date(2024, 1, 7))
    b = (date(2024, 1, 8), date(2024, 1, 14))
    assert ranges_overlap(*a, *b) is False


def test_ranges_overlap_contained():
    a = (date(2024, 1, 1), date(2024, 1, 31))
    b = (date(2024, 1, 10), date(2024, 1, 12))
    assert ranges_overlap(*a, *b) is True
