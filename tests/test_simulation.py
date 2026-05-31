from theo.simulation import (
    DEFENSE_LIBRARY,
    OFFENSE_LIBRARY,
    Simulator,
    make_offense_play,
    rank_defenses,
    rank_offenses,
    simulate_play,
)

VALID = {"complete", "incomplete", "sack", "interception", "run"}
N_PLAYERS = 22  # 6 Skill + 5 OL + 11 Defense


def test_single_play_structure():
    off = OFFENSE_LIBRARY["slant_flat"]
    deff = DEFENSE_LIBRARY["cover2_zone"]
    res = simulate_play(off, deff, seed=0)
    assert res.outcome in VALID
    assert res.frames, "Es sollten Trajektorien aufgezeichnet werden."
    assert len(res.ball) == len(res.frames)
    for step in res.frames:
        assert len(step) == N_PLAYERS


def test_deterministic_with_seed():
    off, deff = OFFENSE_LIBRARY["mesh"], DEFENSE_LIBRARY["cover1_man"]
    a = simulate_play(off, deff, seed=42)
    b = simulate_play(off, deff, seed=42)
    assert (a.outcome, round(a.yards, 3)) == (b.outcome, round(b.yards, 3))


def test_run_play_outcome():
    res = simulate_play(OFFENSE_LIBRARY["inside_zone"], DEFENSE_LIBRARY["cover2_zone"],
                        seed=3)
    assert res.outcome == "run"
    assert res.target == "RB"


def test_distribution_counts():
    sim = Simulator()
    dist = sim.simulate_many(OFFENSE_LIBRARY["four_verticals"],
                             DEFENSE_LIBRARY["cover2_zone"], n=120)
    assert dist.n == 120
    assert sum(dist.outcomes.values()) == 120
    assert all(o in VALID for o in dist.outcomes)


def test_smash_beats_cover2_more_than_cover3():
    # Football-Sanity: Smash ist ein Cover-2-Beater.
    sim = Simulator()
    c2 = sim.simulate_many(OFFENSE_LIBRARY["smash"], DEFENSE_LIBRARY["cover2_zone"], n=200)
    c3 = sim.simulate_many(OFFENSE_LIBRARY["smash"], DEFENSE_LIBRARY["cover3_zone"], n=200)
    assert c2.mean_yards > c3.mean_yards


def test_blitz_creates_some_sacks_on_deep_play():
    sim = Simulator()
    d = sim.simulate_many(OFFENSE_LIBRARY["four_verticals"],
                          DEFENSE_LIBRARY["man_blitz"], n=200)
    assert d.outcomes.get("sack", 0) > 0


def test_custom_play_roundtrip():
    custom = make_offense_play(
        {"WR_L": "post", "WR_R": "go", "SLOT": "slant", "TE": "out", "RB": "flat"})
    res = simulate_play(custom, DEFENSE_LIBRARY["cover3_zone"], seed=1)
    assert res.outcome in VALID
    assert custom.routes["WR_L"] == "post"


def test_cover4_runs():
    res = simulate_play(OFFENSE_LIBRARY["four_verticals"],
                        DEFENSE_LIBRARY["cover4_quarters"], seed=1)
    assert res.outcome in VALID


def test_rank_defenses_sorted_ascending():
    rows = rank_defenses(OFFENSE_LIBRARY["four_verticals"], n=40)
    assert len(rows) == len(DEFENSE_LIBRARY)
    ys = [r["mean_yards"] for r in rows]
    assert ys == sorted(ys)  # beste (wenigste Yards) zuerst


def test_rank_offenses_sorted_descending():
    rows = rank_offenses(DEFENSE_LIBRARY["cover2_zone"], n=40)
    assert len(rows) == len(OFFENSE_LIBRARY)
    ys = [r["mean_yards"] for r in rows]
    assert ys == sorted(ys, reverse=True)
