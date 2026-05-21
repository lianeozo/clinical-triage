"""Lock in encode/decode + dimension-order invariants for Action."""
from sepsisSimDiabetes.Action import Action


def test_num_per_action_dim_labels_match_encode():
    """NUM_PER_ACTION[i] must correspond to the i-th term in get_action_idx's
    mixed-radix encoding (most significant first). The encoding is
    antibiotic * 16 + ventilation * 8 + vasopressors * 4 + soc, so the
    expected label order is [ANTIB, VENT, VASO, SOC]."""
    # We assert this via a class-level constant Action.PER_ACTION_LABELS that
    # the fix introduces, paired with NUM_PER_ACTION of matching length.
    labels = Action.PER_ACTION_LABELS
    assert labels == (Action.ANTIBIOTIC_STRING,
                      Action.VENT_STRING,
                      Action.VASO_STRING,
                      Action.SOC_STRING)
    assert len(labels) == len(Action.NUM_PER_ACTION)


def test_action_idx_roundtrip_components():
    for idx in range(Action.NUM_ACTIONS_TOTAL):
        a = Action(action_idx=idx)
        assert a.get_action_idx() == idx


def test_selected_actions_roundtrip_preserves_soc():
    # Constructing from selected_actions then converting back must preserve SOC.
    a_orig = Action(action_idx=27)  # antib=1, vent=1, vaso=0, soc=3
    sel = a_orig.get_selected_actions()
    a_round = Action(selected_actions=sel)
    assert a_round.get_action_idx() == 27
    assert a_round.soc == 3


def test_action_27_decodes_to_antib_vent_icu():
    # Spot-check the decode of an action used in Phase 3 findings.
    a = Action(action_idx=27)
    assert (a.antibiotic, a.ventilation, a.vasopressors, a.soc) == (1, 1, 0, 3)
