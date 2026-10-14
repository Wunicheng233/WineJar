from xarm_sdk_control.pair_calibration_session import PendingSample


def test_request_waits_for_return_of_hands_and_a_qualified_pair_then_saves_once():
    request = PendingSample()
    assert not request.update(ord('s'), 0, True)[0]
    assert not request.update(-1, 1, True)[0]
    for now in (2, 3, 4):
        assert not request.update(-1, now, False)[0]
    assert request.update(-1, 4.1, True)[0]
    assert not request.update(-1, 4.2, True)[0]


def test_second_press_cancels_even_when_current_pair_is_ready():
    request = PendingSample()
    request.update(ord('s'), 0, False)
    assert not request.update(ord('S'), 3, True)[0]
    assert request.requested_at is None
    assert not request.update(-1, 4, True)[0]


def test_timeout_and_solve_do_not_save_an_unexpected_sample():
    request = PendingSample()
    request.update(ord('s'), 0, False)
    save, message = request.update(-1, 30, True)
    assert not save and 'timed out' in message
    assert not request.update(-1, 31, True)[0]
    request.update(ord('s'), 32, False)
    assert not request.update(ord('c'), 35, True)[0]
    assert request.requested_at is None
