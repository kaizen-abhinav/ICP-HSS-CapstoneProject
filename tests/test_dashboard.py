import csv

import main


def test_dashboard_initializes_without_show():
    """The dashboard should build without launching a window when show=False."""
    result = main.run_dashboard(seed=123, show=False, verbose=False)

    assert result['plan']['path'], 'Planner returned an empty path unexpectedly.'
    assert result['figure'] is not None
    assert callable(result['update'])
    assert isinstance(result['controls'], dict)
    assert result['telemetry'], 'Telemetry records should capture the simulated mission.'

    # Exercise the update hook for a single frame to ensure no runtime errors.
    result['update'](0)


def test_dashboard_writes_telemetry_log(tmp_path):
    log_file = tmp_path / 'telemetry.csv'

    result = main.run_dashboard(seed=321, show=False, verbose=False, log_path=str(log_file))

    assert log_file.exists(), 'Telemetry CSV was not created.'

    with log_file.open(newline='') as csv_file:
        rows = list(csv.reader(csv_file))

    assert rows, 'Telemetry CSV should contain at least a header row.'
    assert rows[0][0] == 'time', 'Telemetry CSV header mismatch.'
    assert len(result['telemetry']) >= 1
