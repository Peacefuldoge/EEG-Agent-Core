from eeg_agent import EEGTools, EEGSession, STRATEGIES, channel_config_path

def test_core_is_headless_and_channel_configs_packaged():
    assert callable(EEGTools) and callable(EEGSession)
    assert set(STRATEGIES) == {"react", "plan-execute", "rewoo"}
    for name in ("standard_26", "standard_64", "hydrocel_to_26", "hydrocel_to_64"):
        assert channel_config_path(name).is_file()

def test_core_does_not_import_ui():
    import sys
    assert 'chainlit' not in sys.modules
    assert 'eeg_agent_ui' not in sys.modules
