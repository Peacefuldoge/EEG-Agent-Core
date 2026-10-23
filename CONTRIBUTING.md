# Contributing

1. Open an issue describing a reproducible EEG or agent failure. Avoid uploading identifiable EEG or private patient metadata.
2. Keep scientific computation independent of LLM output: the EEG engine must run without model access.
3. Add a regression test for changes to channel alignment, window slicing, artifacts or graph reference resolution.
4. Run `python -m pytest -q tests/` and document any tests requiring a live Qwen/vLLM endpoint separately.
5. Do not add shell-execution tools or disable path allowlists to make a demo pass.
6. Follow existing Python naming and Pydantic schemas for new tools. Changes to the strategy interface must be checked across all three strategies.

**Research note:** For real EEG datasets document electrodes, units, reference, sampling, filtering, exclusions and subject-disjoint split policy. Confirm license/redistribution rights for any contributed data or mapping files.
