# Project rename: AI Pilot

Recorded 11 October 2026.

AI Pilot is an independent research playground. Its GitHub repository is
[fssrepository/ai-pilot](https://github.com/fssrepository/ai-pilot), repository ID
1413986284. The repository was renamed in place; its history was retained.
The local checkout is `/home/raxim/workspace/ai-pilot`, and `origin` points to
`https://github.com/fssrepository/ai-pilot.git`. Re-cloning and re-downloading
models are unnecessary. Reopen this directory if an editor still targets the
previous location.

## Local configuration

- Model weights, experiment outputs, `.tools` and `.venv` moved with the checkout.
- Virtual-environment entry points and Jupyter kernel paths were updated.
  The notebook kernel is `ai-pilot-qwen`, displayed as **AI Pilot Qwen (CPU)**.
- The adaptive C++ executable was rebuilt because its library search paths and
  llama.cpp backend directory include the checkout's absolute path.
- Existing launchers resolve the repository root relative to their own files.
  Their commands and ports remain documented in the root README.
- The adaptive chat and the two measurement viewers were stopped for relocation
  and restarted manually. Qwen/Jupyter remains manually started on demand.
  No boot/login service or autostart entry was added.

## Documents and provenance

Current PDF covers use plain **AI Pilot** text. The former logo asset was removed,
and the shared PDF builder, cover metadata, notebooks and HTML titles use the
new project identity. All four current PDF reports were regenerated from their
existing sources; this did not retrain any model.

Historical verification manifests retain their original hashes, paths and
measurement descriptions, including references to the former logo. They describe
past runs, not the current file layout. Archived trained model metadata likewise
remains untouched. New exports use the current name. External tool paths that
happen to refer to a separate repository are not project branding.

Qwen and adaptive Llama retain separate implementation commits. Repair Graph
remains an independent experiment directory and uses its own checkpoint format.
