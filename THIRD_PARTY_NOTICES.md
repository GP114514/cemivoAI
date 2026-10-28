# Third-party notices

The project source license does not replace the licenses or service terms of its dependencies. Dependencies are installed separately; their source distributions and notices remain authoritative.

## Runtime and authoring dependencies

- **PyTorch**: neural models, training and inference. The repository pins the CUDA build used in development. CUDA drivers/toolkits are provided by their respective distributors.
- **NumPy** and **Pillow**: arrays, training targets, image loading and pixel verification.
- **Node.js**, **@napi-rs/canvas** and **FFmpeg**: film composition, image/font rendering and audiovisual encoding. FFmpeg builds may have different licensing configurations; no FFmpeg binary is bundled here.
- **edge-tts** and **srt**: optional narration regeneration. These packages are needed only when synthesizing new speech, not for playing the delivered film or re-rendering its retained audio.

## External AI planning

The APP can call a user-configured DeepSeek-compatible service. DeepSeek model weights are not bundled or licensed by this repository. The user's own provider account, API key, model availability and service terms apply. Screens and interaction context sent to that provider leave the local APP process.

## Fonts

Microsoft YaHei, Segoe UI, Consolas and SimSun are referenced as installed font paths by some dataset, auxiliary-training and film scripts. No Microsoft font files are included. Noto/DejaVu fallback paths in the main dataset generator also refer to separately installed fonts. Use fonts licensed for your environment and retain the notices accompanying their distribution. Replacing a font changes training targets and may require model revalidation.

## Demonstration narration and music

The English film's retained narration was generated with Microsoft's generic `en-US-AndrewNeural` online voice through `edge-tts` 7.3.1. Its metadata explicitly records `online=true` and `offline=false`; no voice cloning or user's API key was used. Microsoft speech model weights are not included. The project's source license grants no rights in Microsoft's voice models, trademarks or service implementation; applicable service terms remain separate.

The music and completion sounds are original procedural JavaScript synthesis in `film/make-audio.cjs`, without third-party samples or a third-party music track. The original model PNGs in the demonstration were recorded from the local cemivoAI APP. Evidence hashes and the bounded recorded checks accompany them in `film/trace.json` and `film/audit.json`.
