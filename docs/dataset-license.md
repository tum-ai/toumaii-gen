# Dataset license notes for Hugging Face releases

Use the text below in the dataset card of each TOUMAII release:

- `tum-ai/toumaii-toucan-kimi`
- `tum-ai/toumaii-toucan-oss`
- `tum-ai/toumaii-toucan-qwen`
- `tum-ai/toumaii-toolace`

## Front matter

```yaml
license: cc-by-sa-4.0
license_name: cc-by-sa-4.0
license_link: https://creativecommons.org/licenses/by-sa/4.0/
```

## Card section

```markdown
## License

Copyright (C) 2026 Moritz Pötzsch, Andrea Gentilini, Lisa Rossgoderer, Tom Osterfeld,
Kesava Prasad Sanjivi Arul, and Swanand Ravindra Kadhe.

The preference pairs, reference scores, and generated alternatives contributed by
TOUMAII-Gen are licensed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). See `LICENSE-DATA`
in the [code repository](https://github.com/tum-ai/toumaii-gen) for the full text.

Rows copied from or derived from the upstream datasets remain under
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0):

- [Agent-Ark/Toucan-1.5M](https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M)
- [Team-ACE/ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE)

Modifications relative to the upstream data: records are converted to a shared
canonical format, an alternative response is generated for each demonstration, both
responses are scored, and pairs are filtered by score. See the `NOTICE` file in the
code repository for the full statement of changes and attribution.

### ShareAlike scope

Filtered, relabeled, augmented, or re-curated versions of this dataset must be
released under CC BY-SA 4.0. Training, fine-tuning, or evaluating a model on this
dataset does not by itself extend the ShareAlike obligation to the resulting model
weights or model outputs; we additionally permit such use as licensor of the TOUMAII
contributions.
```

## Notes

- Keep the `license: cc-by-sa-4.0` tag accurate. The Apache-2.0 portions are not
  relicensed by the TOUMAII license, and downstream users keep the rights Apache-2.0
  grants them for those rows.
- The generated alternatives come from Kimi-K2, Qwen, and OSS models. The upstream
  Toucan-1.5M release already redistributes outputs from these models under Apache-2.0.
  Re-check the relevant model terms before generating alternatives with a new model.
- Hugging Face's license filter treats CC BY-SA 4.0 as commercially usable, but some
  enterprise compliance lists exclude share-alike licenses. The training clarification
  above exists to reduce that friction.
