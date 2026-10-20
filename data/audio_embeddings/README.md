# AST embeddings

The AST CSVs (`AST_ESC_CSV`, `AST_URBAN_CSV` in `scripts/reproduce_paper.sh --list-extra`) come from
`extract_ast_embeddings.py`: `pooler_output` of `MIT/ast-finetuned-audioset-10-10-0.4593` pinned at
revision `f826b80`, on 16 kHz mono audio, written as feature columns `0`..`767` next to the rows, folds
and labels of the EfficientNet CSV it is given.

```bash
python data/audio_embeddings/extract_ast_embeddings.py /path/ESC-50/audio /path/esc-50.csv /path/ast_768/esc-50.csv
python data/audio_embeddings/extract_ast_embeddings.py /path/UrbanSound8K/audio /path/urbansound8k.csv /path/ast_768/urbansound8k.csv
```

It needs `transformers`, `torchaudio` and `soundfile`. Quinn has no AST embeddings because its audio is
not public.
