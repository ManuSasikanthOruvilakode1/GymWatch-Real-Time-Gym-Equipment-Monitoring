# Dataset (not included)

The training photos are private and self-collected, so they are not in this repository.
Put your own images here, one folder per class. The folder name is the label:

```
data/
  IN-USE-LEG-EXTENSION/       person seated and working the leg-extension machine
  NOT-IN-USE-LEG-EXTENSION/   machine empty, or people only passing / standing nearby
  IN-USE-PEC-FLY/             person seated and working the pec-fly machine
  NOT-IN-USE-PEC-FLY/         machine empty, or people only passing / standing nearby
```

- Formats: `.jpg`, `.jpeg` or `.png`. Phone photos are fine; EXIF orientation is applied.
- Capture time is read from a `YYYY-MM-DD HH.MM.SS` file name if present, otherwise from the
  photo's EXIF data. It is used to keep near-identical burst shots on the same side of the
  train/test split, so keep EXIF data when copying photos.
- Then run `uv run gymwatch prepare`.
